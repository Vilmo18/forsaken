import pandas as pd
import numpy as np
import torch
import random
import wandb
import os
from datasets import Dataset, concatenate_datasets
from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM,
                          DataCollatorForSeq2Seq, TrainingArguments, Trainer)
from peft import LoraConfig, get_peft_model, TaskType
from pathlib import Path
import argparse

DEFAULT_MODEL_ID = "facebook/nllb-200-distilled-600M"

def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class BilingualFineTuner:
    def __init__(self, language_name, language_code, model_id=DEFAULT_MODEL_ID, tracker=None):
        """
        Initialize the fine-tuner for a specific language
        
        Args:
            language_name: Name of the language (e.g., 'Tem', 'Ewe', 'Mina')
            language_code: Language code for NLLB (e.g., 'tem_Latn', 'ewe_Latn')
            model_id: Base model to fine-tune
        """
        self.language_name = language_name
        self.language_code = language_code
        self.model_id = model_id
        self.tokenizer = None
        self.model = None
        self.tracker=tracker
        self.use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        self.compute_dtype = torch.bfloat16 if self.use_bf16 else torch.float16
        
    def load_data(self, train_t2f_path, val_t2f_path, train_f2t_path, val_f2t_path):
        """Load the cleaned JSONL datasets"""
        print(f"Loading {self.language_name}-French datasets...")
        # 🔍 DEBUG: verify what is actually being passed
        for p in [train_t2f_path, val_t2f_path, train_f2t_path, val_f2t_path]:
          print(p, os.path.exists(p), os.path.getsize(p) if os.path.exists(p) else "MISSING")
          
        train_t2f = pd.read_json(train_t2f_path, lines=True)
        val_t2f   = pd.read_json(val_t2f_path, lines=True)
        train_f2t = pd.read_json(train_f2t_path, lines=True)
        val_f2t   = pd.read_json(val_f2t_path, lines=True)

        print(f"{self.language_name}-French Dataset sizes:")
        print(f"T2F: Train={len(train_t2f)}, Val={len(val_t2f)}")
        print(f"F2T: Train={len(train_f2t)}, Val={len(val_f2t)}")

        train_t2f = Dataset.from_pandas(train_t2f, preserve_index=False)
        val_t2f   = Dataset.from_pandas(val_t2f, preserve_index=False)
        train_f2t = Dataset.from_pandas(train_f2t, preserve_index=False)
        val_f2t   = Dataset.from_pandas(val_f2t, preserve_index=False)
        
        return train_t2f, val_t2f, train_f2t, val_f2t
    
    def setup_model_and_tokenizer(self):
        """Initialize the model and tokenizer"""
        print(f"Loading model and tokenizer: {self.model_id}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        
        # Add language token if it doesn't exist
        lang_token = self.language_code
        token_added = lang_token not in self.tokenizer.get_vocab()
        if token_added:
            self.tokenizer.add_tokens([lang_token], special_tokens=True)
            print(f"Added new token: {lang_token}")
        else:
            print(f"Token {lang_token} already exists")

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_id,
            device_map={"": 0},
            torch_dtype=self.compute_dtype
        )
        print(f"Training precision: {self.compute_dtype}")

        # Resize embeddings if new token was added
        if token_added:
            self.model.resize_token_embeddings(len(self.tokenizer))
            print(f"Resized embeddings to {len(self.tokenizer)} tokens")
            
    def setup_lora(self, r=32, lora_alpha=64, lora_dropout=0.1):
        """Configure LoRA for efficient fine-tuning"""
        lora_config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"],
            task_type=TaskType.SEQ_2_SEQ_LM,
            bias="none"
        )

        self.model = get_peft_model(self.model, lora_config)
        self.model.config.use_cache = False
        self.model.config.gradient_checkpointing = False
        
        self.model.print_trainable_parameters()
        
    def tokenize_data(
        self,
        train_t2f,
        val_t2f,
        train_f2t,
        val_f2t,
        max_length=256,
        balance_directions=True,
    ):
        """Tokenize the datasets for both translation directions"""
        
        def map_lang_to_french(batch):
            """Tokenize Language->French pairs"""
            self.tokenizer.src_lang = self.language_code
            self.tokenizer.tgt_lang = "fra_Latn"
            
            inputs = self.tokenizer(batch[self.language_name], truncation=True, max_length=max_length, padding=False)
            targets = self.tokenizer(text_target=batch["French"], truncation=True, max_length=max_length, padding=False)
            
            inputs["labels"] = targets["input_ids"]
            return inputs

        def map_french_to_lang(batch):
            """Tokenize French->Language pairs"""
            self.tokenizer.src_lang = "fra_Latn"
            self.tokenizer.tgt_lang = self.language_code
            
            inputs = self.tokenizer(batch["French"], truncation=True, max_length=max_length, padding=False)
            targets = self.tokenizer(text_target=batch[self.language_name], truncation=True, max_length=max_length, padding=False)
            
            inputs["labels"] = targets["input_ids"]
            return inputs

        print(f"Tokenizing {self.language_name}->French data...")
        tok_train_t2f = train_t2f.map(map_lang_to_french, batched=True, remove_columns=train_t2f.column_names)
        tok_val_t2f   = val_t2f.map(map_lang_to_french, batched=True, remove_columns=val_t2f.column_names)

        print(f"Tokenizing French->{self.language_name} data...")
        tok_train_f2t = train_f2t.map(map_french_to_lang, batched=True, remove_columns=train_f2t.column_names)
        tok_val_f2t   = val_f2t.map(map_french_to_lang, batched=True, remove_columns=val_f2t.column_names)

        if balance_directions:
            target_size = max(len(tok_train_t2f), len(tok_train_f2t))

            def upsample(dataset, seed):
                if len(dataset) == 0:
                    raise ValueError("Cannot balance an empty translation dataset")
                if len(dataset) == target_size:
                    return dataset

                indices = list(range(len(dataset)))
                rng = random.Random(seed)
                while len(indices) < target_size:
                    extra = list(range(len(dataset)))
                    rng.shuffle(extra)
                    indices.extend(extra[:target_size - len(indices)])
                return dataset.select(indices)

            before_t2f = len(tok_train_t2f)
            before_f2t = len(tok_train_f2t)
            tok_train_t2f = upsample(tok_train_t2f, seed=42)
            tok_train_f2t = upsample(tok_train_f2t, seed=43)
            print(
                "Balanced training directions: "
                f"T2F {before_t2f}->{len(tok_train_t2f)}, "
                f"F2T {before_f2t}->{len(tok_train_f2t)}"
            )

        train_ds = concatenate_datasets([tok_train_t2f, tok_train_f2t]).shuffle(seed=42)
        eval_ds  = concatenate_datasets([tok_val_t2f, tok_val_f2t]).shuffle(seed=42)

        print(f"Combined {self.language_name}-French dataset sizes: Train={len(train_ds)}, Eval={len(eval_ds)}")
        
        return train_ds, eval_ds
    
    def train(self, train_dataset, eval_dataset, output_dir, **training_args):
        """Run the training process"""
        
        collator = DataCollatorForSeq2Seq(
            self.tokenizer,
            model=self.model,
            padding=True,
            pad_to_multiple_of=8,
        )
        
        default_args = {
            "output_dir": output_dir,
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 8,
            "learning_rate": 3e-5,
            "num_train_epochs": 8,
            "bf16": self.use_bf16,
            "fp16": torch.cuda.is_available() and not self.use_bf16,
            "logging_steps": 50,
            "save_strategy": "epoch",
            "eval_strategy": "epoch",
            "report_to": "wandb",
            "run_name": f"{self.language_name}-nllb-lora",
            "seed": 42,
            "gradient_checkpointing": False,
            "warmup_steps": 200,
        }
        
        # Update with any custom arguments
        default_args.update(training_args)
        
        args = TrainingArguments(**default_args)

        trainer = Trainer(
            model=self.model,
            args=args,
            data_collator=collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )

        print(f"Starting {self.language_name}-French bidirectional training...")
        trainer.train()
        log_history = trainer.state.log_history

        training_result = {
            "log_history": log_history,
            "final_metrics": log_history[-1] if log_history else {},
            "num_train_samples": len(train_dataset),
            "num_eval_samples": len(eval_dataset),
        }

        if self.tracker:
            self.tracker.log_training(training_result)
        return trainer
    
    def save_models(self, adapter_dir, merged_dir):
        """Save the trained models"""
        print("Saving models")
        
        # Save LoRA adapter
        lora_dir = Path(adapter_dir)
        self.model.save_pretrained(lora_dir)
        self.tokenizer.save_pretrained(lora_dir)

        # Save merged model
        print("Merging and saving full model")
        merged = self.model.merge_and_unload()
        merged_path = Path(merged_dir)
        merged.save_pretrained(merged_path, safe_serialization=True)
        self.tokenizer.save_pretrained(merged_path)
        
        print(f"Models saved to {lora_dir} and {merged_dir}")

def main():
    """Main function to run the fine-tuning process"""
    parser = argparse.ArgumentParser(description="Fine-tune NLLB model for bilingual translation")
    parser.add_argument("--language", required=True, help="Name of the language (e.g., Tem, Ewe, Mina)")
    parser.add_argument("--code", required=True, help="NLLB language code (e.g., tem_Latn, ewe_Latn)")
    parser.add_argument("--train_t2f", required=True, help="Path to train Lang->French JSONL")
    parser.add_argument("--val_t2f", required=True, help="Path to val Lang->French JSONL")
    parser.add_argument("--train_f2t", required=True, help="Path to train French->Lang JSONL")
    parser.add_argument("--val_f2t", required=True, help="Path to val French->Lang JSONL")
    parser.add_argument("--output_dir", default="nllb-bidirectional", help="Output directory for models")
    parser.add_argument("--model_id", default=DEFAULT_MODEL_ID, help="Base model ID")
    
    args = parser.parse_args()
    
    set_seed()
    
    finetuner = BilingualFineTuner(args.language, args.code, args.model_id)
    
    train_t2f, val_t2f, train_f2t, val_f2t = finetuner.load_data(
        args.train_t2f, args.val_t2f, args.train_f2t, args.val_f2t
    )
    
    finetuner.setup_model_and_tokenizer()
    
    finetuner.setup_lora()
    
    train_ds, eval_ds = finetuner.tokenize_data(train_t2f, val_t2f, train_f2t, val_f2t)
    
    trainer = finetuner.train(train_ds, eval_ds, f"{args.output_dir}-{args.language.lower()}-fr")
    
    finetuner.save_models(adapter_dir=f"{args.output_dir}-{args.language.lower()}-fr/adapter",
                          merged_dir=f"{args.output_dir}-{args.language.lower()}-fr/model-merged")
    
    print(f"{args.language}-French bidirectional training complete!")

if __name__ == "__main__":

    main()

