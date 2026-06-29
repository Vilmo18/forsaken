import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
import evaluate
from tqdm import tqdm

DEFAULT_MODEL_ID = "facebook/nllb-200-distilled-600M"

class BilingualEvaluator:
    def __init__(
        self,
        language_name,
        language_code,
        model_path,
        model_type="merged",
        base_model_id=DEFAULT_MODEL_ID,
        max_input_length=256,
        max_new_tokens=128,
        num_beams=4,
        no_repeat_ngram_size=3,
        eval_batch_size=32,
    ):
        """
        Initialize evaluator for a specific language
        
        Args:
            language_name: Name of the language (e.g., 'Ewe', 'Tem', 'Mina')
            language_code: NLLB language code (e.g., 'ewe_Latn', 'tem_Latn')
            model_path: Path to the trained model
            model_type: Type of model ('merged' or 'lora')
        """
        self.language_name = language_name
        self.language_code = language_code
        self.model_path = model_path
        self.model_type = model_type
        self.base_model_id = base_model_id
        self.max_input_length = max_input_length
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams
        self.no_repeat_ngram_size = no_repeat_ngram_size
        self.eval_batch_size = eval_batch_size
        self.model = None
        self.tokenizer = None

    def load_model(self):
        """Load the trained model and tokenizer"""
        print(f"Loading {self.model_type} model from {self.model_path}")
        use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        if use_bf16:
            dtype = torch.bfloat16
        elif torch.cuda.is_available():
            dtype = torch.float16
        else:
            dtype = torch.float32

        if self.model_type == "lora":
            base_model = AutoModelForSeq2SeqLM.from_pretrained(
                self.base_model_id,
                device_map="auto",
                torch_dtype=dtype
            )
            self.model = PeftModel.from_pretrained(base_model, self.model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        else:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_path,
                device_map="auto",
                torch_dtype=dtype
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model.eval()

        print(f"Model loaded successfully")

    def _model_device(self):
        return next(self.model.parameters()).device

    def _clean_translation(self, translation):
        """Remove language tags sometimes emitted by NLLB."""
        translation = translation.strip()
        for tag in (self.language_code, "fra_Latn"):
            prefix = f"{tag} "
            if translation.startswith(prefix):
                translation = translation[len(prefix):].strip()
        return translation

    def _translate_batch(self, texts, src_lang, tgt_lang):
        """Translate a batch while preserving empty-string positions."""
        if not texts:
            return []

        results = [""] * len(texts)
        clean_inputs = []
        clean_positions = []
        for index, text in enumerate(texts):
            text = "" if pd.isna(text) else str(text).strip()
            if text:
                clean_positions.append(index)
                clean_inputs.append(text)

        if not clean_inputs:
            return results

        self.tokenizer.src_lang = src_lang
        self.tokenizer.tgt_lang = tgt_lang
        inputs = self.tokenizer(
            clean_inputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_input_length,
        ).to(self._model_device())

        generation_kwargs = {
            "forced_bos_token_id": self.tokenizer.convert_tokens_to_ids(tgt_lang),
            "max_new_tokens": self.max_new_tokens,
            "num_beams": self.num_beams,
            "early_stopping": True,
            "do_sample": False,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if self.no_repeat_ngram_size and self.no_repeat_ngram_size > 0:
            generation_kwargs["no_repeat_ngram_size"] = self.no_repeat_ngram_size

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **generation_kwargs)

        translations = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        for index, translation in zip(clean_positions, translations):
            results[index] = self._clean_translation(translation)
        return results

    def translate_lang_to_french_batch(self, lang_texts):
        """Translate a batch from the local language to French."""
        return self._translate_batch(lang_texts, self.language_code, "fra_Latn")

    def translate_french_to_lang_batch(self, french_texts):
        """Translate a batch from French to the local language."""
        return self._translate_batch(french_texts, "fra_Latn", self.language_code)

    def translate_lang_to_french(self, lang_text):
        """Translate language text to French"""
        try:
            return self.translate_lang_to_french_batch([lang_text])[0]

        except Exception as e:
            print(f"Translation error for '{str(lang_text)[:50]}...': {e}")
            return ""
    
    def translate_french_to_lang(self, french_text):
        """Translate French text to language"""
        try:
            return self.translate_french_to_lang_batch([french_text])[0]

        except Exception as e:
            print(f"Translation error for '{str(french_text)[:50]}...': {e}")
            return ""

    def evaluate_direction(self, val_df, direction, translate_fn, source_col, target_col, batch_translate_fn=None):
        """Evaluate translation quality for one direction"""
        print(f"\n{'='*60}")
        print(f"EVALUATING {direction.upper()}")
        print(f"{'='*60}")
        
        # Load metrics
        try:
            bleu = evaluate.load("bleu")
            meteor = evaluate.load("meteor")
        except Exception as e:
            print(f"Error loading metrics: {e}")
            return None
        
        source_texts = val_df[source_col].tolist()
        reference_texts = val_df[target_col].tolist()
        
        print(f"Translating {len(source_texts)} samples...")
        
        generated_translations = []
        batch_size = self.eval_batch_size

        for i in tqdm(range(0, len(source_texts), batch_size), desc=f"{direction} translation"):
            batch_sources = source_texts[i:i+batch_size]
            if batch_translate_fn is not None:
                batch_translations = batch_translate_fn(batch_sources)
            else:
                batch_translations = []
                for source_text in batch_sources:
                    translation = translate_fn(source_text)
                    batch_translations.append(translation)
            
            generated_translations.extend(batch_translations)
            
            if i % (batch_size * 4) == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        print("Calculating BLEU and METEOR scores")
        
        try:
            references_for_bleu = [[ref] for ref in reference_texts]
            bleu_score = bleu.compute(
                predictions=generated_translations,
                references=references_for_bleu
            )
            
            meteor_score = meteor.compute(
                predictions=generated_translations,
                references=reference_texts
            )
            
            results = {
                'direction': direction,
                'total_samples': len(generated_translations),
                'bleu': bleu_score['bleu'],
                'meteor': meteor_score['meteor']
            }
            
            print(f"\n{direction.upper()} RESULTS:")
            print(f"Total samples: {results['total_samples']}")
            print(f"BLEU Score: {results['bleu']:.4f}")
            print(f"METEOR Score: {results['meteor']:.4f}")
            
            return results, generated_translations, source_texts, reference_texts
            
        except Exception as e:
            print(f"Error calculating metrics: {e}")
            return None, [], [], []

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate bilingual translation model")
    parser.add_argument("--language", required=True, help="Language name (e.g., Ewe, Tem, Mina)")
    parser.add_argument("--code", required=True, help="NLLB language code (e.g., ewe_Latn, tem_Latn)")
    parser.add_argument("--model_path", required=True, help="Path to trained model")
    parser.add_argument("--model_type", default="merged", choices=["merged", "lora"], help="Model type")
    parser.add_argument("--base_model_id", default=DEFAULT_MODEL_ID, help="Base model used by a LoRA adapter")
    parser.add_argument("--val_lang2fr", required=True, help="Path to validation Lang->French JSONL")
    parser.add_argument("--val_fr2lang", required=True, help="Path to validation French->Lang JSONL")
    parser.add_argument("--eval_num_beams", type=int, default=4, help="Beam size used during generation")
    parser.add_argument("--eval_max_new_tokens", type=int, default=128, help="Maximum generated tokens")
    parser.add_argument("--eval_batch_size", type=int, default=32, help="Evaluation generation batch size")
    parser.add_argument("--eval_no_repeat_ngram_size", type=int, default=3, help="No-repeat ngram size; use 0 to disable")
    
    args = parser.parse_args()
    
    evaluator = BilingualEvaluator(
        args.language,
        args.code,
        args.model_path,
        args.model_type,
        args.base_model_id,
        max_new_tokens=args.eval_max_new_tokens,
        num_beams=args.eval_num_beams,
        no_repeat_ngram_size=args.eval_no_repeat_ngram_size,
        eval_batch_size=args.eval_batch_size,
    )
    
    evaluator.load_model()
    
    print("Loading validation datasets")
    try:
        val_lang2fr = pd.read_json(args.val_lang2fr, lines=True)
        val_fr2lang = pd.read_json(args.val_fr2lang, lines=True)
        print(f"Loaded {args.language}->French validation: {len(val_lang2fr)} samples")
        print(f"Loaded French->{args.language} validation: {len(val_fr2lang)} samples")
    except Exception as e:
        print(f"Error loading datasets: {e}")
        return
    
    # Evaluate both directions
    lang2fr_results, lang2fr_preds, lang2fr_src, lang2fr_refs = evaluator.evaluate_direction(
        val_lang2fr, 
        f"{args.language}-to-French",
        evaluator.translate_lang_to_french,
        args.language,
        "French",
        batch_translate_fn=evaluator.translate_lang_to_french_batch,
    )
    
    fr2lang_results, fr2lang_preds, fr2lang_src, fr2lang_refs = evaluator.evaluate_direction(
        val_fr2lang, 
        f"French-to-{args.language}",
        evaluator.translate_french_to_lang,
        "French",
        args.language,
        batch_translate_fn=evaluator.translate_french_to_lang_batch,
    )
    
    print(f"\n{'='*60}")
    print("EVALUATION COMPLETE")
    print(f"{'='*60}")
    
    if lang2fr_results:
        print(f"{args.language}-to-French: BLEU={lang2fr_results['bleu']:.4f}, METEOR={lang2fr_results['meteor']:.4f}")
    if fr2lang_results:
        print(f"French-to-{args.language}: BLEU={fr2lang_results['bleu']:.4f}, METEOR={fr2lang_results['meteor']:.4f}")
    if lang2fr_results and fr2lang_results:
        avg_bleu = (lang2fr_results['bleu'] + fr2lang_results['bleu']) / 2
        avg_meteor = (lang2fr_results['meteor'] + fr2lang_results['meteor']) / 2
        print(f"Average: BLEU={avg_bleu:.4f}, METEOR={avg_meteor:.4f}")

if __name__ == "__main__":
    main()
