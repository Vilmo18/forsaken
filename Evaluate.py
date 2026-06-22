import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
import evaluate
from tqdm import tqdm

class BilingualEvaluator:
    def __init__(self, language_name, language_code, model_path, model_type="merged"):
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
        self.model = None
        self.tokenizer = None
        
    def load_model(self):
        """Load the trained model and tokenizer"""
        print(f"Loading {self.model_type} model from {self.model_path}")
        
        if self.model_type == "lora":
            base_model = AutoModelForSeq2SeqLM.from_pretrained(
                "facebook/nllb-200-distilled-600M", 
                device_map="auto",
                torch_dtype=torch.float16
            )
            self.model = PeftModel.from_pretrained(base_model, self.model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        else:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_path, 
                device_map="auto",
                torch_dtype=torch.float16
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            
        print(f"Model loaded successfully")
        
    def translate_lang_to_french(self, lang_text):
        """Translate language text to French"""
        if not lang_text or not lang_text.strip():
            return ""
        
        try:
            self.tokenizer.src_lang = self.language_code
            self.tokenizer.tgt_lang = "fra_Latn"
            
            inputs = self.tokenizer(lang_text, return_tensors="pt", truncation=True, max_length=256).to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=96,
                    num_beams=4,
                    no_repeat_ngram_size=3,
                    early_stopping=True,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            translation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Clean up language tags
            if translation.startswith("fra_Latn "):
                translation = translation[9:]
            if translation.startswith(f"{self.language_code} "):
                translation = translation[len(self.language_code) + 1:]
                
            return translation.strip()
            
        except Exception as e:
            print(f"Translation error for '{lang_text[:50]}...': {e}")
            return ""
    
    def translate_french_to_lang(self, french_text):
        """Translate French text to language"""
        if not french_text or not french_text.strip():
            return ""
        
        try:
            self.tokenizer.src_lang = "fra_Latn"
            self.tokenizer.tgt_lang = self.language_code
            
            inputs = self.tokenizer(french_text, return_tensors="pt", truncation=True, max_length=256).to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=96,
                    num_beams=4,
                    no_repeat_ngram_size=3,
                    early_stopping=True,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            translation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Clean up language tags
            if translation.startswith(f"{self.language_code} "):
                translation = translation[len(self.language_code) + 1:]
            if translation.startswith("fra_Latn "):
                translation = translation[9:]
                
            return translation.strip()
            
        except Exception as e:
            print(f"Translation error for '{french_text[:50]}...': {e}")
            return ""
    
    def evaluate_direction(self, val_df, direction, translate_fn, source_col, target_col):
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
        batch_size = 25
        
        for i in tqdm(range(0, len(source_texts), batch_size), desc=f"{direction} translation"):
            batch_sources = source_texts[i:i+batch_size]
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
    parser.add_argument("--val_lang2fr", required=True, help="Path to validation Lang->French JSONL")
    parser.add_argument("--val_fr2lang", required=True, help="Path to validation French->Lang JSONL")
    
    args = parser.parse_args()
    
    evaluator = BilingualEvaluator(args.language, args.code, args.model_path, args.model_type)
    
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
        "French"
    )
    
    fr2lang_results, fr2lang_preds, fr2lang_src, fr2lang_refs = evaluator.evaluate_direction(
        val_fr2lang, 
        f"French-to-{args.language}", 
        evaluator.translate_french_to_lang, 
        "French", 
        args.language
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