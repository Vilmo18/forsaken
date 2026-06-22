from pathlib import Path
import random
from difflib import SequenceMatcher

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def normalize_text(text):
    return " ".join(str(text).strip().lower().split())


def is_valid_synthetic_pair(source_text, french_text, min_length_ratio=0.4, max_length_ratio=2.5):
    """Reject empty, copied, and severely length-mismatched generations."""
    source = normalize_text(source_text)
    french = normalize_text(french_text)
    if not source or not french or source == french:
        return False

    source_length = max(1, len(source.split()))
    ratio = len(french.split()) / source_length
    return min_length_ratio <= ratio <= max_length_ratio


def text_similarity(left, right):
    """Character-level cycle similarity after whitespace/case normalization."""
    return SequenceMatcher(None, normalize_text(left), normalize_text(right)).ratio()


def load_monolingual_texts(path, column=None, max_samples=None, seed=42):
    """Load unique monolingual sentences from txt, csv, jsonl, or Excel."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Monolingual data not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".txt":
        texts = path.read_text(encoding="utf-8").splitlines()
    elif suffix == ".csv":
        frame = pd.read_csv(path)
        selected_column = column or frame.columns[0]
        if selected_column not in frame.columns:
            raise ValueError(f"Column '{selected_column}' not found in {path}")
        texts = frame[selected_column].tolist()
    elif suffix in {".jsonl", ".json"}:
        frame = pd.read_json(path, lines=suffix == ".jsonl")
        selected_column = column or frame.columns[0]
        if selected_column not in frame.columns:
            raise ValueError(f"Column '{selected_column}' not found in {path}")
        texts = frame[selected_column].tolist()
    elif suffix in {".xlsx", ".xls"}:
        frame = pd.read_excel(path)
        selected_column = column or frame.columns[0]
        if selected_column not in frame.columns:
            raise ValueError(f"Column '{selected_column}' not found in {path}")
        texts = frame[selected_column].tolist()
    else:
        raise ValueError("Monolingual data must be txt, csv, jsonl, json, xlsx, or xls")

    unique_texts = []
    seen = set()
    for value in texts:
        if pd.isna(value):
            continue
        text = " ".join(str(value).strip().split())
        normalized = normalize_text(text)
        if text and normalized not in seen:
            seen.add(normalized)
            unique_texts.append(text)

    random.Random(seed).shuffle(unique_texts)
    if max_samples is not None:
        unique_texts = unique_texts[:max_samples]
    return unique_texts


class BackTranslator:
    """Generate French sources for original monolingual low-resource text."""

    def __init__(self, model_path, language_name, language_code):
        self.language_name = language_name
        self.language_code = language_code
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        if use_bf16:
            dtype = torch.bfloat16
        elif torch.cuda.is_available():
            dtype = torch.float16
        else:
            dtype = torch.float32
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=dtype,
        )
        self.model.eval()

    def generate(
        self,
        texts,
        batch_size=16,
        max_length=128,
        num_beams=4,
        min_cycle_similarity=0.45,
    ):
        french_language_id = self.tokenizer.convert_tokens_to_ids("fra_Latn")
        source_language_id = self.tokenizer.convert_tokens_to_ids(self.language_code)
        device = next(self.model.parameters()).device
        pairs = []
        seen_pairs = set()

        for start in tqdm(range(0, len(texts), batch_size), desc="Back-translating to French"):
            batch = texts[start:start + batch_size]
            self.tokenizer.src_lang = self.language_code
            self.tokenizer.tgt_lang = "fra_Latn"
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(device)

            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    forced_bos_token_id=french_language_id,
                    max_new_tokens=max_length,
                    num_beams=num_beams,
                    do_sample=False,
                )

            translations = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            if min_cycle_similarity > 0:
                self.tokenizer.src_lang = "fra_Latn"
                self.tokenizer.tgt_lang = self.language_code
                round_trip_inputs = self.tokenizer(
                    translations,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                ).to(device)
                with torch.inference_mode():
                    round_trip_outputs = self.model.generate(
                        **round_trip_inputs,
                        forced_bos_token_id=source_language_id,
                        max_new_tokens=max_length,
                        num_beams=num_beams,
                        do_sample=False,
                    )
                reconstructed_sources = self.tokenizer.batch_decode(
                    round_trip_outputs,
                    skip_special_tokens=True,
                )
            else:
                reconstructed_sources = batch

            for source, french, reconstructed in zip(batch, translations, reconstructed_sources):
                french = " ".join(french.strip().split())
                key = (normalize_text(source), normalize_text(french))
                cycle_is_valid = (
                    min_cycle_similarity <= 0
                    or text_similarity(source, reconstructed) >= min_cycle_similarity
                )
                if (
                    is_valid_synthetic_pair(source, french)
                    and cycle_is_valid
                    and key not in seen_pairs
                ):
                    seen_pairs.add(key)
                    pairs.append({self.language_name: source, "French": french})

        return pd.DataFrame(pairs, columns=[self.language_name, "French"])
