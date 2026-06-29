import pandas as pd
import numpy as np
import re
import unicodedata
import os
import argparse
import time
import tqdm
import gc
from difflib import SequenceMatcher
from unidecode import unidecode
from num2words import num2words
from datetime import datetime
from datasets import Dataset
from babel.dates import format_date
from openai import OpenAI

class BilingualDataCleaner:
    """
    A general data cleaner for any language + corresponding French bilingual datasets
    with paraphrasing augmentation support
    """

    def __init__(self, language_name: str, language_columns: tuple = None,
                 azure_config: dict = None):
        """
        Args:
            language_name: Name of the language (e.g., 'Kabye', 'Mina', 'Ewe', 'Tem')
            language_columns: Tuple of (source_col, target_col) names
            azure_config: Dictionary with Azure OpenAI configuration
                         {'endpoint': str, 'deployment': str, 'api_key': str, 'api_version': str}
        """
        self.language_name = language_name
        self.source_col, self.target_col = language_columns or (f'{language_name}', 'French')

        # Azure OpenAI
        if azure_config:
            self.client = OpenAI(
                base_url=azure_config.get('base_url'),
                api_key=azure_config.get('api_key'),
            )
                
            self.deployment = azure_config.get('deployment')
        else:
            self.client = None
            self.deployment = None

    def load_data(self, file_path, sheets=None, concat=True):
        """
        Load data from Excel file
        """
        xls = pd.ExcelFile(file_path)
    
        if sheets is None:
            # always take first sheet
            sheet = xls.sheet_names[0]
            data = pd.read_excel(xls, sheet_name=sheet)
    
        elif isinstance(sheets, str):
            data = pd.read_excel(xls, sheet_name=sheets)
    
        else:
            data_frames = []
            for sheet in sheets:
                df = pd.read_excel(xls, sheet_name=sheet)
                data_frames.append(df)
    
            data = (
                pd.concat(data_frames, ignore_index=True)
                if concat
                else data_frames
            )

        return data

    def extract_columns(self, data, source_col_map, target_col_map):
        """
        Extract and rename relevant columns

        Args:
            data: Input DataFrame
            source_col_map: Mapping for source language column
            target_col_map: Mapping for target language column
        """
        return data[[source_col_map, target_col_map]].rename(
            columns={source_col_map: self.source_col, target_col_map: self.target_col}
        )

    @staticmethod
    def clean_dash(val):
        if isinstance(val, str):
            return re.sub(r'^\s*-\s*', '', val)
        return val

    @staticmethod
    def clean_sentence(text):
        """
        Removes leading bullets, dashes, numbering from sentences.
        Keeps timestamps and non-string values unchanged.
        """
        if not isinstance(text, str):
            return text

        text = re.sub(r"^\s*[\-\–\—•]+\s*", "", text)
        text = re.sub(r"^\s*\d+[\.\)\-–—]\s*", "", text)

        return text.strip()

    @staticmethod
    def clean_french(val):
        if pd.isna(val):
            return val

        if isinstance(val, (int, float, np.number)):
            return int(val) if float(val).is_integer() else val

        for fmt in ('%Y-%d-%m', '%Y-%m-%d'):
            try:
                dt = pd.to_datetime(val, format=fmt, errors='raise')
                return dt.strftime('%Y-%d-%m')
            except:
                continue

        try:
            num = float(val)
            return int(num) if num.is_integer() else num
        except:
            return val

    @staticmethod
    def french_normalize(val):
        if pd.isna(val):
            return val

        if isinstance(val, (int, float)):
            return num2words(int(val) if float(val).is_integer() else val, lang='fr')

        try:
            dt = pd.to_datetime(val, format='%Y-%d-%m', errors='raise')
            day = 'premier' if dt.day == 1 else num2words(dt.day, lang='fr')
            month = format_date(dt, format="MMMM", locale="fr")
            year = num2words(dt.year, lang='fr')
            return f"{day} {month} {year}"
        except Exception:
            return str(val)

    @staticmethod
    def transliterate_text(text: str) -> str:
      if not isinstance(text, str):
        return text
        
      text = unidecode(text)
      return unicodedata.normalize('NFC', text)

    @staticmethod
    def remove_diacritics(text: str) -> str:
        text = str(text)
        text = ''.join(
            c for c in unicodedata.normalize('NFD', text)
            if unicodedata.category(c) != 'Mn'
        )
        return unicodedata.normalize('NFC', text)

    @staticmethod
    def normalize_typography(text: str) -> str:
        """Normalize punctuation without changing sentence meaning."""
        if not isinstance(text, str):
            return text
        translations = str.maketrans({
            '’': "'",
            '‘': "'",
            '“': '"',
            '”': '"',
            '–': '-',
            '—': '-',
            '…': '...',
        })
        text = text.translate(translations)
        text = re.sub(r'\s+([,.;:!?])', r'\1', text)
        text = re.sub(r"\s+'\s*", "'", text)
        return " ".join(text.split())

    @staticmethod
    def normalize_unicode(text: str) -> str:
        """Normalize Unicode composition and repeated whitespace."""
        if not isinstance(text, str):
            return text
        return unicodedata.normalize('NFC', " ".join(text.split()))

    @staticmethod
    def strip_outer_quotes(text: str) -> str:
        """Remove wrapping quotes that often appear after spreadsheet export."""
        if not isinstance(text, str):
            return text
        text = text.strip()
        quote_pairs = [
            ('"', '"'),
            ("'", "'"),
            ('“', '”'),
            ('«', '»'),
        ]
        for left, right in quote_pairs:
            if len(text) >= 2 and text.startswith(left) and text.endswith(right):
                return text[1:-1].strip()
        return text

    @staticmethod
    def lowercase_first_letter(text: str) -> str:
        """Add a light casing variant without fully changing the sentence."""
        if not isinstance(text, str) or not text:
            return text
        for index, char in enumerate(text):
            if char.isalpha():
                return text[:index] + char.lower() + text[index + 1:]
        return text

    @staticmethod
    def lowercase_text(text: str) -> str:
        """Create a lowercase input-only robustness variant."""
        if not isinstance(text, str):
            return text
        return text.lower()

    @staticmethod
    def strip_terminal_punctuation(text: str) -> str:
        """Create a punctuation-light input variant for noisy user text."""
        if not isinstance(text, str):
            return text
        return re.sub(r'[\s.!?;:…]+$', '', text).strip()

    def normalize_noisy_input(self, text: str) -> str:
        """Combine typography and diacritic normalization in one variant."""
        text = self.normalize_typography(text)
        text = self.remove_diacritics(text)
        return self.strip_terminal_punctuation(text)

    @staticmethod
    def normalize_text(text) -> str:
        """Normalize only if text is a string; otherwise return as-is for deduplication."""
        if isinstance(text, str):
            return " ".join(text.lower().strip().split())
        return text

    @staticmethod
    def is_valid_french_paraphrase(
        original,
        paraphrase,
        min_length_ratio=0.5,
        max_length_ratio=1.8,
        max_similarity=0.995,
    ):
        """Filter empty, copied, length-mismatched, and near-identical paraphrases."""
        original_norm = BilingualDataCleaner.normalize_text(original)
        paraphrase_norm = BilingualDataCleaner.normalize_text(paraphrase)
        if not original_norm or not paraphrase_norm or original_norm == paraphrase_norm:
            return False

        original_length = max(1, len(original_norm.split()))
        ratio = len(paraphrase_norm.split()) / original_length
        if ratio < min_length_ratio or ratio > max_length_ratio:
            return False

        if max_similarity is not None and max_similarity < 1:
            similarity = SequenceMatcher(None, original_norm, paraphrase_norm).ratio()
            if similarity > max_similarity:
                return False

        return True

    @staticmethod
    def _strip_language_tags(text, language_codes):
        text = " ".join(str(text).strip().split())
        for code in language_codes:
            prefix = f"{code} "
            if text.startswith(prefix):
                text = text[len(prefix):].strip()
        return text

    def generate_french_pivot_paraphrases(
        self,
        data,
        model_id="facebook/nllb-200-distilled-600M",
        pivot_language_code="eng_Latn",
        batch_size=16,
        max_length=128,
        num_beams=5,
        num_return_sequences=1,
        max_samples=None,
        max_similarity=0.995,
        do_sample=False,
    ):
        """
        Generate French paraphrase pairs with a multilingual pivot model.

        The trusted low-resource sentence is kept unchanged. Only the French
        side is paraphrased, which creates valid extra pairs for both training
        directions after the normal bidirectional split.
        """
        if num_return_sequences < 1:
            raise ValueError("num_return_sequences must be positive")
        if num_beams < num_return_sequences and not do_sample:
            raise ValueError("num_beams must be >= num_return_sequences when sampling is disabled")

        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except Exception as exc:
            raise ImportError(
                "French paraphrasing requires torch and transformers to be installed"
            ) from exc

        work = data[[self.source_col, self.target_col]].dropna().copy()
        work[self.source_col] = work[self.source_col].astype(str).str.strip()
        work[self.target_col] = work[self.target_col].astype(str).str.strip()
        work = work[work[self.source_col].ne('') & work[self.target_col].ne('')]
        work = work.drop_duplicates(subset=[self.source_col, self.target_col]).reset_index(drop=True)
        if max_samples is not None:
            work = work.head(max_samples)

        if work.empty:
            return pd.DataFrame(columns=[self.source_col, self.target_col]), {
                "model": model_id,
                "pivot_language_code": pivot_language_code,
                "input_examples": 0,
                "generated_examples": 0,
            }

        use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        if use_bf16:
            dtype = torch.bfloat16
        elif torch.cuda.is_available():
            dtype = torch.float16
        else:
            dtype = torch.float32

        print(
            f"\nGenerating French paraphrases with {model_id} "
            f"via pivot {pivot_language_code}..."
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_id,
            device_map="auto" if torch.cuda.is_available() else None,
            torch_dtype=dtype,
        )
        model.eval()

        french_code = "fra_Latn"
        french_id = tokenizer.convert_tokens_to_ids(french_code)
        pivot_id = tokenizer.convert_tokens_to_ids(pivot_language_code)
        invalid_ids = {None, tokenizer.unk_token_id}
        if french_id in invalid_ids:
            raise ValueError(f"Model/tokenizer does not know French token: {french_code}")
        if pivot_id in invalid_ids:
            raise ValueError(f"Model/tokenizer does not know pivot token: {pivot_language_code}")

        device = next(model.parameters()).device

        def translate_batches(texts, src_lang, tgt_lang, forced_bos_id, returns_per_input=1):
            translated = []
            for start in tqdm.tqdm(
                range(0, len(texts), batch_size),
                desc=f"{src_lang}->{tgt_lang} paraphrase pass",
            ):
                batch = texts[start:start + batch_size]
                tokenizer.src_lang = src_lang
                tokenizer.tgt_lang = tgt_lang
                inputs = tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                ).to(device)
                generation_kwargs = {
                    "forced_bos_token_id": forced_bos_id,
                    "max_new_tokens": max_length,
                    "num_beams": max(num_beams, returns_per_input),
                    "num_return_sequences": returns_per_input,
                    "do_sample": do_sample,
                    "early_stopping": True,
                    "pad_token_id": tokenizer.pad_token_id,
                    "eos_token_id": tokenizer.eos_token_id,
                }
                with torch.inference_mode():
                    outputs = model.generate(**inputs, **generation_kwargs)
                decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                decoded = [
                    self._strip_language_tags(text, [src_lang, tgt_lang, french_code, pivot_language_code])
                    for text in decoded
                ]
                if returns_per_input == 1:
                    translated.extend(decoded)
                else:
                    for offset in range(0, len(decoded), returns_per_input):
                        translated.append(decoded[offset:offset + returns_per_input])

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            return translated

        french_texts = work[self.target_col].tolist()
        pivot_groups = translate_batches(
            french_texts,
            src_lang=french_code,
            tgt_lang=pivot_language_code,
            forced_bos_id=pivot_id,
            returns_per_input=num_return_sequences,
        )

        flat_pivots = []
        flat_source_indices = []
        for row_index, candidates in enumerate(pivot_groups):
            if isinstance(candidates, str):
                candidates = [candidates]
            seen_pivots = set()
            for candidate in candidates:
                normalized = self.normalize_text(candidate)
                if normalized and normalized not in seen_pivots:
                    seen_pivots.add(normalized)
                    flat_pivots.append(candidate)
                    flat_source_indices.append(row_index)

        if not flat_pivots:
            del model
            del tokenizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return pd.DataFrame(columns=[self.source_col, self.target_col]), {
                "model": model_id,
                "pivot_language_code": pivot_language_code,
                "input_examples": len(work),
                "generated_examples": 0,
            }

        french_back = translate_batches(
            flat_pivots,
            src_lang=pivot_language_code,
            tgt_lang=french_code,
            forced_bos_id=french_id,
            returns_per_input=1,
        )

        generated_rows = []
        seen_pairs = set()
        for row_index, paraphrase in zip(flat_source_indices, french_back):
            original_row = work.iloc[row_index]
            original_french = original_row[self.target_col]
            paraphrase = self.normalize_typography(paraphrase)
            if not self.is_valid_french_paraphrase(
                original_french,
                paraphrase,
                max_similarity=max_similarity,
            ):
                continue
            pair_key = (
                self.normalize_text(original_row[self.source_col]),
                self.normalize_text(paraphrase),
            )
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)
            generated_rows.append({
                self.source_col: original_row[self.source_col],
                self.target_col: paraphrase,
            })

        del model
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        generated = pd.DataFrame(generated_rows, columns=[self.source_col, self.target_col])
        stats = {
            "model": model_id,
            "pivot_language_code": pivot_language_code,
            "input_examples": len(work),
            "generated_examples": len(generated),
            "num_return_sequences": num_return_sequences,
            "max_similarity": max_similarity,
        }
        print(f"Generated {len(generated)} French paraphrase pairs")
        return generated, stats

    def generate_paraphrases(self, data, batch_size=20, num_paraphrases=5, max_retries=3):
        """
        Generate paraphrases for French sentences using Azure OpenAI

        Args:
            data: DataFrame with source and target columns
            batch_size: Number of sentences to process in one API call
            num_paraphrases: Number of paraphrases to generate per sentence
            max_retries: Maximum number of retry attempts for API calls
        """
        if not self.client:
            print("Warning: Azure OpenAI client not configured. Skipping paraphrasing.")
            return data

        augmented_pairs = []

        for batch_start in tqdm.tqdm(range(0, len(data), batch_size), desc="Generating paraphrases"):
            batch_end = min(batch_start + batch_size, len(data))
            batch = data.iloc[batch_start:batch_end].reset_index(drop=True)

            # Add original pairs
            for _, row in batch.iterrows():
                augmented_pairs.append((row[self.source_col], row[self.target_col]))

            # Prepare batch prompt
            prompt_lines = []
            for pos, (_, row) in enumerate(batch.iterrows()):
                prompt_lines.append(f"{pos}: {row[self.target_col]}")

            batch_prompt = (
                f"Génère exactement {num_paraphrases} paraphrases différentes pour chaque phrase ci-dessous, "
                "en respectant ces contraintes :\n"
                "- Français uniquement\n"
                "- Sens strictement conservé\n"
                "- Syntaxe et vocabulaire variés\n"
                "- Une phrase par ligne\n"
                "- Pas de numérotation\n\n"

                "IMPORTANT :\n"
                "- Ne fournis AUCUNE explication\n"
                "- Ne fais AUCUN commentaire\n"
                "- Ne fais AUCUNE introduction\n"
                "- Chaque ligne DOIT commencer STRICTEMENT par :\n"
                "  <numéro>: <phrase>\n"
                "- Exemple valide : 0: Je suis allé au marché hier.\n"
                "- Exemple invalide : Je constate que...\n\n"

                "Phrases à paraphraser :\n"
                + "\n".join(prompt_lines)
            )

            # Call API with retry logic
            raw_text = ""
            for attempt in range(max_retries):
                try:
                    response = self.client.chat.completions.create(
                        model=self.deployment,
                        messages=[
                            {"role": "system", "content": "Tu es un expert en reformulation linguistique."},
                            {"role": "user", "content": batch_prompt},
                        ],
                    )
                    raw_text = response.choices[0].message.content
                    break
                except Exception as e:
                    if attempt == max_retries - 1:
                        print(f"[ERROR] Max retries reached for batch {batch_start}-{batch_end}. Skipping.")
                    else:
                        time.sleep(2 ** attempt)

            # Parse paraphrases
            if raw_text:
                lines = [line.strip() for line in raw_text.split("\n") if line.strip()]

                seen_dict = {pos: set([self.normalize_text(row[self.target_col])])
                            for pos, (_, row) in enumerate(batch.iterrows())}

                for line in lines:
                    match = re.match(r"^\s*(\d+)\s*[:\-–]\s*(.+)$", line)
                    if not match:
                        continue
                    idx = int(match.group(1))
                    para = match.group(2).strip()

                    if idx < 0 or idx >= len(batch):
                        continue

                    source_sentence = batch.iloc[idx][self.source_col]

                    norm_para = self.normalize_text(para)
                    if norm_para not in seen_dict[idx]:
                        augmented_pairs.append((source_sentence, para))
                        seen_dict[idx].add(norm_para)

        return pd.DataFrame(augmented_pairs, columns=[self.source_col, self.target_col])

    def create_augmentations(self, data, input_column, max_augmented_variants=3):
        """
        Apply bounded, meaning-preserving augmentations ONLY to the input column.
        """
        if max_augmented_variants < 0:
            raise ValueError("max_augmented_variants must be non-negative")

        base = data.reset_index(drop=True).copy()
        base['_augmentation_source_id'] = range(len(base))
        augmented_dfs = [base.assign(is_aug='original')]

        transformations = [
            ('transliterated', self.transliterate_text),
            ('no_diacritics', self.remove_diacritics),
            ('normalized_typography', self.normalize_typography),
            ('no_terminal_punctuation', self.strip_terminal_punctuation),
            ('normalized_noisy_input', self.normalize_noisy_input),
            ('unicode_normalized', self.normalize_unicode),
            ('outer_quotes_removed', self.strip_outer_quotes),
            ('lowercase_first_letter', self.lowercase_first_letter),
            ('lowercase_text', self.lowercase_text),
        ]

        for name, transform in transformations:
            augmented_data = base.copy()
            augmented_data[input_column] = augmented_data[input_column].map(transform)
            changed = ~augmented_data[input_column].fillna('').eq(base[input_column].fillna(''))
            if changed.any():
                augmented_dfs.append(augmented_data.loc[changed].assign(is_aug=name))

        augmented = pd.concat(augmented_dfs, ignore_index=True)
        augmented = augmented.drop_duplicates(subset=[self.source_col, self.target_col])
        augmented = (
            augmented
            .groupby('_augmentation_source_id', sort=False, group_keys=False)
            .head(max_augmented_variants + 1)
            .drop(columns=['_augmentation_source_id'])
            .reset_index(drop=True)
        )
        return augmented

    def prepare_datasets(
        self,
        data,
        test_size=0.2,
        seed=42,
        enable_paraphrasing=False,
        max_augmented_variants=3,
        source_max_augmented_variants=None,
        target_max_augmented_variants=None,
        enable_french_pivot_paraphrasing=False,
        paraphrase_model="facebook/nllb-200-distilled-600M",
        paraphrase_pivot_language_code="eng_Latn",
        paraphrase_batch_size=16,
        paraphrase_max_length=128,
        paraphrase_num_beams=5,
        paraphrase_num_return_sequences=1,
        paraphrase_max_samples=None,
        paraphrase_max_similarity=0.995,
    ):
        """
        Prepare train/validation datasets for both translation directions
        """
        if source_max_augmented_variants is None:
            source_max_augmented_variants = max_augmented_variants
        if target_max_augmented_variants is None:
            target_max_augmented_variants = max_augmented_variants
        if source_max_augmented_variants < 0 or target_max_augmented_variants < 0:
            raise ValueError("augmentation variant limits must be non-negative")

        for col in [self.source_col, self.target_col]:
            data[col] = (
                data[col]
                .replace({pd.NaT: ''})
                .fillna('')
                .astype(str)
                .str.strip()
            )
        # Build one clean bilingual table and split it before augmentation. This
        # keeps validation examples (and their augmented variants) out of train.
        source_target = data[[self.source_col, self.target_col]].dropna().copy()
        source_target[self.target_col] = source_target[self.target_col].apply(self.clean_french)
        source_target[self.target_col] = source_target[self.target_col].apply(self.french_normalize)
        source_target = source_target[
            source_target[self.source_col].ne('') & source_target[self.target_col].ne('')
        ].drop_duplicates(subset=[self.source_col, self.target_col]).reset_index(drop=True)

        original_source_target_size = len(source_target)
        original_dataset = Dataset.from_pandas(
            source_target.astype('string'),
            preserve_index=False,
        )
        split = original_dataset.train_test_split(test_size=test_size, seed=seed)
        train_original = split['train'].to_pandas()
        val_source_target = split['test'].to_pandas()

        # Optional paraphrases are generated only after the split so synthetic
        # versions of validation examples can never leak into training.
        pivot_paraphrase_stats = None
        if enable_french_pivot_paraphrasing:
            paraphrased_train, pivot_paraphrase_stats = self.generate_french_pivot_paraphrases(
                train_original,
                model_id=paraphrase_model,
                pivot_language_code=paraphrase_pivot_language_code,
                batch_size=paraphrase_batch_size,
                max_length=paraphrase_max_length,
                num_beams=paraphrase_num_beams,
                num_return_sequences=paraphrase_num_return_sequences,
                max_samples=paraphrase_max_samples,
                max_similarity=paraphrase_max_similarity,
            )
            if not paraphrased_train.empty:
                train_original = (
                    pd.concat([train_original, paraphrased_train], ignore_index=True)
                    .drop_duplicates(subset=[self.source_col, self.target_col])
                    .reset_index(drop=True)
                )
                print(f"Training pairs after French paraphrasing: {len(train_original)}")

        if enable_paraphrasing:
            print("\nGenerating training-only paraphrases...")
            train_original = self.generate_paraphrases(train_original)
            train_original.to_excel('augmented_train_df.xlsx', index=False)
            print(f"Augmented training data saved ({len(train_original)} examples)")

        train_source_target_augmented = self.create_augmentations(
            train_original,
            input_column=self.source_col,
            max_augmented_variants=source_max_augmented_variants,
        )
        source_augmentation_counts = {
            str(name): int(count)
            for name, count in train_source_target_augmented['is_aug'].value_counts().items()
        }
        train_source_target = train_source_target_augmented.drop(columns=['is_aug'])

        train_target_source_original = train_original[[self.target_col, self.source_col]].copy()
        train_target_source_augmented = self.create_augmentations(
            train_target_source_original,
            input_column=self.target_col,
            max_augmented_variants=target_max_augmented_variants,
        )
        target_augmentation_counts = {
            str(name): int(count)
            for name, count in train_target_source_augmented['is_aug'].value_counts().items()
        }
        train_target_source = train_target_source_augmented.drop(columns=['is_aug'])
        val_target_source = val_source_target[[self.target_col, self.source_col]].copy()

        augmented_source_target_size = len(train_source_target)
        original_target_source_size = original_source_target_size
        augmented_target_source_size = len(train_target_source)

        train_st = Dataset.from_pandas(
            train_source_target.astype('string'),
            preserve_index=False,
        )
        val_st = Dataset.from_pandas(
            val_source_target.astype('string'),
            preserve_index=False,
        )
        train_ts = Dataset.from_pandas(
            train_target_source.astype('string'),
            preserve_index=False,
        )
        val_ts = Dataset.from_pandas(
            val_target_source.astype('string'),
            preserve_index=False,
        )

        lang_lower = self.language_name.lower()
        target_lower = self.target_col.lower()

        stats = {
            "source_to_target": {
                "original_examples": original_source_target_size,
                "train_examples_before_augmentation": len(train_original),
                "augmented_examples": augmented_source_target_size,
                "augmentation_gain": augmented_source_target_size - len(train_original),
                "augmentation_types": source_augmentation_counts,
                "max_augmented_variants": source_max_augmented_variants,
                "validation_examples": len(val_source_target),
            },
            "target_to_source": {
                "original_examples": original_target_source_size,
                "train_examples_before_augmentation": len(train_target_source_original),
                "augmented_examples": augmented_target_source_size,
                "augmentation_gain": augmented_target_source_size - len(train_target_source_original),
                "augmentation_types": target_augmentation_counts,
                "max_augmented_variants": target_max_augmented_variants,
                "validation_examples": len(val_target_source),
            },
            "french_pivot_paraphrasing": pivot_paraphrase_stats,
        }

        return {
            "datasets": {
                f'train_{lang_lower}2{target_lower}': train_st,
                f'val_{lang_lower}2{target_lower}': val_st,
                f'train_{target_lower}2{lang_lower}': train_ts,
                f'val_{target_lower}2{lang_lower}': val_ts
            },
            "stats": stats
        }

    def process_dataset(self, file_path, sheets=None, source_col=None,
                       target_col=None, enable_paraphrasing=False, **kwargs):
        """
        Complete processing pipeline
        """
        data = self.load_data(file_path, sheets)
        print(data.columns)
        print(data.shape)
        print(data.head(10))                   

        if source_col is None:
            if 'Phrase en langue nationale' in data.columns:
                source_col = 'Phrase en langue nationale'
            elif 'Column 1' in data.columns:
                source_col = 'Column 1'
            elif self.language_name in data.columns:
                source_col = self.language_name
            else:
                raise ValueError(f"Could not find source column. Available columns: {data.columns.tolist()}")

        if target_col is None:
            if 'Phrase équivalente en français' in data.columns:
                target_col = 'Phrase équivalente en français'
            elif 'French' in data.columns:
                target_col = 'French'
            else:
                raise ValueError(f"Could not find target column. Available columns: {data.columns.tolist()}")

        # Extract and rename columns
        data = self.extract_columns(data, source_col, target_col)

        # Clean both columns
        data[self.source_col] = data[self.source_col].apply(self.clean_sentence)
        data[self.target_col] = data[self.target_col].apply(self.clean_sentence)

        # Remove empty rows
        data = data.dropna()

        # Prepare datasets
        result = self.prepare_datasets(data, enable_paraphrasing=enable_paraphrasing, **kwargs)
        return result


def main():
    parser = argparse.ArgumentParser(description='Clean and prepare bilingual translation datasets')
    parser.add_argument('--language', type=str, required=True, help='Language name (e.g., Kabye, Mina, Ewe)')
    parser.add_argument('--data_path', type=str, required=True, help='Path to Excel file')
    parser.add_argument('--output_dir', type=str, default='./datasets', help='Output directory for cleaned datasets')
    parser.add_argument('--sheets', nargs='+', default=None, help='Sheet names to process (default: all sheets)')
    parser.add_argument('--source_col', type=str, default=None, help='Source column name (auto-detected if not provided)')
    parser.add_argument('--target_col', type=str, default=None, help='Target column name (auto-detected if not provided)')
    parser.add_argument('--test_size', type=float, default=0.2, help='Validation split ratio (default: 0.2)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    parser.add_argument(
        '--max_augmented_variants',
        type=int,
        default=3,
        help='Maximum synthetic input variants per original training pair',
    )
    parser.add_argument(
        '--source_max_augmented_variants',
        type=int,
        default=None,
        help='Maximum input variants for language->French training pairs',
    )
    parser.add_argument(
        '--target_max_augmented_variants',
        type=int,
        default=None,
        help='Maximum input variants for French->language training pairs',
    )

    # Paraphrasing arguments
    parser.add_argument('--enable_paraphrasing', action='store_true', help='Enable paraphrasing augmentation')
    parser.add_argument('--azure_endpoint', type=str, default=None, help='Azure OpenAI endpoint')
    parser.add_argument('--azure_deployment', type=str, default='gpt-4o', help='Azure OpenAI deployment name')
    parser.add_argument('--azure_api_key', type=str, default=None, help='Azure OpenAI API key')
    parser.add_argument('--enable_french_pivot_paraphrasing', action='store_true', help='Enable local French pivot paraphrasing')
    parser.add_argument('--paraphrase_model', type=str, default='facebook/nllb-200-distilled-600M', help='Multilingual seq2seq model for French pivot paraphrasing')
    parser.add_argument('--paraphrase_pivot_language_code', type=str, default='eng_Latn', help='Pivot language code for paraphrasing')
    parser.add_argument('--paraphrase_batch_size', type=int, default=16, help='Paraphrase generation batch size')
    parser.add_argument('--paraphrase_max_length', type=int, default=128, help='Maximum paraphrase input/output length')
    parser.add_argument('--paraphrase_num_beams', type=int, default=5, help='Beam size for paraphrase generation')
    parser.add_argument('--paraphrase_num_return_sequences', type=int, default=1, help='Paraphrases generated per French sentence')
    parser.add_argument('--paraphrase_max_samples', type=int, default=None, help='Limit number of training rows paraphrased')
    parser.add_argument('--paraphrase_max_similarity', type=float, default=0.995, help='Reject near-copy paraphrases above this similarity')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print(f"CLEANING {args.language.upper()}-FRENCH DATASET")
    print("=" * 60)

    # Setup Azure config if paraphrasing is enabled
    azure_config = None
    if args.enable_paraphrasing:
        azure_endpoint = args.azure_endpoint or os.getenv("ENDPOINT_URL")
        azure_api_key = args.azure_api_key or os.getenv("AZURE_OPENAI_API_KEY")

        if not azure_endpoint or not azure_api_key:
            print("Warning: Azure credentials not provided. Paraphrasing will be skipped.")
        else:
            azure_config = {
                'base_url': azure_endpoint,
                'deployment': args.azure_deployment,
                'api_key': azure_api_key,
            }

    cleaner = BilingualDataCleaner(args.language, azure_config=azure_config)

    print(f"\nProcessing: {args.data_path}")
    result = cleaner.process_dataset(
        file_path=args.data_path,
        sheets=args.sheets,
        source_col=args.source_col,
        target_col=args.target_col,
        test_size=args.test_size,
        seed=args.seed,
        enable_paraphrasing=args.enable_paraphrasing,
        max_augmented_variants=args.max_augmented_variants,
        source_max_augmented_variants=args.source_max_augmented_variants,
        target_max_augmented_variants=args.target_max_augmented_variants,
        enable_french_pivot_paraphrasing=args.enable_french_pivot_paraphrasing,
        paraphrase_model=args.paraphrase_model,
        paraphrase_pivot_language_code=args.paraphrase_pivot_language_code,
        paraphrase_batch_size=args.paraphrase_batch_size,
        paraphrase_max_length=args.paraphrase_max_length,
        paraphrase_num_beams=args.paraphrase_num_beams,
        paraphrase_num_return_sequences=args.paraphrase_num_return_sequences,
        paraphrase_max_samples=args.paraphrase_max_samples,
        paraphrase_max_similarity=args.paraphrase_max_similarity,
    )
    datasets = result['datasets']

    # Save datasets
    print(f"\nSaving datasets to: {args.output_dir}")
    for name, dataset in datasets.items():
        filepath = os.path.join(args.output_dir, f"{name}.jsonl")
        dataset.to_json(filepath, lines=True, force_ascii=False)
        print(f" {name}.jsonl ({len(dataset)} examples)")

    lang_lower = args.language.lower()

    print("\n" + "=" * 60)
    print("Cleaning Completed")
    print("=" * 60)
    print(f"\nDataset files saved in: {args.output_dir}/")
    print(f"  - train_{lang_lower}2french.jsonl")
    print(f"  - val_{lang_lower}2french.jsonl")
    print(f"  - train_french2{lang_lower}.jsonl")
    print(f"  - val_french2{lang_lower}.jsonl")


if __name__ == "__main__":
    main()
