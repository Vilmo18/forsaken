import pandas as pd
import numpy as np
import re
import unicodedata
import os
import argparse
import time
import tqdm
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
    def normalize_text(text) -> str:
        """Normalize only if text is a string; otherwise return as-is for deduplication."""
        if isinstance(text, str):
            return " ".join(text.lower().strip().split())
        return text

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

    def create_augmentations(self, data, input_column):
        """
        Apply augmentations ONLY to the input column.
        """
        augmented_dfs = [data.assign(is_aug=0)]  # Original

        # Transliteration
        aug_data_1 = data.copy()
        aug_data_1[input_column] = aug_data_1[input_column].map(self.transliterate_text)

        mask_1 = ~aug_data_1[input_column].fillna('').eq(data[input_column].fillna(''))
        if mask_1.any():
          augmented_dfs.append(aug_data_1.loc[mask_1].assign(is_aug=1))

        # Diacritic removal
        aug_data_2 = data.copy()
        aug_data_2[input_column] = aug_data_2[input_column].map(self.remove_diacritics)

        mask_2 = ~aug_data_2[input_column].fillna('').eq(data[input_column].fillna(''))
        if mask_2.any():
          augmented_dfs.append(aug_data_2.loc[mask_2].assign(is_aug=2))

        augmented = pd.concat(augmented_dfs, ignore_index=True)
        augmented = augmented.drop_duplicates(subset=[self.source_col, self.target_col])
        return augmented  

    def prepare_datasets(self, data, test_size=0.2, seed=42, enable_paraphrasing=False):
        """
        Prepare train/validation datasets for both translation directions
        """
        for col in [self.source_col, self.target_col]:
            data[col] = (
                data[col]
                .replace({pd.NaT: ''})
                .fillna('')
                .astype(str)
                .str.strip()
            )
        # Generate paraphrases if enabled
        if enable_paraphrasing:
            print("\nGenerating paraphrases for augmentation...")
            data = self.generate_paraphrases(data)
            data.to_excel('augmented_df.xlsx', index=False)
            print(f"Augmented data saved to augmented_df.xlsx ({len(data)} examples)")

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

        train_source_target = self.create_augmentations(
            train_original,
            input_column=self.source_col,
        ).drop(columns=['is_aug'])

        train_target_source_original = train_original[[self.target_col, self.source_col]].copy()
        train_target_source = self.create_augmentations(
            train_target_source_original,
            input_column=self.target_col,
        ).drop(columns=['is_aug'])
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
                "validation_examples": len(val_source_target),
            },
            "target_to_source": {
                "original_examples": original_target_source_size,
                "train_examples_before_augmentation": len(train_target_source_original),
                "augmented_examples": augmented_target_source_size,
                "augmentation_gain": augmented_target_source_size - len(train_target_source_original),
                "validation_examples": len(val_target_source),
            }
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

    # Paraphrasing arguments
    parser.add_argument('--enable_paraphrasing', action='store_true', help='Enable paraphrasing augmentation')
    parser.add_argument('--azure_endpoint', type=str, default=None, help='Azure OpenAI endpoint')
    parser.add_argument('--azure_deployment', type=str, default='gpt-4o', help='Azure OpenAI deployment name')
    parser.add_argument('--azure_api_key', type=str, default=None, help='Azure OpenAI API key')

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
    datasets = cleaner.process_dataset(
        file_path=args.data_path,
        sheets=args.sheets,
        source_col=args.source_col,
        target_col=args.target_col,
        test_size=args.test_size,
        seed=args.seed,
        enable_paraphrasing=args.enable_paraphrasing
    )

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
