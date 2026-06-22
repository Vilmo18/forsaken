# LRL-French_Models_Pipeline
A repository which contains the complete pipeline from data pre-processing to evaluation for fine-tuning machine translation models from low-resource languages to French.

## Data Pre-Processing 
- Loads data from Excel files with multiple sheets
- Cleans and preprocesses text data
- Handles numbers, dates, and special characters in French
- Creates augmented datasets through transliteration and diacritic removal
- Prepares train/validation splits for both translation directions
- Outputs Hugging Face Dataset compatible JSONL files

**Installation**

```bash
pip install pandas numpy datasets num2words babel unidecode
```

**Usage**
```
python Cleaning.py \
    --language LANGUAGE_NAME \
    --data_path path/to/your/data.xlsx \
    --output_dir ./datasets \
    --sheets SHEET_NAME \
    --source_col "Custom Source Column" \
    --target_col "Custom Target Column" \
    --test_size 0.2 \
    --seed 42
```
You can have multiple sheet names (do not separate with a comma just a white space)

## Model Fine-tuning
This script fine-tunes the NLLB model for translation between any language and French.

**Install Requirements**

```bash
pip install torch transformers datasets peft accelerate
```
**Usage**
```
python Finetune.py \
    --language LANGUAGE_NAME \
    --code LANGUAGE_CODE \
    --train_t2f path/to/train_lang2fr.jsonl \
    --val_t2f path/to/val_lang2fr.jsonl \
    --train_f2t path/to/train_fr2lang.jsonl \
    --val_f2t path/to/val_fr2lang.jsonl
```

To run the complete pipeline with the GPU-optimized training profile:

```bash
python3 lrl_fr-pipeline.py --dataset ewe --fast
```

Replace `ewe` with `mina` or `kabye`. The pipeline uses the NLLB 1.3B model.
Fast mode defaults to a batch size of 8 with 2 gradient-accumulation steps,
8 epochs, BF16 when supported (otherwise FP16), a 128-token limit, and keeps
the checkpoint with the best validation loss. The effective batch remains 16,
and the two translation directions are balanced to contribute equally.

The main training settings can be overridden from the command line:

```bash
python3 lrl_fr-pipeline.py --dataset ewe --fast \
    --model-id facebook/nllb-200-1.3B \
    --batch-size 8 \
    --gradient-accumulation-steps 2 \
    --epochs 8 \
    --max-length 128
```

## Model Evaluation

Evaluate your trained bilingual translation models using BLEU and METEOR metrics.

**Install Requirements**

```bash
pip install pandas torch transformers datasets evaluate tqdm
```

**Usage**
```
python Evaluate.py \
    --language LANGUAGE_NAME \
    --code LANGUAGE_CODE \
    --model_path PATH_TO_MODEL \
    --val_lang2fr PATH_TO_VAL_LANG2FR \
    --val_fr2lang PATH_TO_VAL_FR2LANG
```
