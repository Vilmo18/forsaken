# LRL-French_Models_Pipeline
A repository which contains the complete pipeline from data pre-processing to evaluation for fine-tuning machine translation models from low-resource languages to French.

## Data Pre-Processing 
- Loads data from Excel files with multiple sheets
- Cleans and preprocesses text data
- Handles numbers, dates, and special characters in French
- Creates bounded training-only augmentations through transliteration,
  diacritic removal, typography normalization, and punctuation variants
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

Replace `ewe` with `mina` or `kabye`. The pipeline uses the NLLB distilled 600M
model. Fast mode defaults to a batch size of 16, 8 epochs, BF16 when supported
(otherwise FP16), a 128-token limit, and keeps the checkpoint with the best
validation loss. Augmentation is applied only to the training split, and the
two translation directions are balanced to contribute equally.

The main training settings can be overridden from the command line:

```bash
python3 lrl_fr-pipeline.py --dataset ewe --fast \
    --model-id facebook/nllb-200-distilled-600M \
    --batch-size 16 \
    --gradient-accumulation-steps 1 \
    --epochs 8 \
    --max-length 128 \
    --max-augmented-variants 3
```

### Back-translation with monolingual data

To let the pipeline augment its own training split without an external file:

```bash
python3 lrl_fr-pipeline.py --dataset ewe --fast --self-backtranslation
```

This uses each unique Ewe training target to generate an alternative French
input. The most recent completed Ewe model under `outputs/runs` is selected as
the generator; when none exists, the configured base model is used.

Put one original low-resource-language sentence per line in a UTF-8 text file,
for example `Dataset/ewe_monolingual.txt`. Then run:

```bash
python3 lrl_fr-pipeline.py --dataset ewe --fast \
    --monolingual-data Dataset/ewe_monolingual.txt \
    --backtranslation-model outputs/runs/PRIOR_EWE_RUN/model-merged \
    --backtranslation-batch-size 16 \
    --backtranslation-min-cycle-similarity 0.45 \
    --max-synthetic-samples 5000
```

The back-translation model generates synthetic French sources while the
original Ewe sentences remain the trusted targets. Empty, copied,
length-mismatched, cycle-inconsistent, and duplicate pairs are filtered before
fine-tuning. If `--backtranslation-model` is omitted, the configured base model
is used. Set the cycle-similarity threshold to `0` to disable round-trip
filtering.

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
