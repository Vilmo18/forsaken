from pathlib import Path
import pandas as pd
import argparse
import gc
import math
import random
import torch

from Cleaning import BilingualDataCleaner
from Finetune import BilingualFineTuner, DEFAULT_MODEL_ID, set_seed
from Evaluate import BilingualEvaluator
from Tracking import ExperimentTracker
from Backtranslation import BackTranslator, load_monolingual_texts


DATASET_DIR = "Dataset"
OUTPUT_DIR = "outputs"

LANGUAGE_CODES = {
    "ewe": "ewe_Latn",
    "kabye": "kbp_Latn",
    "mina": "gej_Latn",
    "tem": "tem_Latn",
}

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default=None)
parser.add_argument(
    "--fast",
    action="store_true",
    help="Use the GPU-optimized profile with larger batches and quality safeguards",
)
parser.add_argument("--batch-size", type=int, default=None)
parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
parser.add_argument("--epochs", type=int, default=None)
parser.add_argument("--max-length", type=int, default=None)
parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
parser.add_argument("--learning-rate", type=float, default=None)
parser.add_argument("--weight-decay", type=float, default=None)
parser.add_argument("--warmup-ratio", type=float, default=None)
parser.add_argument("--lr-scheduler-type", default=None)
parser.add_argument("--lora-r", type=int, default=None)
parser.add_argument("--lora-alpha", type=int, default=None)
parser.add_argument("--lora-dropout", type=float, default=None)
parser.add_argument("--early-stopping-patience", type=int, default=None)
parser.add_argument("--max-augmented-variants", type=int, default=None)
parser.add_argument("--source-max-augmented-variants", type=int, default=None)
parser.add_argument("--french-max-augmented-variants", type=int, default=None)
parser.add_argument("--language-to-french-weight", type=float, default=1.0)
parser.add_argument("--french-to-language-weight", type=float, default=1.0)
parser.add_argument("--monolingual-data", default=None)
parser.add_argument("--monolingual-column", default=None)
parser.add_argument(
    "--self-backtranslation",
    action="store_true",
    help="Generate synthetic French inputs from the Ewe targets already in the training split",
)
parser.add_argument("--backtranslation-model", default=None)
parser.add_argument("--backtranslation-batch-size", type=int, default=16)
parser.add_argument("--backtranslation-min-cycle-similarity", type=float, default=0.45)
parser.add_argument("--max-synthetic-samples", type=int, default=None)
parser.add_argument("--eval-num-beams", type=int, default=None)
parser.add_argument("--eval-max-new-tokens", type=int, default=None)
parser.add_argument("--eval-batch-size", type=int, default=None)
parser.add_argument("--eval-no-repeat-ngram-size", type=int, default=3)
args = parser.parse_args()

batch_size = args.batch_size or (16 if args.fast else 2)
gradient_accumulation_steps = args.gradient_accumulation_steps
if gradient_accumulation_steps is None:
    gradient_accumulation_steps = (
        max(1, math.ceil(16 / batch_size))
        if args.fast or args.batch_size is not None
        else 8
    )
num_train_epochs = args.epochs or 8
max_length = args.max_length or (128 if args.fast else 256)
max_augmented_variants = (
    args.max_augmented_variants
    if args.max_augmented_variants is not None
    else (5 if args.fast else 3)
)
source_max_augmented_variants = (
    args.source_max_augmented_variants
    if args.source_max_augmented_variants is not None
    else max_augmented_variants
)
french_max_augmented_variants = (
    args.french_max_augmented_variants
    if args.french_max_augmented_variants is not None
    else max_augmented_variants
)
lora_r = args.lora_r or 32
lora_alpha = args.lora_alpha or lora_r * 2
lora_dropout = 0.1 if args.lora_dropout is None else args.lora_dropout
learning_rate = args.learning_rate or (1e-4 if args.fast else 3e-5)
weight_decay = args.weight_decay if args.weight_decay is not None else (0.01 if args.fast else 0.0)
warmup_ratio = args.warmup_ratio if args.warmup_ratio is not None else (0.05 if args.fast else 0.0)
lr_scheduler_type = args.lr_scheduler_type or ("cosine" if args.fast else "linear")
early_stopping_patience = (
    args.early_stopping_patience
    if args.early_stopping_patience is not None
    else (3 if args.fast else None)
)
eval_num_beams = args.eval_num_beams or (5 if args.fast else 4)
eval_max_new_tokens = args.eval_max_new_tokens or max_length
eval_batch_size = args.eval_batch_size or (32 if args.fast else 25)

if min(batch_size, gradient_accumulation_steps, num_train_epochs, max_length) < 1:
    parser.error("batch size, accumulation steps, epochs, and max length must be positive")
if min(max_augmented_variants, source_max_augmented_variants, french_max_augmented_variants) < 0:
    parser.error("max augmented variants must be non-negative")
if args.backtranslation_batch_size < 1:
    parser.error("backtranslation batch size must be positive")
if args.max_synthetic_samples is not None and args.max_synthetic_samples < 1:
    parser.error("max synthetic samples must be positive")
if not 0 <= args.backtranslation_min_cycle_similarity <= 1:
    parser.error("backtranslation cycle similarity must be between 0 and 1")
if args.language_to_french_weight <= 0 or args.french_to_language_weight <= 0:
    parser.error("direction weights must be positive")
if min(lora_r, lora_alpha, eval_num_beams, eval_max_new_tokens, eval_batch_size) < 1:
    parser.error("LoRA rank/alpha and evaluation generation settings must be positive")
if not 0 <= lora_dropout < 1:
    parser.error("LoRA dropout must be in [0, 1)")
if learning_rate <= 0 or weight_decay < 0 or warmup_ratio < 0:
    parser.error("learning rate must be positive; weight decay and warmup ratio must be non-negative")
if early_stopping_patience is not None and early_stopping_patience < 1:
    parser.error("early stopping patience must be positive")
if args.eval_no_repeat_ngram_size < 0:
    parser.error("eval no-repeat ngram size must be non-negative")

dataset_files = Path(DATASET_DIR).glob("*.xlsx")


for file_path in dataset_files:
    if args.dataset is not None:
        if args.dataset.lower() not in file_path.stem.lower():
            continue
    print("\n" + "=" * 80)
    print(f"PROCESSING: {file_path.name}")
    print("=" * 80)

    # DETECT COLUMNS
    df = pd.read_excel(file_path, nrows=5)

    columns = df.columns.tolist()

    source_col = None
    target_col = None

    # target column

    if "Phrase équivalente en français" in columns:
        target_col = "Phrase équivalente en français"

    elif "French" in columns:
        target_col = "French"

    elif "french" in columns:
        target_col = "french"

    else:
        raise ValueError(f"No French column found in {file_path}")

    # source column

    if "Phrase en langue nationale" in columns:
        source_col = "Phrase en langue nationale"

    elif "Column 1" in columns:
        source_col = "Column 1"

    else:

        for col in columns:

            if col != target_col:
                source_col = col
                break

    if source_col is None:
        raise ValueError(f"Could not determine source column")

    print(f"Source column: {source_col}")
    print(f"Target column: {target_col}")


    # LANGUAGE DETECTION

    filename = file_path.stem.lower()
    language_key = None

    for lang in LANGUAGE_CODES:
        if lang in filename:
            language_key = lang
            break
            
    if language_key is None:
        raise ValueError(
            f"Couldn't detemine language from file: {file_path.name}"
        )

    language = language_key.capitalize()

    language_code = LANGUAGE_CODES[language_key]

    print(f"Language: {language}")
    print(f"Language code: {language_code}")

    tracker = ExperimentTracker(language=language_key)

    # OUTPUT DIRECTORY
    dataset_output = tracker.data_dir
    dataset_output.mkdir(parents=True, exist_ok=True)


    # CLEANING

    cleaner = BilingualDataCleaner(language)

    cleaning_result = cleaner.process_dataset(
        file_path=file_path,
        source_col=source_col,
        target_col=target_col,
        test_size=0.2,
        seed=42,
        enable_paraphrasing=False,
        max_augmented_variants=max_augmented_variants,
        source_max_augmented_variants=source_max_augmented_variants,
        target_max_augmented_variants=french_max_augmented_variants,
    )

    datasets = cleaning_result["datasets"]
    cleaning_stats = cleaning_result["stats"]

    tracker.log_cleaning_stats(cleaning_stats)

    run_config = {
    "language": language,
    "language_code": language_code,
    "source_column": source_col,
    "target_column": target_col,
    "test_size": 0.2,
    "seed": 42,
    "base_model": args.model_id,
    "fast_mode": args.fast,
    "batch_size": batch_size,
    "gradient_accumulation_steps": gradient_accumulation_steps,
    "effective_batch_size": batch_size * gradient_accumulation_steps,
    "num_train_epochs": num_train_epochs,
    "max_length": max_length,
    "max_augmented_variants": max_augmented_variants,
    "source_max_augmented_variants": source_max_augmented_variants,
    "french_max_augmented_variants": french_max_augmented_variants,
    "balance_directions": True,
    "language_to_french_weight": args.language_to_french_weight,
    "french_to_language_weight": args.french_to_language_weight,
    "learning_rate": learning_rate,
    "weight_decay": weight_decay,
    "warmup_ratio": warmup_ratio,
    "lr_scheduler_type": lr_scheduler_type,
    "lora_r": lora_r,
    "lora_alpha": lora_alpha,
    "lora_dropout": lora_dropout,
    "early_stopping_patience": early_stopping_patience,
    "eval_num_beams": eval_num_beams,
    "eval_max_new_tokens": eval_max_new_tokens,
    "eval_batch_size": eval_batch_size,
    "eval_no_repeat_ngram_size": args.eval_no_repeat_ngram_size,
    "monolingual_data": args.monolingual_data,
    "self_backtranslation": args.self_backtranslation,
    "backtranslation_model": args.backtranslation_model,
    "cleaning_stats": cleaning_stats
    }
    tracker.log_config(run_config)

    cleaned_paths = {}

    for name, dataset in datasets.items():

        path = tracker.save_dataset(name, dataset)

        cleaned_paths[name] = str(path)

        print(f"Saved: {path}")

    
    # FINETUNING

    set_seed()

    finetuner = BilingualFineTuner(
        language_name=language,
        language_code=language_code,
        model_id=args.model_id,
        tracker=tracker
    )

    train_t2f, val_t2f, train_f2t, val_f2t = finetuner.load_data(
        cleaned_paths[f"train_{language_key}2french"],
        cleaned_paths[f"val_{language_key}2french"],
        cleaned_paths[f"train_french2{language_key}"],
        cleaned_paths[f"val_french2{language_key}"]
    )

    if args.monolingual_data or args.self_backtranslation:
        if args.monolingual_data:
            monolingual_texts = load_monolingual_texts(
                args.monolingual_data,
                column=args.monolingual_column,
                max_samples=args.max_synthetic_samples,
                seed=42,
            )
            backtranslation_source = "external_monolingual_data"
        else:
            training_frame = train_f2t.to_pandas()
            monolingual_texts = []
            seen_monolingual = set()
            for value in training_frame[language].tolist():
                if pd.isna(value):
                    continue
                text = " ".join(str(value).strip().split())
                normalized = text.lower()
                if text and normalized not in seen_monolingual:
                    seen_monolingual.add(normalized)
                    monolingual_texts.append(text)
            random.Random(42).shuffle(monolingual_texts)
            if args.max_synthetic_samples is not None:
                monolingual_texts = monolingual_texts[:args.max_synthetic_samples]
            backtranslation_source = "training_targets"

        if not monolingual_texts:
            raise ValueError("No usable monolingual sentences were found")

        backtranslation_model = args.backtranslation_model
        if backtranslation_model is None:
            prior_models = sorted(
                (
                    path for path in Path("outputs/runs").glob(
                        f"{language_key}_*/model-merged"
                    )
                    if (path / "config.json").exists()
                ),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
            backtranslation_model = str(prior_models[0]) if prior_models else args.model_id

        print(
            f"Back-translating {len(monolingual_texts)} {language} sentences "
            f"with {backtranslation_model}"
        )
        backtranslator = BackTranslator(
            model_path=backtranslation_model,
            language_name=language,
            language_code=language_code,
        )
        synthetic_pairs = backtranslator.generate(
            monolingual_texts,
            batch_size=args.backtranslation_batch_size,
            max_length=max_length,
            min_cycle_similarity=args.backtranslation_min_cycle_similarity,
        )
        synthetic_path = tracker.data_dir / f"synthetic_french2{language_key}.jsonl"
        synthetic_pairs.to_json(
            synthetic_path,
            orient="records",
            lines=True,
            force_ascii=False,
        )
        train_f2t, synthetic_added = finetuner.add_synthetic_french_to_language(
            train_f2t,
            synthetic_pairs,
        )
        run_config["backtranslation"] = {
            "source": backtranslation_source,
            "model": backtranslation_model,
            "monolingual_examples": len(monolingual_texts),
            "generated_pairs": len(synthetic_pairs),
            "unique_pairs_added": synthetic_added,
            "min_cycle_similarity": args.backtranslation_min_cycle_similarity,
            "synthetic_dataset": str(synthetic_path),
        }
        tracker.log_config(run_config)
        print(f"Added {synthetic_added} unique synthetic French->{language} pairs")

        del backtranslator
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    finetuner.setup_model_and_tokenizer()

    finetuner.setup_lora(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    )

    train_ds, eval_ds = finetuner.tokenize_data(
        train_t2f,
        val_t2f,
        train_f2t,
        val_f2t,
        max_length=max_length,
        language_to_french_weight=args.language_to_french_weight,
        french_to_language_weight=args.french_to_language_weight,
    )

    model_output = tracker.model_dir
    merged_output = tracker.merged_dir

    training_args = {
        "per_device_train_batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "num_train_epochs": num_train_epochs,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "warmup_ratio": warmup_ratio,
        "lr_scheduler_type": lr_scheduler_type,
    }
    if early_stopping_patience is not None:
        training_args["early_stopping_patience"] = early_stopping_patience
        training_args["load_best_model_at_end"] = True
        training_args["metric_for_best_model"] = "eval_loss"
        training_args["greater_is_better"] = False

    if args.fast:
        training_args.update({
            "per_device_eval_batch_size": batch_size * 2,
            "save_strategy": "epoch",
            "eval_strategy": "epoch",
            "load_best_model_at_end": True,
            "metric_for_best_model": "eval_loss",
            "greater_is_better": False,
            "save_total_limit": 2,
            "logging_steps": 50,
            "warmup_steps": 0,
            "group_by_length": True,
            "dataloader_num_workers": 4,
        })

    finetuner.train(
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        output_dir=str(model_output),
        **training_args,
    )

    finetuner.save_models(adapter_dir=str(tracker.model_dir / "adapter"), 
                          merged_dir=str(tracker.merged_dir))

    
    # EVALUATION

    merged_model_path = str(tracker.merged_dir)

    evaluator = BilingualEvaluator(
        language_name=language,
        language_code=language_code,
        model_path=merged_model_path,
        model_type="merged",
        base_model_id=args.model_id,
        max_input_length=max_length,
        max_new_tokens=eval_max_new_tokens,
        num_beams=eval_num_beams,
        no_repeat_ngram_size=args.eval_no_repeat_ngram_size,
        eval_batch_size=eval_batch_size,
    )

    evaluator.load_model()

    val_lang2fr = pd.read_json(
        cleaned_paths[f"val_{language_key}2french"],
        lines=True
    )

    val_fr2lang = pd.read_json(
        cleaned_paths[f"val_french2{language_key}"],
        lines=True
    )

    lang2fr_results, lang2fr_preds, lang2fr_src, lang2fr_refs = evaluator.evaluate_direction(
        val_df=val_lang2fr,
        direction=f"{language}-to-French",
        translate_fn=evaluator.translate_lang_to_french,
        batch_translate_fn=evaluator.translate_lang_to_french_batch,
        source_col=language,
        target_col="French"
    )

    fr2lang_results, fr2lang_preds, fr2lang_src, fr2lang_refs = evaluator.evaluate_direction(
        val_df=val_fr2lang,
        direction=f"French-to-{language}",
        translate_fn=evaluator.translate_french_to_lang,
        batch_translate_fn=evaluator.translate_french_to_lang_batch,
        source_col="French",
        target_col=language
    )

    tracker.log_metrics(
    "lang2fr",
    lang2fr_results
    )

    tracker.log_metrics(
        "fr2lang",
        fr2lang_results
    )

    tracker.log_predictions(
    "predictions_lang2fr",
    [
        {
            "source": src,
            "reference": ref,
            "prediction": pred
        }
        for src, ref, pred in zip(
            lang2fr_src,
            lang2fr_refs,
            lang2fr_preds
        )
    ]
    )

    tracker.log_predictions(
    "predictions_fr2lang",
    [
        {
            "source": src,
            "reference": ref,
            "prediction": pred
        }
        for src, ref, pred in zip(
            fr2lang_src,
            fr2lang_refs,
            fr2lang_preds
        )
    ]
    )

    print("\nFINAL RESULTS")

    print(lang2fr_results)
    print(fr2lang_results)

    print("\nDONE")
