from pathlib import Path
import pandas as pd
import argparse
import math

from Cleaning import BilingualDataCleaner
from Finetune import BilingualFineTuner, DEFAULT_MODEL_ID, set_seed
from Evaluate import BilingualEvaluator
from Tracking import ExperimentTracker


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
args = parser.parse_args()

batch_size = args.batch_size or (8 if args.fast else 2)
gradient_accumulation_steps = args.gradient_accumulation_steps
if gradient_accumulation_steps is None:
    gradient_accumulation_steps = (
        max(1, math.ceil(16 / batch_size))
        if args.fast or args.batch_size is not None
        else 8
    )
num_train_epochs = args.epochs or 8
max_length = args.max_length or (128 if args.fast else 256)

if min(batch_size, gradient_accumulation_steps, num_train_epochs, max_length) < 1:
    parser.error("batch size, accumulation steps, epochs, and max length must be positive")

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
        enable_paraphrasing=False
    )

    datasets = cleaning_result["datasets"]
    cleaning_stats = cleaning_result["stats"]

    tracker.log_cleaning_stats(cleaning_stats)

    tracker.log_config({
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
    "balance_directions": True,
    "cleaning_stats": cleaning_stats
    })

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

    finetuner.setup_model_and_tokenizer()

    finetuner.setup_lora()

    train_ds, eval_ds = finetuner.tokenize_data(
        train_t2f,
        val_t2f,
        train_f2t,
        val_f2t,
        max_length=max_length,
    )

    model_output = tracker.model_dir
    merged_output = tracker.merged_dir

    training_args = {
        "per_device_train_batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "num_train_epochs": num_train_epochs,
    }

    if args.fast:
        training_args.update({
            "per_device_eval_batch_size": batch_size * 2,
            "learning_rate": 1e-4,
            "weight_decay": 0.01,
            "save_strategy": "epoch",
            "eval_strategy": "epoch",
            "load_best_model_at_end": True,
            "metric_for_best_model": "eval_loss",
            "greater_is_better": False,
            "save_total_limit": 2,
            "logging_steps": 50,
            "warmup_steps": 0,
            "warmup_ratio": 0.05,
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
        source_col=language,
        target_col="French"
    )

    fr2lang_results, fr2lang_preds, fr2lang_src, fr2lang_refs = evaluator.evaluate_direction(
        val_df=val_fr2lang,
        direction=f"French-to-{language}",
        translate_fn=evaluator.translate_french_to_lang,
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
