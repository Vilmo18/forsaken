from pathlib import Path
import json
from datetime import datetime
import uuid


class ExperimentTracker:
    """
    A tracking system for each of the runs of the pipeline
    """

    def __init__(self, base_dir="outputs/runs", language=None):
        self.base_dir = Path(base_dir)

        self.language = language
        self.run_id = self._generate_run_id()

        self.run_path = self.base_dir / self.run_id

        #main folders
        self.data_dir = self.run_path / "datasets"
        self.model_dir = self.run_path / "model"
        self.merged_dir = self.run_path / "model-merged"
        self.eval_dir = self.run_path / "eval"

        for d in [self.data_dir, self.model_dir, self.merged_dir, self.eval_dir]:
            d.mkdir(parents=True, exist_ok=True)

        self.config = {}
        self.metrics = {}

    def _generate_run_id(self):
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        uid = uuid.uuid4().hex[:6]
        lang = self.language if self.language else "run"
        return f"{lang}_{ts}_{uid}"

    # CONFIG
    def log_config(self, config: dict):
        self.config = config
        self._write_json(self.run_path / "config.json", config)

    # METRICS
    def log_metrics(self, name: str, metrics: dict):
        self.metrics[name] = metrics
        self._write_json(self.eval_dir / "metrics.json", self.metrics)

    def log_training(self, training_result):
        self._write_json(self.eval_dir / "training.json", training_result)    

    # DATASETS
    def save_dataset(self, name: str, df):
        """
        name examples:
        train_kabye2french
        val_french2kabye
        """
        path = self.data_dir / f"{name}.jsonl"
        df.to_json(path, orient="records", lines=True, force_ascii=False)
        return path
    
    def log_cleaning_stats(self, stats):
        path = self.run_path / "cleaning_stats.json"

        with open(path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

    # MODELS
    def get_model_dir(self):
        return self.model_dir

    def get_merged_dir(self):
        return self.merged_dir

    # EVAL
    def log_predictions(self, name: str, data):
        """
        name examples:
        predictions_lang2fr
        predictions_fr2lang
        """
        path = self.eval_dir / f"{name}.jsonl"

        import pandas as pd
        pd.DataFrame(data).to_json(
            path,
            orient="records",
            lines=True,
            force_ascii=False
        )
        return path

    def log_text(self, filename: str, text: str):
        (self.eval_dir / filename).write_text(text, encoding="utf-8")

    
    def _write_json(self, path: Path, obj: dict):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)