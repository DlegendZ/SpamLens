"""Fine-tune DistilBERT with validation-only early stopping and dynamic padding."""

import inspect
import json
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

from smsspam.config import Config
from smsspam.data.build import sha256
from smsspam.preprocessing import normalize
from smsspam.viz import plt, save_figure


def tokenized_split(path, tokenizer, max_length: int, expected_split: str) -> Dataset:
    frame = pd.read_csv(path)
    if not frame.split.eq(expected_split).all():
        raise ValueError(f"Expected only {expected_split} rows in {path.name}")
    dataset = Dataset.from_dict({"text": frame.text.map(normalize).tolist(), "labels": frame.label.tolist()})
    return dataset.map(lambda batch: tokenizer(batch["text"], truncation=True, max_length=max_length), batched=True, remove_columns=["text"])


def compute_metrics(prediction) -> dict:
    predicted = np.argmax(prediction.predictions, axis=1)
    return {"macro_f1": f1_score(prediction.label_ids, predicted, average="macro")}


def make_trainer(config: Config, tokenizer, train, val, batch_size: int, accumulation: int) -> Trainer:
    """Trainer pattern adapted from https://huggingface.co/docs/transformers/main_classes/trainer."""
    settings = config.values["distilbert"]
    model = AutoModelForSequenceClassification.from_pretrained(settings["checkpoint"], num_labels=2, id2label={0: "ham", 1: "spam"}, label2id={"ham": 0, "spam": 1}, cache_dir=config.root / ".cache" / "huggingface")
    strategy_key = "eval_strategy" if "eval_strategy" in inspect.signature(TrainingArguments).parameters else "evaluation_strategy"
    kwargs = {
        strategy_key: "epoch", "save_strategy": "epoch", "logging_strategy": "steps", "logging_steps": 50,
        "learning_rate": float(settings["lr"]), "weight_decay": settings["weight_decay"],
        "warmup_ratio": settings["warmup_ratio"], "lr_scheduler_type": "linear",
        "per_device_train_batch_size": batch_size, "per_device_eval_batch_size": batch_size,
        "gradient_accumulation_steps": accumulation, "num_train_epochs": settings["epochs"],
        "fp16": bool(settings["fp16"] and torch.cuda.is_available()), "seed": config.seed,
        "data_seed": config.seed, "load_best_model_at_end": True, "metric_for_best_model": "macro_f1",
        "greater_is_better": True, "save_total_limit": 1, "report_to": [], "dataloader_num_workers": 0,
        "output_dir": str(config.path("models") / "checkpoints" / f"batch_{batch_size}"),
    }
    arguments = TrainingArguments(**kwargs)
    return Trainer(model=model, args=arguments, train_dataset=train, eval_dataset=val,
                   processing_class=tokenizer, data_collator=DataCollatorWithPadding(tokenizer),
                   compute_metrics=compute_metrics,
                   callbacks=[EarlyStoppingCallback(early_stopping_patience=settings["patience"])])


def training_figure(history: list[dict], output) -> None:
    losses = [entry for entry in history if "loss" in entry]
    evaluations = [entry for entry in history if "eval_macro_f1" in entry]
    figure, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].plot([entry["epoch"] for entry in losses], [entry["loss"] for entry in losses], "k-o", markersize=3)
    axes[0].set(xlabel="Epoch", ylabel="Training loss", title="DistilBERT training loss")
    axes[1].plot([entry["epoch"] for entry in evaluations], [entry["eval_macro_f1"] for entry in evaluations], "k-s")
    axes[1].set(xlabel="Epoch", ylabel="Validation macro-F1", title="Validation selection (no test data)")
    save_figure(figure, output)


def train_distilbert(config: Config) -> dict:
    """Train on train.csv only; val.csv selects the best checkpoint."""
    set_seed(config.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    settings = config.values["distilbert"]
    tokenizer = AutoTokenizer.from_pretrained(settings["checkpoint"], cache_dir=config.root / ".cache" / "huggingface")
    processed = config.path("processed")
    train = tokenized_split(processed / "train.csv", tokenizer, settings["max_length"], "train")
    val = tokenized_split(processed / "val.csv", tokenizer, settings["max_length"], "val")
    batch_size, accumulation = settings["batch_size"], 1
    trainer = make_trainer(config, tokenizer, train, val, batch_size, accumulation)
    started = perf_counter()
    try:
        trainer.train()
    except torch.OutOfMemoryError:
        del trainer
        torch.cuda.empty_cache()
        set_seed(config.seed)
        batch_size, accumulation = 16, 2
        trainer = make_trainer(config, tokenizer, train, val, batch_size, accumulation)
        print("CUDA OOM: restarting at batch 16 with accumulation 2", flush=True)
        trainer.train()
    output = config.path("models") / "distilbert"
    trainer.save_model(str(output))
    tokenizer.save_pretrained(output)
    validation = trainer.evaluate()
    report = {
        "seed": config.seed, "checkpoint": settings["checkpoint"],
        "checkpoint_revision": getattr(trainer.model.config, "_commit_hash", None),
        "training_seconds": perf_counter() - started, "cuda": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "batch_size": batch_size, "gradient_accumulation_steps": accumulation,
        "evaluation_strategy_key": "eval_strategy" if "eval_strategy" in inspect.signature(TrainingArguments).parameters else "evaluation_strategy",
        "validation_metrics": validation, "history": trainer.state.log_history,
        "sha256_train": sha256(processed / "train.csv"), "sha256_val": sha256(processed / "val.csv"),
        "selection": "best validation macro-F1; test split not loaded",
    }
    encoded = json.dumps(report, indent=2) + "\n"
    (output / "training_log.json").write_text(encoded, encoding="utf-8")
    (config.path("reports") / "metrics" / "training_distilbert.json").write_text(encoded, encoding="utf-8")
    training_figure(report["history"], config.path("reports") / "figures" / "F10_training_curves.png")
    print(f"Best validation macro-F1: {validation['eval_macro_f1']:.4f}", flush=True)
    if validation["eval_macro_f1"] < 0.98:
        raise SystemExit("T10 validation acceptance criterion not met; investigate without loading test data.")
    return report
