"""Lightweight raw-logit model loading shared by calibration and inference."""

import os
from collections.abc import Sequence

import joblib
import numpy as np
from scipy.special import logit

from smsspam.config import Config
from smsspam.preprocessing import normalize

MODEL_NAMES = ("tfidf_nb", "tfidf_lr", "distilbert")


class RawScorer:
    """Return uncalibrated binary logit differences for raw user text."""

    def __init__(self, name: str, config: Config):
        if name not in MODEL_NAMES:
            raise ValueError(f"Unknown model: {name}")
        self.name, self.config = name, config
        if name != "distilbert":
            self.model = joblib.load(config.path("models") / f"{name}.joblib")
        else:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self.torch = torch
            torch.set_num_threads(min(config.values.get("inference", {}).get("cpu_threads", 12), os.cpu_count() or 1))
            local = config.path("models") / "distilbert"
            location = str(local) if (local / "config.json").exists() else config.values.get("hf_model_id")
            if not location:
                raise FileNotFoundError("DistilBERT weights are absent and hf_model_id is not configured")
            self.tokenizer = AutoTokenizer.from_pretrained(location)
            self.model = AutoModelForSequenceClassification.from_pretrained(location).to("cpu").eval()

    def logits(self, texts: Sequence[str]) -> np.ndarray:
        texts = list(texts)
        if not texts:
            return np.empty(0, dtype=float)
        if self.name != "distilbert":
            raw = self.model.predict_proba(texts)[:, 1]
            return logit(np.clip(raw, 1e-6, 1 - 1e-6))
        logits = []
        length = self.config.values["distilbert"]["max_length"]
        with self.torch.inference_mode():
            for start in range(0, len(texts), 32):
                normalized = [normalize(text) for text in texts[start:start + 32]]
                encoded = self.tokenizer(normalized, truncation=True, padding=True, max_length=length, return_tensors="pt")
                outputs = self.model(**encoded).logits
                logits.extend((outputs[:, 1] - outputs[:, 0]).numpy().astype(float))
        return np.asarray(logits)
