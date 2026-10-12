"""Uniform calibrated prediction interface for evaluation and the application."""

import json
from collections.abc import Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Literal, Protocol

import numpy as np
from scipy.special import expit

from smsspam.config import load_config
from smsspam.models.runtime import RawScorer

ModelName = Literal["distilbert", "tfidf_lr", "tfidf_nb"]
RiskBand = Literal["likely_spam", "suspicious", "likely_ham"]


@dataclass(frozen=True)
class Prediction:
    label: Literal["spam", "ham"]
    spam_probability: float
    risk_band: RiskBand
    model_name: str
    latency_ms: float


class Predictor(Protocol):
    name: str

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray: ...
    def predict(self, texts: Sequence[str]) -> list[Prediction]: ...


class ModelPredictor:
    """Calibrated probability and risk bands with immutable validation thresholds."""

    def __init__(self, name: ModelName):
        self.name = name
        config = load_config()
        calibration = json.loads((config.path("models") / "calibration.json").read_text(encoding="utf-8"))
        self.settings = calibration[name]
        self.scorer = RawScorer(name, config)

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        return expit(self.scorer.logits(texts) / self.settings["temperature"])

    def predict(self, texts: Sequence[str]) -> list[Prediction]:
        started = perf_counter()
        probabilities = self.predict_proba(texts)
        latency = (perf_counter() - started) * 1000 / max(1, len(probabilities))
        results = []
        for probability in probabilities:
            spam = probability >= self.settings["threshold"]
            band: RiskBand = "likely_spam" if spam else "suspicious" if probability >= self.settings["threshold_low"] else "likely_ham"
            results.append(Prediction("spam" if spam else "ham", float(probability), band, self.name, latency))
        return results


def load_predictor(name: ModelName) -> ModelPredictor:
    """Load one CPU predictor; callers supply raw text, normalization is internal."""
    return ModelPredictor(name)
