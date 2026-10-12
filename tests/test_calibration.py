"""Synthetic calibration recovery and validation-only threshold behavior."""

import numpy as np
from scipy.special import expit

from smsspam.calibration import calibration_metrics, choose_thresholds, fit_temperature


def test_temperature_recovers_known_scale():
    rng = np.random.default_rng(42)
    latent = rng.normal(0, 2, 100_000)
    labels = rng.binomial(1, expit(latent))
    temperature = fit_temperature(latent * 2.5, labels)
    assert abs(temperature - 2.5) / 2.5 < 0.1


def test_perfectly_calibrated_synthetic_ece():
    probabilities = np.repeat(np.arange(0.05, 1, 0.1), 100)
    labels = np.concatenate([np.r_[np.ones(int(round(value * 100))), np.zeros(100 - int(round(value * 100)))] for value in np.arange(0.05, 1, 0.1)])
    assert calibration_metrics(labels, probabilities)["ece"] < 0.02


def test_threshold_respects_fpr_and_low_band_recall():
    labels = np.r_[np.zeros(100), np.ones(100)]
    probabilities = np.r_[np.repeat(0.2, 99), 0.8, np.repeat(0.9, 99), 0.1]
    result = choose_thresholds(labels, probabilities)
    assert result["ham_fpr"] <= 0.01
    assert result["fpr_constraint_met"]
    assert result["threshold_low"] <= result["threshold"]
    assert (probabilities[labels == 1] >= result["threshold_low"]).mean() >= 0.99


def test_threshold_fallback_is_explicit():
    result = choose_thresholds([0, 0, 1, 1], [0.99, 0.99, 0.99, 0.99])
    assert not result["fpr_constraint_met"]
    assert "minimum FPR" in result["fallback_reason"]
