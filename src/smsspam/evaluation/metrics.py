"""Binary metrics, seeded bootstrap intervals, and exact paired McNemar tests."""

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

from smsspam.calibration import calibration_metrics


def classification_metrics(labels, probabilities, threshold: float) -> dict:
    labels, probabilities = np.asarray(labels), np.asarray(probabilities)
    predicted = (probabilities >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predicted, labels=[0, 1], zero_division=0)
    matrix = confusion_matrix(labels, predicted, labels=[0, 1])
    return {
        "accuracy": float(accuracy_score(labels, predicted)), "macro_f1": float(f1.mean()),
        "ham_precision": float(precision[0]), "ham_recall": float(recall[0]), "ham_f1": float(f1[0]),
        "spam_precision": float(precision[1]), "spam_recall": float(recall[1]), "spam_f1": float(f1[1]),
        "ham_fpr": float(matrix[0, 1] / max(1, matrix[0].sum())),
        "roc_auc": float(roc_auc_score(labels, probabilities)), "pr_auc": float(average_precision_score(labels, probabilities)),
        "confusion_matrix": matrix.tolist(), "calibration": calibration_metrics(labels, probabilities),
        "threshold": threshold, "n": len(labels),
    }


def bootstrap_intervals(labels, predicted, resamples: int = 1000, seed: int = 42) -> dict:
    """Ordinary paired-row bootstrap; compute binary metrics from confusion counts."""
    labels, predicted = np.asarray(labels), np.asarray(predicted)
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(labels), size=(resamples, len(labels)))
    actual, guessed = labels[samples], predicted[samples]
    tp = ((actual == 1) & (guessed == 1)).sum(axis=1)
    tn = ((actual == 0) & (guessed == 0)).sum(axis=1)
    fp = ((actual == 0) & (guessed == 1)).sum(axis=1)
    fn = ((actual == 1) & (guessed == 0)).sum(axis=1)
    values = {
        "macro_f1": (2 * tp / np.maximum(1, 2 * tp + fp + fn) + 2 * tn / np.maximum(1, 2 * tn + fp + fn)) / 2,
        "spam_recall": tp / np.maximum(1, tp + fn), "ham_fpr": fp / np.maximum(1, fp + tn),
    }
    return {name: {"lower": float(np.percentile(values, 2.5)), "upper": float(np.percentile(values, 97.5)), "resamples": resamples, "seed": seed} for name, values in values.items()}


def mcnemar(labels, prediction_a, prediction_b) -> dict:
    labels = np.asarray(labels)
    correct_a, correct_b = np.asarray(prediction_a) == labels, np.asarray(prediction_b) == labels
    b = int((correct_a & ~correct_b).sum())
    c = int((~correct_a & correct_b).sum())
    return {"a_correct_b_wrong": b, "a_wrong_b_correct": c,
            "p_value": float(binomtest(b, b + c, 0.5).pvalue) if b + c else 1.0, "method": "exact binomial, two-sided"}
