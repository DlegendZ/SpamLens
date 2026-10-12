"""Validation-only temperature scaling, risk thresholds, and calibration diagnostics."""

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit
from sklearn.metrics import f1_score


def negative_log_likelihood(labels, logits) -> float:
    labels, logits = np.asarray(labels), np.asarray(logits)
    return float(np.mean(np.logaddexp(0, logits) - labels * logits))


def fit_temperature(logits, labels) -> float:
    """Fit T in the exact bounded interval prescribed in guide §8.1."""
    logits, labels = np.asarray(logits), np.asarray(labels)
    result = minimize_scalar(lambda temperature: negative_log_likelihood(labels, logits / temperature), bounds=(0.05, 20), method="bounded")
    if not result.success:
        raise RuntimeError("Temperature optimization did not converge")
    return float(result.x)


def calibration_metrics(labels, probabilities, bins: int = 15) -> dict:
    labels, probabilities = np.asarray(labels), np.asarray(probabilities)
    indices = np.minimum((probabilities * bins).astype(int), bins - 1)
    ece = 0.0
    rows = []
    for index in range(bins):
        mask = indices == index
        count = int(mask.sum())
        confidence = float(probabilities[mask].mean()) if count else None
        frequency = float(labels[mask].mean()) if count else None
        if count:
            ece += count / len(labels) * abs(confidence - frequency)
        rows.append({"bin": index, "count": count, "mean_probability": confidence, "spam_frequency": frequency})
    p = np.clip(probabilities, 1e-12, 1 - 1e-12)
    return {"ece": float(ece), "brier": float(np.mean((probabilities - labels) ** 2)),
            "nll": float(-np.mean(labels * np.log(p) + (1 - labels) * np.log1p(-p))), "bins": rows}


def threshold_metrics(labels, probabilities, threshold: float) -> dict:
    labels, probabilities = np.asarray(labels), np.asarray(probabilities)
    predicted = probabilities >= threshold
    ham, spam = labels == 0, labels == 1
    return {"threshold": float(threshold), "macro_f1": float(f1_score(labels, predicted, average="macro", zero_division=0)),
            "ham_fpr": float(predicted[ham].mean()), "spam_recall": float(predicted[spam].mean())}


def choose_thresholds(labels, probabilities, max_ham_fpr: float = 0.01, min_spam_recall: float = 0.99) -> dict:
    """Use the 0.01 tau grid; break ties deterministically without test data."""
    candidates = [threshold_metrics(labels, probabilities, float(threshold)) for threshold in np.round(np.arange(0.05, 0.951, 0.01), 2)]
    feasible = [candidate for candidate in candidates if candidate["ham_fpr"] <= max_ham_fpr]
    if feasible:
        selected = max(feasible, key=lambda item: (item["macro_f1"], -item["ham_fpr"], -abs(item["threshold"] - 0.5), item["threshold"]))
    else:
        selected = min(candidates, key=lambda item: (item["ham_fpr"], -item["macro_f1"], -item["threshold"]))
    threshold = selected["threshold"]
    # Include zero: a suspicious band must still attain the specified 99% recall.
    low_candidates = [candidate for candidate in np.round(np.arange(0, threshold + 0.001, 0.01), 2) if threshold_metrics(labels, probabilities, float(candidate))["spam_recall"] >= min_spam_recall]
    low = float(max(low_candidates)) if low_candidates else threshold
    return {**selected, "threshold_low": min(low, threshold), "fpr_constraint_met": bool(feasible),
            "selection_split": "val", "fallback_reason": None if feasible else "No grid threshold met the ham FPR constraint; selected minimum FPR"}


def calibrate_logits(logits, labels, settings: dict) -> dict:
    temperature = fit_temperature(logits, labels)
    probabilities = expit(np.asarray(logits) / temperature)
    before = calibration_metrics(labels, expit(logits), settings["ece_bins"])
    after = calibration_metrics(labels, probabilities, settings["ece_bins"])
    thresholds = choose_thresholds(labels, probabilities, settings["max_ham_fpr"], settings["min_spam_recall_low"])
    return {"temperature": temperature, **thresholds, "val_ece_before": before["ece"], "val_ece_after": after["ece"], "validation_before": before, "validation_after": after}
