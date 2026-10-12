"""Eight mandatory dataset invariants and label-shortcut diagnostics (guide §5.4)."""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

from smsspam.data.build import ARTIFACT_PUNCT, LEMMA_P1, dedup_key, is_raw_format, sha256
from smsspam.preprocessing import normalize

PTB_CONTRACTION = re.compile(r"\w (?:n't|'s|'re|'m|'ve|'ll|'d)\b", re.I)
PTB_QUOTES = re.compile(r"``|''")
MOJIBAKE = re.compile(r"Ã|Â|â€")
RAW_X = re.compile(r"\bX{2,}\b|(?<![A-Za-z'\-])X(?![A-Za-z'\-])")


def artifact_features(texts: pd.Series) -> np.ndarray:
    """Use only the eight formatting signals specified in the execution guide."""
    return np.array([
        [
            text.islower(), not bool(ARTIFACT_PUNCT.search(text)), not is_raw_format(text),
            bool(PTB_CONTRACTION.search(text)), bool(PTB_QUOTES.search(text)),
            bool(LEMMA_P1.search(text)), bool(re.search(r"\b(?:have|has|had) be\b", text, re.I)),
            bool(MOJIBAKE.search(text)),
        ]
        for text in texts
    ], dtype=float)


def cv_accuracy(features: np.ndarray, labels: pd.Series, seed: int) -> dict:
    """Diagnostic classifiers are separate from the predictive-model training."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    scores = cross_val_score(LogisticRegression(max_iter=1000, random_state=seed), features, labels, cv=cv, scoring="accuracy")
    return {"accuracy": float(scores.mean()), "std": float(scores.std()), "folds": scores.tolist()}


def audit_dataset(processed: Path, seed: int = 42) -> dict:
    """Run the complete audit and return machine-readable pass/fail evidence."""
    corpus_path = processed / "sms_spam_curated_v1.csv"
    full = pd.read_csv(corpus_path, keep_default_na=False)
    pool = pd.read_csv(processed / "unused_ham_pool.csv", keep_default_na=False)
    parts = {name: pd.read_csv(processed / f"{name}.csv", keep_default_na=False) for name in ("train", "val", "test")}
    checks = {}
    diagnostics = {}
    schema_ok = {"text", "label", "source", "split"}.issubset(full.columns)
    schema_ok = schema_ok and not full.text.str.strip().eq("").any() and full.label.isin([0, 1]).all()
    schema_ok = schema_ok and full.split.isin(parts).all() and set(full.split) == set(parts)
    schema_ok = schema_ok and all(part.split.eq(name).all() and part.label.isin([0, 1]).all() for name, part in parts.items())
    schema_ok = schema_ok and pool.label.eq(0).all() and not pool.text.str.strip().eq("").any()
    checks["valid_schema"] = bool(schema_ok)
    if not schema_ok:
        return {"passed": False, "checks": checks, "error": "Invalid text/label/split schema"}

    counts = {name: part.label.value_counts().to_dict() for name, part in {"all": full, **parts}.items()}
    checks["balanced_classes"] = all(set(count) == {0, 1} and abs(count[0] - count[1]) <= (0 if name == "all" else 1) for name, count in counts.items())
    keys = full.text.map(dedup_key)
    part_keys = [set(part.text.map(dedup_key)) for part in parts.values()]
    consistent = set.union(*part_keys) == set(keys) and sum(map(len, parts.values())) == len(full)
    disjoint = all(part_keys[i].isdisjoint(part_keys[j]) for i in range(3) for j in range(i + 1, 3))
    checks["unique_disjoint_keys"] = bool(not keys.duplicated().any() and disjoint and consistent)

    artifacts = {
        "format_stripped": int((~full.text.map(is_raw_format)).sum()),
        "lemmatisation_p1": int(full.text.map(lambda text: bool(LEMMA_P1.search(text))).sum()),
        "ptb_contraction": int(full.text.map(lambda text: bool(PTB_CONTRACTION.search(text))).sum()),
        "ptb_quotes": int(full.text.map(lambda text: bool(PTB_QUOTES.search(text))).sum()),
        "mojibake": int(full.text.map(lambda text: bool(MOJIBAKE.search(text))).sum()),
    }
    checks["no_publisher_artifacts"] = not any(artifacts.values())
    diagnostics["artifact_counts"] = artifacts
    diagnostics["artifact_only"] = cv_accuracy(artifact_features(full.text), full.label, seed)
    checks["artifact_accuracy_at_most_056"] = diagnostics["artifact_only"]["accuracy"] <= 0.56
    normalized = full.text.map(normalize)
    masks = np.array([[bool(RAW_X.search(text)), "<#>" in text] for text in normalized], dtype=float)
    diagnostics["normalized_mask_only"] = cv_accuracy(masks, full.label, seed)
    checks["mask_accuracy_at_most_055"] = diagnostics["normalized_mask_only"]["accuracy"] <= 0.55
    diagnostics["source_only"] = cv_accuracy(pd.get_dummies(full.source, dtype=float).to_numpy(), full.label, seed)
    checks["source_accuracy_at_most_056"] = diagnostics["source_only"]["accuracy"] <= 0.56
    checks["unused_ham_disjoint"] = set(pool.text.map(dedup_key)).isdisjoint(keys)
    return {
        "passed": bool(all(checks.values())), "checks": checks, "diagnostics": diagnostics,
        "class_counts": counts, "sha256_corpus": sha256(corpus_path), "seed": seed,
        "note": "Diagnostic CV is explicitly required by §5.4. It does not tune the predictive models.",
    }


def save_audit(processed: Path, output: Path, seed: int = 42) -> dict:
    report = audit_dataset(processed, seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
