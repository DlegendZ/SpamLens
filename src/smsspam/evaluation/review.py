"""Export private manual-review queues and finalize the human error taxonomy."""

import json
from pathlib import Path

import pandas as pd

from smsspam.config import Config

CATEGORIES = {
    "transactional_otp": "Transactional / OTP resembles promotion",
    "short_ambiguous": "Short or ambiguous message",
    "conversational_spam": "Conversational spam / romance / job scam",
    "obfuscated": "Obfuscated wording",
    "code_mixed": "Non-English or code-mixed",
    "probable_label_noise": "Probable published-label noise",
}


def write_private_queue(frame: pd.DataFrame, path: Path) -> None:
    """Preserve annotations on rerun only when the stable row ID and text match."""
    fields = ("category", "anonymized_example", "review_notes")
    for field in fields:
        frame[field] = ""
    if path.exists():
        old = pd.read_csv(path, keep_default_na=False).set_index("id")
        for index, row in frame.iterrows():
            if row["id"] in old.index and old.loc[row["id"], "text"] == row["text"]:
                for field in fields:
                    frame.loc[index, field] = old.loc[row["id"], field]
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig", lineterminator="\n")


def export_review(config: Config, test: pd.DataFrame, probabilities, threshold: float, suspects: pd.DataFrame) -> dict:
    predicted = probabilities >= threshold
    mask = predicted.astype(int) != test.label.to_numpy()
    errors = test.loc[mask, ["text", "label", "source"]].copy()
    errors["probability"] = probabilities[mask]
    errors["predicted_label"] = predicted[mask].astype(int)
    errors["error_type"] = ["FP" if label == 0 else "FN" for label in errors.label]
    errors["id"] = [f"test_{index}" for index in errors.index]
    folder = config.path("reports") / "review"
    write_private_queue(errors, folder / "errors_private.csv")
    write_private_queue(suspects, folder / "suspects_private.csv")
    report = {"deployed_model": config.values["app"]["default_model"], "test_errors": len(errors), "false_positives": int((errors.label == 0).sum()), "false_negatives": int((errors.label == 1).sum()), "train_suspects": len(suspects), "manual_review_complete": False}
    (config.path("reports") / "metrics" / "error_review_status.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (config.path("reports") / "error_analysis.md").write_text(
        "# Error analysis — manual review pending\n\n"
        + f"The frozen deployed model has {len(errors)} test errors ({report['false_positives']} FP, {report['false_negatives']} FN).\n"
        + "The numbers come from metrics/error_review_status.json. No taxonomy is claimed until the team labels every error.\n\n"
        + "Follow docs/MANUAL_REVIEW.md. Private review CSVs contain dataset text and are excluded from Git.\n", encoding="utf-8")
    return report


def finalize_review(config: Config) -> dict:
    folder = config.path("reports") / "review"
    errors = pd.read_csv(folder / "errors_private.csv", keep_default_na=False)
    suspects = pd.read_csv(folder / "suspects_private.csv", keep_default_na=False)
    status_path = config.path("reports") / "metrics" / "error_review_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if len(errors) != status["test_errors"] or len(suspects) != status["train_suspects"]:
        raise ValueError("Review row count changed; restore all exported rows before finalizing")
    if errors.id.duplicated().any() or suspects.id.duplicated().any():
        raise ValueError("Duplicate review IDs; restore the original exported rows")
    for name, frame in [("test errors", errors), ("train suspects", suspects)]:
        invalid = ~frame.category.isin(CATEGORIES)
        if invalid.any():
            raise ValueError(f"T14 incomplete: {int(invalid.sum())} {name} need a valid manual category")
    rows, examples = [], []
    for category, title in CATEGORIES.items():
        group = errors[errors.category == category]
        rows.append({"category": category, "description": title, "count": len(group), "false_positives": int((group.error_type == "FP").sum()), "false_negatives": int((group.error_type == "FN").sum())})
        approved = group[group.anonymized_example.str.strip() != ""].head(2)
        needed = min(2, len(group))
        if len(approved) < needed:
            raise ValueError(f"Provide {needed} manually anonymized examples for {category}; names, numbers and links must be removed")
        examples.extend((title, row.anonymized_example) for row in approved.itertuples())
    pd.DataFrame(rows).to_csv(config.path("reports") / "tables" / "T10_error_taxonomy.csv", index=False, lineterminator="\n")
    text = ["# Human-reviewed error analysis", "", "Published labels and frozen models remain unchanged.", "",
            "| Category | Count | FP | FN |", "|---|---:|---:|---:|"]
    text.extend(f"| {row['description']} | {row['count']} | {row['false_positives']} | {row['false_negatives']} |" for row in rows)
    text.extend(["", "## Team-anonymized examples", ""])
    text.extend(f"**{title}:** {example}\n" for title, example in examples)
    text.extend(["", f"Train-only suspect labels reviewed: {len(suspects)}. Suspect flags are diagnostics, not confirmed ground-truth errors.", ""])
    (config.path("reports") / "error_analysis.md").write_text("\n".join(text), encoding="utf-8")
    status["manual_review_complete"] = True
    status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    return status
