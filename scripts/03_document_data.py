"""Generate the measured data card and archive small dataset statistics."""

import json

import _bootstrap  # noqa: F401

from smsspam.config import load_config


def main() -> None:
    config = load_config()
    stats = json.loads((config.path("processed") / "data_stats.json").read_text(encoding="utf-8"))
    audit = json.loads((config.path("reports") / "metrics" / "data_audit.json").read_text(encoding="utf-8"))
    if not audit["passed"]:
        raise SystemExit("Finish the data audit before documenting the curated corpus")
    output = config.path("reports") / "metrics" / "data_stats.json"
    output.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Curated SMS dataset card", "", "## Purpose and provenance", "",
        "English SMS binary classification (0 = ham, 1 = spam). Seed 42. Sources are",
        "UCI SMS Spam Collection (CC BY 4.0) and the author-published Super SMS Dataset",
        "(research release without an explicit license file; academic use only; no redistribution).",
        "Attribution and source links are in [DATA_LICENSES.md](../DATA_LICENSES.md).", "",
        "## Measured build", "", "All counts below are generated from reports/metrics/data_stats.json.", "",
        "| Partition | Ham | Spam | Total |", "|---|---:|---:|---:|",
    ]
    for split in ("train", "val", "test"):
        ham = stats["final"]["split_label"]["0"][split]
        spam = stats["final"]["split_label"]["1"][split]
        lines.append(f"| {split} | {ham:,} | {spam:,} | {ham + spam:,} |")
    lines.extend([
        "", f"Total: {stats['final']['total']:,} messages. Unused held-out ham pool: {stats['final']['unused_ham_pool']:,}.",
        "", f"Curated file SHA-256: `{stats['sha256_output']}`.", "",
        "## Curation", "", "UCI copies take precedence. Apply ftfy, Unicode NFKC, whitespace/quote repair",
        "and PTB detokenization, then the following ordered filters:", "",
        "| Filter | Ham removed | Spam removed |", "|---|---:|---:|",
    ])
    lines.extend(f"| {name} | {value['removed_ham']:,} | {value['removed_spam']:,} |" for name, value in stats["steps"].items())
    lines.extend([
        "", "Balance by seeded random ham undersampling; retain all clean spam. Split 70/15/15",
        "with stratification. Apply the same reference normalizer to all model inputs.", "",
        "## Audit evidence", "", f"All {len(audit['checks'])} invariants pass. Diagnostic five-fold CV accuracies:", "",
        "| Diagnostic | Accuracy | Limit |", "|---|---:|---:|",
        f"| Formatting artefacts | {audit['diagnostics']['artifact_only']['accuracy']:.4f} | 0.56 |",
        f"| Raw masks after normalization | {audit['diagnostics']['normalized_mask_only']['accuracy']:.4f} | 0.55 |",
        f"| Source only | {audit['diagnostics']['source_only']['accuracy']:.4f} | 0.56 |", "",
        "No near-duplicate keys cross splits, and the unused ham pool shares no key with the balanced corpus.", "",
        "## Limits and privacy", "", "English-only, with regional UK/SG/IN/US skew and informal/Singlish ham.",
        "Sources cover data up to 2024; future spam can drift. Labels are kept as published",
        "and borderline promotional/transactional messages may contain residual label noise.",
        "A balanced test set inflates deployment precision; E2 must also evaluate at 13.4% spam prevalence.",
        "The corpus is already anonymized by publishers. Never de-anonymize or redistribute it.",
        "The model is an assistive estimate, not a security guarantee. Keep user input out of logs and files.", "",
    ])
    (config.root / "docs" / "DATA_CARD.md").write_text("\n".join(lines), encoding="utf-8")
    print("Generated docs/DATA_CARD.md and reports/metrics/data_stats.json")


if __name__ == "__main__":
    main()
