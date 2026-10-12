"""Generate the model card and local QA status from measured report artifacts."""

import json
import re
import xml.etree.ElementTree as ET

import _bootstrap  # noqa: F401

from smsspam.config import load_config


def main() -> None:
    config = load_config()
    metrics = config.path("reports") / "metrics"
    main_results = json.loads((metrics / "main_results.json").read_text(encoding="utf-8"))
    natural = json.loads((metrics / "natural_prevalence.json").read_text(encoding="utf-8"))
    calibration = json.loads((config.path("models") / "calibration.json").read_text(encoding="utf-8"))
    test_calibration = json.loads((metrics / "test_calibration.json").read_text(encoding="utf-8"))
    environment = json.loads((metrics / "environment.json").read_text(encoding="utf-8"))
    lines = [
        "---", "language: en", "tags:", "- text-classification", "- spam-detection", "- academic-research", "---", "",
        "# SpamLens model card", "", "## Intended use", "",
        "Assistive English SMS spam/ham classification for a non-commercial academic",
        "NLP project (COMP6885001). Output is an automated estimate, not a security",
        "guarantee. Verify unexpected requests with the official sender. Never use the",
        "model to decide whether a URL, payment request, password request, or OTP is safe.", "",
        "## Models and provenance", "",
        "M1: word TF-IDF + Multinomial Naive Bayes. M2: word/character TF-IDF + Logistic",
        "Regression. M3: fine-tuned `distilbert-base-uncased` (Sanh et al., 2019,",
        "[arXiv:1910.01108](https://arxiv.org/abs/1910.01108); pretrained model Apache-2.0).",
        "The shared reference normalizer is applied at training and inference.", "",
        "Data: [UCI SMS Spam Collection](https://doi.org/10.24432/C5CC84), Almeida & Hidalgo",
        "(2011), CC BY 4.0; and Super SMS Dataset, Salman, Ikram & Kaafar (IEEE Access, 2024),",
        "[DOI 10.1109/ACCESS.2024.3364671](https://doi.org/10.1109/ACCESS.2024.3364671).",
        "Super is an author-published research release without an explicit dataset",
        "license file. This project limits use to academic, non-commercial work and",
        "does not redistribute raw or processed datasets. No broader license grant is claimed.", "",
        "Curated data: 19,152 messages, balanced 1:1, seed 42, train/validation/test",
        "70/15/15. Counts, filtering, source hashes, audit evidence, and limits are in",
        "`docs/DATA_CARD.md`, `DATA_LICENSES.md`, and `reports/metrics/data_stats.json`.", "",
        "## Training and selection", "",
        "Classical hyperparameters were selected with five-fold stratified CV on train only.",
        "M3 used dynamic padding, max length 128, AdamW (2e-5), four maximum epochs,",
        "batch 32/fp16 on RTX 3050 6 GB, and best validation macro-F1 selection.",
        "Temperature scaling and thresholds use validation only. Test data were never",
        "used for model fitting, early stopping, or threshold selection.", "",
        f"Recorded Python: {environment['python']}; scikit-learn: {environment['packages']['scikit-learn']};",
        f"Transformers: {environment['packages']['transformers']}; PyTorch: {environment['packages']['torch']}.", "",
        "## Measured held-out results", "",
        "Values are generated from reports/metrics/main_results.json and natural_prevalence.json.", "",
        "| Model | Macro-F1 (95% CI) | Spam recall | Ham FPR | E2 spam precision | Test ECE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, result in main_results.items():
        ci = result["confidence_intervals"]["macro_f1"]
        ece = test_calibration[name]["after"]["ece"]
        lines.append(f"| {name} | {result['macro_f1']:.4f} [{ci['lower']:.4f}, {ci['upper']:.4f}] | {result['spam_recall']:.4f} | {result['ham_fpr']:.4f} | {natural[name]['spam_precision']:.4f} | {ece:.4f} |")
    lines.extend(["", "E2 uses measured spam prevalence approximately 13.4%, including a held-out ham pool.",
                  "Intervals use 1,000 seeded paired-row bootstrap resamples. Exact McNemar comparisons",
                  "are in reports/metrics/mcnemar.json. Success criteria SC1–SC6 pass; the public-app",
                  "part of SC7 is pending publication. See docs/MEASURED_FINDINGS.md for robustness limits.", "",
                  "## Validation-selected calibration", "",
                  "| Model | Temperature | Spam threshold | Suspicious-band lower threshold |",
                  "|---|---:|---:|---:|"])
    for name, values in calibration.items():
        lines.append(f"| {name} | {values['temperature']:.4f} | {values['threshold']:.2f} | {values['threshold_low']:.2f} |")
    lines.extend(["", "The app uses 'calibrated spam probability' only when that model's test ECE ≤0.03.", "",
                  "## Limitations, privacy, and ethics", "",
                  "English-only; regional/dialect coverage is uneven; sources end in 2024; future spam",
                  "can drift. Promotional/transactional boundaries differ by source and some published",
                  "labels are doubtful. Informal or mixed-language ham can be falsely flagged.",
                  "M3 ham FPR rises under leetspeak and punctuation stripping (T9); unperturbed success",
                  "criteria do not imply adversarial robustness. Balanced-test precision is optimistic",
                  "relative to real-world prevalence; use E2 for that comparison.", "",
                  "The application keeps input, upload results, and explanations in session memory only,",
                  "never on disk, in logs, or in analytics. Model resource caches contain only models.",
                  "The explanation uses leave-one-word-out occlusion (first 60 words, top 8 effects),",
                  "which ignores word interactions and is not a causal explanation.", "",
                  "## Availability", "",
                  "Classical artifacts and calibration are committed locally. M3 weights remain in",
                  "gitignored models/distilbert/. No Hugging Face upload or Git push has been performed.",
                  "A checkout without local M3 automatically uses M2 and displays a fallback notice.", ""])
    (config.path("models") / "MODEL_CARD.md").write_text("\n".join(lines), encoding="utf-8")
    labels = {"tfidf_nb": "M1: TF-IDF + Naive Bayes", "tfidf_lr": "M2: Word + char TF-IDF + Logistic Regression", "distilbert": "M3: DistilBERT"}
    table = ["| Model | Test macro-F1 | Ham FPR | Spam precision at 13.4% prevalence |", "|---|---:|---:|---:|"]
    table.extend(f"| {labels[name]} | {result['macro_f1']:.4f} | {result['ham_fpr']:.2%} | {natural[name]['spam_precision']:.2%} |" for name, result in main_results.items())
    readme_path = config.root / "README.md"
    readme = readme_path.read_text(encoding="utf-8")
    block = "<!-- BEGIN GENERATED RESULTS -->\n" + "\n".join(table) + "\n<!-- END GENERATED RESULTS -->"
    readme = re.sub(r"<!-- BEGIN GENERATED RESULTS -->.*?<!-- END GENERATED RESULTS -->", lambda _: block, readme, flags=re.S)
    readme_path.write_text(readme, encoding="utf-8")
    xml = metrics / "test_results.xml"
    if xml.exists():
        suite = ET.parse(xml).getroot().find("testsuite")
        summary = {name: int(suite.get(name, "0")) for name in ("tests", "failures", "errors", "skipped")}
        summary["passed"] = summary["tests"] - summary["failures"] - summary["errors"] - summary["skipped"]
        (metrics / "test_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        criteria_path = metrics / "success_criteria.json"
        criteria = json.loads(criteria_path.read_text(encoding="utf-8"))
        criteria["SC7"] = {"passed": None, "automated_tests_passed": summary["failures"] == summary["errors"] == 0,
                           "local_app_verified": True, "public_app_verified": False,
                           "reason": "Local Streamlit and tests verified; public deployment awaits authorization at T17/T18"}
        criteria_path.write_text(json.dumps(criteria, indent=2) + "\n", encoding="utf-8")
    print("Generated measured MODEL_CARD.md and available local QA evidence")


if __name__ == "__main__":
    main()
