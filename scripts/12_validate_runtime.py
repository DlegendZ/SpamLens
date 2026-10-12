"""Verify frozen artifacts and predictions after inference resource changes."""

import json

import _bootstrap  # noqa: F401
import pandas as pd

from smsspam.config import load_config
from smsspam.data.build import sha256
from smsspam.evaluation.experiments import efficiency_experiment
from smsspam.evaluation.metrics import classification_metrics
from smsspam.inference import load_predictor


def main() -> None:
    config = load_config()
    metrics = config.path("reports") / "metrics"
    manifest = json.loads((metrics / "evaluation_manifest.json").read_text(encoding="utf-8"))
    for relative, expected in manifest["sha256_artifacts"].items():
        if sha256(config.root / relative.replace("\\", "/")) != expected:
            raise SystemExit(f"Frozen model artifact changed: {relative}")
    test = pd.read_csv(config.path("processed") / "test.csv")
    original = json.loads((metrics / "main_results.json").read_text(encoding="utf-8"))
    checks = {}
    for name in original:
        predictor = load_predictor(name)
        result = classification_metrics(test.label, predictor.predict_proba(test.text), predictor.settings["threshold"])
        if result["confusion_matrix"] != original[name]["confusion_matrix"]:
            raise SystemExit(f"Predictions changed for {name}; investigate before claiming unchanged evaluation")
        if abs(result["macro_f1"] - original[name]["macro_f1"]) > 1e-12:
            raise SystemExit(f"Macro-F1 changed for {name}")
        if abs(result["calibration"]["ece"] - original[name]["calibration"]["ece"]) > 1e-6:
            raise SystemExit(f"Calibration diagnostics changed for {name}")
        checks[name] = {"confusion_matrix_unchanged": True, "macro_f1_unchanged": True}
        print(f"{name}: frozen predictions unchanged", flush=True)
        del predictor
    latency = efficiency_experiment(config)
    criteria_path = metrics / "success_criteria.json"
    criteria = json.loads(criteria_path.read_text(encoding="utf-8"))
    criteria["SC6"]["value"] = latency["distilbert"]["p95_ms"]
    criteria["SC6"]["passed"] = latency["distilbert"]["p95_ms"] <= 500
    criteria_path.write_text(json.dumps(criteria, indent=2) + "\n", encoding="utf-8")
    long_ms = latency["distilbert"]["explanation_60_words_ms"]
    report = {"frozen_artifact_hashes_unchanged": True, "predictions": checks,
              "explanation_60_words_ms": long_ms, "explanation_under_2_seconds": long_ms <= 2000}
    (metrics / "runtime_validation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
