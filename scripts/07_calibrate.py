"""Fit calibration temperatures and risk thresholds using only val.csv."""

import json

import _bootstrap  # noqa: F401
import pandas as pd

from smsspam.calibration import calibrate_logits
from smsspam.config import load_config
from smsspam.data.build import sha256
from smsspam.models.runtime import MODEL_NAMES, RawScorer


def main() -> None:
    config = load_config()
    path = config.path("processed") / "val.csv"
    val = pd.read_csv(path)
    if not val.split.eq("val").all():
        raise SystemExit("Calibration accepts only validation rows")
    results = {}
    for name in MODEL_NAMES:
        scorer = RawScorer(name, config)
        results[name] = calibrate_logits(scorer.logits(val.text), val.label, config.values["calibration"])
        results[name]["sha256_val"] = sha256(path)
        print(f"{name}: T={results[name]['temperature']:.3f}, tau={results[name]['threshold']:.2f}, low={results[name]['threshold_low']:.2f}, val FPR={results[name]['ham_fpr']:.4f}", flush=True)
        del scorer
    output = config.path("models") / "calibration.json"
    output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    if not all(model["fpr_constraint_met"] for model in results.values()):
        raise SystemExit("A model missed the T11 FPR requirement; fallback details are in calibration.json")


if __name__ == "__main__":
    main()
