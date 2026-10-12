"""Record CPU latency over 200 held-out texts after warm-up; no parameter selection."""

import json
from time import perf_counter

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

from smsspam.config import load_config
from smsspam.explain import explain
from smsspam.inference import load_predictor
from smsspam.models.runtime import MODEL_NAMES


def main() -> None:
    config = load_config()
    texts = pd.read_csv(config.path("processed") / "test.csv").text.sample(n=200, random_state=config.seed).tolist()
    results = {}
    for name in MODEL_NAMES:
        predictor = load_predictor(name)
        for text in texts[:5]:
            predictor.predict([text])
        times = []
        for text in texts:
            started = perf_counter()
            predictor.predict([text])
            times.append((perf_counter() - started) * 1000)
        started = perf_counter()
        explain(predictor, texts[0])
        explanation_ms = (perf_counter() - started) * 1000
        results[name] = {"n": len(texts), "device": "cpu", "mean_ms": float(np.mean(times)), "p95_ms": float(np.percentile(times, 95)), "explanation_ms": explanation_ms}
        print(f"{name}: CPU p95 {results[name]['p95_ms']:.1f} ms; explanation {explanation_ms:.1f} ms", flush=True)
        del predictor
    (config.path("reports") / "metrics" / "cpu_latency.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
