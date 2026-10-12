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
    test = pd.read_csv(config.path("processed") / "test.csv")
    texts = test.text.sample(n=200, random_state=config.seed).tolist()
    long_text = test.loc[test.text.str.len().idxmax(), "text"][:config.values["app"]["max_chars"]]
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
        long_times = []
        explain(predictor, long_text)
        for _ in range(3):
            started = perf_counter()
            explain(predictor, long_text)
            long_times.append((perf_counter() - started) * 1000)
        results[name] = {
            "n": len(texts), "device": "cpu", "mean_ms": float(np.mean(times)),
            "p95_ms": float(np.percentile(times, 95)), "explanation_ms": explanation_ms,
            "explanation_60_words_ms": float(np.median(long_times)),
            "explanation_60_words_runs_ms": long_times,
            "cpu_threads": predictor.scorer.torch.get_num_threads() if name == "distilbert" else None,
        }
        print(f"{name}: CPU p95 {results[name]['p95_ms']:.1f} ms; explanation {explanation_ms:.1f} ms", flush=True)
        del predictor
    (config.path("reports") / "metrics" / "cpu_latency.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
