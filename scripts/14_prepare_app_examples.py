"""Export selected, anonymized held-out examples without modifying source SMS."""

import json

import _bootstrap  # noqa: F401
import pandas as pd

from smsspam.config import load_config
from smsspam.data.build import sha256
from smsspam.preprocessing import normalize

# Manually inspected for names and personal details; normalized links/codes remain inert.
SINGLE_IDS = [("Spam: prize claim", 2869), ("Spam: recipe link", 2215), ("Spam: medicine offer", 1062),
              ("Ham: checking in", 1238), ("Ham: project plans", 2061), ("Ham: funny video", 1548)]
BATCH_IDS = [2869, 2215, 1062, 950, 1238, 2061, 1548, 2285, 2757, 632]


def anonymized(text: str) -> str:
    """Redact reviewed details outside the fixed reference normalizer's scope."""
    text = normalize(text)
    replacements = {
        "[NUM] pounds": "[MONEY]", "150p/msg": "[MONEY]/msg", "TCR/W1": "[LOCATION]",
        "PHARMA15": "[CODE]", "klr.pw/ e1UQu/dVDNLk": "[URL]",
    }
    for original, replacement in replacements.items():
        text = text.replace(original, replacement)
    return text


def main() -> None:
    config = load_config()
    path = config.path("processed") / "test.csv"
    test = pd.read_csv(path)
    if not test.split.eq("test").all():
        raise SystemExit("App examples must come exclusively from the held-out test split")
    assets = config.root / "app" / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    examples = [{"title": title, "text": anonymized(test.loc[index, "text"]), "label": int(test.loc[index, "label"]), "test_row": index} for title, index in SINGLE_IDS]
    batch = pd.DataFrame({"text": [anonymized(test.loc[index, "text"]) for index in BATCH_IDS]})
    batch.to_csv(assets / "sample_messages.csv", index=False, lineterminator="\n")
    metadata = {"source": "anonymized held-out test examples", "sha256_test": sha256(path), "single_examples": examples,
                "batch_test_rows": BATCH_IDS, "anonymization": "Manually selected name-free messages; reference normalizer plus reviewed redactions for remaining codes, money, postcode, and a .pw link. Redactions apply only to app examples; no source text, labels, or preprocessing rules changed."}
    (assets / "examples.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print("Prepared 6 anonymized single examples and a 10-row sample CSV")


if __name__ == "__main__":
    main()
