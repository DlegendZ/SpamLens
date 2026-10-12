"""Seeded, label-preserving stress tests fixed before test-set evaluation."""

import re
import string

import numpy as np

from smsspam.preprocessing import _MASK_URL, _URL

PERTURBATIONS = ("lowercase", "strip_punctuation", "leetspeak", "character_spacing", "swap_url_domain", "append_benign")
LEET = {"a": "4", "e": "3", "i": "1", "o": "0", "s": "5"}


def perturb(text: str, kind: str, rng: np.random.Generator) -> str:
    if kind == "lowercase":
        return text.lower()
    if kind == "strip_punctuation":
        return text.translate(str.maketrans("", "", string.punctuation))
    if kind == "leetspeak":
        return "".join(LEET[character.lower()] if character.lower() in LEET and rng.random() < 0.3 else character for character in text)
    if kind == "character_spacing":
        words = list(re.finditer(r"\b[A-Za-z]{3,}\b", text))
        if not words:
            return text
        word = words[int(rng.integers(len(words)))]
        return text[:word.start()] + " ".join(word.group()) + text[word.end():]
    if kind == "swap_url_domain":
        def replace(match):
            return match.group() if match.start() and text[match.start() - 1] == "@" else "https://replacement.example.com"
        return _MASK_URL.sub("https://replacement.example.com", _URL.sub(replace, text))
    if kind == "append_benign":
        return text + " See you soon."
    raise ValueError(f"Unknown perturbation: {kind}")


def perturb_many(texts, kind: str, seed: int = 42) -> list[str]:
    rng = np.random.default_rng(seed)
    return [perturb(text, kind, rng) for text in texts]
