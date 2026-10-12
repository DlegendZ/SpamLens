"""Model-agnostic, batched leave-one-word-out probability explanations."""

import html
import re
from dataclasses import dataclass

import numpy as np

from smsspam.inference import Predictor


@dataclass(frozen=True)
class WordEffect:
    word: str
    word_index: int
    start: int
    end: int
    effect_pp: float


def explain(predictor: Predictor, text: str, top_k: int = 8, max_words: int = 60) -> list[WordEffect]:
    """Occlude the first max_words words, preserving the full message's remainder."""
    matches = list(re.finditer(r"\S+", text))[:max_words]
    if not matches or top_k <= 0:
        return []
    variants = [text[:match.start()] + text[match.end():] for match in matches]
    probabilities = predictor.predict_proba([text, *variants])
    changes = (probabilities[0] - probabilities[1:]) * 100
    selected = np.argsort(-np.abs(changes), kind="stable")[:top_k]
    return [WordEffect(matches[index].group(), int(index), matches[index].start(), matches[index].end(), float(changes[index])) for index in selected]


def highlighted_text(text: str, effects: list[WordEffect]) -> str:
    """Escape all user content and label direction independently of color."""
    parts, position = [], 0
    for effect in sorted(effects, key=lambda item: item.start):
        parts.append(html.escape(text[position:effect.start]))
        direction = "raises spam score" if effect.effect_pp >= 0 else "lowers spam score"
        parts.append(f'<mark title="{direction}">{html.escape(text[effect.start:effect.end])} ({direction})</mark>')
        position = effect.end
    parts.append(html.escape(text[position:]))
    return "".join(parts)
