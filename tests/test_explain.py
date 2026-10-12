"""Word occlusion preserves context, batches variants, and escapes user HTML."""

import numpy as np

from smsspam.explain import explain, highlighted_text


class FixturePredictor:
    name = "fixture"

    def __init__(self):
        self.calls = []

    def predict_proba(self, texts):
        self.calls.append(list(texts))
        return np.array([0.9 if "[URL]" in text else 0.1 for text in texts])


def test_occlusion_detects_influence_in_one_batch():
    predictor = FixturePredictor()
    effects = explain(predictor, "Claim your prize [URL] now", top_k=3)
    assert len(effects) == 3
    assert effects[0].word == "[URL]"
    assert effects[0].effect_pp > 0
    assert len(predictor.calls) == 1


def test_long_text_retains_the_unexplained_tail():
    predictor = FixturePredictor()
    explain(predictor, "one two three [URL]", max_words=2)
    assert all("three [URL]" in text for text in predictor.calls[0])


def test_user_html_is_escaped():
    text = '<script>alert(1)</script> [URL]'
    rendered = highlighted_text(text, explain(FixturePredictor(), text))
    assert "<script>" not in rendered
    assert "&lt;script&gt;" in rendered
    assert "raises spam score" in rendered


def test_empty_input_has_no_effects():
    assert explain(FixturePredictor(), "  ") == []
