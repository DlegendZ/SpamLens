"""Inference acceptance contract on the committed classical artifacts."""

import numpy as np
import pytest

from smsspam.config import load_config
from smsspam.inference import load_predictor

TEXTS = ["", "   ", "🎉", "مرحبا 你好", "Hello friend, lunch at noon?", "Win £1000 now! Claim at https://example.com", "FREE " * 200]


@pytest.mark.parametrize("name", ["tfidf_nb", "tfidf_lr"])
def test_classical_predictor_contract(name):
    predictor = load_predictor(name)
    probabilities = predictor.predict_proba(TEXTS)
    assert probabilities.shape == (len(TEXTS),)
    assert np.all((probabilities >= 0) & (probabilities <= 1))
    np.testing.assert_array_equal(probabilities, predictor.predict_proba(TEXTS))
    np.testing.assert_allclose(probabilities, [predictor.predict_proba([text])[0] for text in TEXTS], atol=1e-12)
    for probability, result in zip(probabilities, predictor.predict(TEXTS), strict=True):
        assert result.label == ("spam" if probability >= predictor.settings["threshold"] else "ham")
        if result.risk_band == "suspicious":
            assert predictor.settings["threshold_low"] <= probability < predictor.settings["threshold"]
        assert result.model_name == name
        assert result.latency_ms >= 0
    assert predictor.predict([]) == []
    assert predictor.predict_proba([]).shape == (0,)


@pytest.mark.slow
def test_distilbert_predictor_contract():
    if not (load_config().path("models") / "distilbert" / "config.json").exists():
        pytest.skip("Local transformer weights absent")
    predictor = load_predictor("distilbert")
    probabilities = predictor.predict_proba(TEXTS)
    assert np.all(np.isfinite(probabilities))
    np.testing.assert_allclose(probabilities, [predictor.predict_proba([text])[0] for text in TEXTS], atol=1e-6)
    assert len(predictor.predict(TEXTS)) == len(TEXTS)
