"""Statistical helpers match independently known binary cases."""

import numpy as np
import pytest

from smsspam.evaluation.metrics import bootstrap_intervals, classification_metrics, mcnemar


def test_bootstrap_contains_point_estimates_and_is_seeded():
    labels = np.tile([0, 1], 100)
    probabilities = labels * 0.8 + 0.1
    probabilities[:12] = 1 - probabilities[:12]
    point = classification_metrics(labels, probabilities, 0.5)
    ci = bootstrap_intervals(labels, probabilities >= 0.5)
    for metric in ("macro_f1", "spam_recall", "ham_fpr"):
        assert ci[metric]["lower"] <= point[metric] <= ci[metric]["upper"]
    assert ci == bootstrap_intervals(labels, probabilities >= 0.5)


def test_exact_mcnemar_known_discordant_pairs():
    labels = np.zeros(10)
    result = mcnemar(labels, labels, np.ones(10))
    assert result["p_value"] == pytest.approx(2 / 2**10)
    assert result["a_correct_b_wrong"] == 10


def test_identical_models_have_p_value_one():
    assert mcnemar([0, 1], [0, 1], [0, 1])["p_value"] == 1
