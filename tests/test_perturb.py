"""Robustness transformations are deterministic and maintain intended URL semantics."""

from smsspam.evaluation.perturb import PERTURBATIONS, perturb_many
from smsspam.preprocessing import normalize


def test_all_perturbations_are_seeded():
    texts = ["FREE offer at www.example.com!", "Meet me for coffee tomorrow."]
    for kind in PERTURBATIONS:
        assert perturb_many(texts, kind) == perturb_many(texts, kind)


def test_domain_swap_preserves_normalized_urls_and_email():
    text = "Go to https://example.com or email person@example.com"
    changed = perturb_many([text], "swap_url_domain")[0]
    assert normalize(changed) == normalize(text)
    assert "person@example.com" in changed
