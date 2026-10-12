"""Real-corpus audit; skip only when local processed data are unavailable."""

import pytest

from smsspam.config import load_config
from smsspam.data.audit import audit_dataset


def test_all_eight_invariants_on_the_real_corpus():
    processed = load_config().path("processed")
    if not (processed / "sms_spam_curated_v1.csv").exists():
        pytest.skip("Rebuild local processed data before running the real-corpus audit")
    report = audit_dataset(processed)
    assert len(report["checks"]) == 8
    assert report["passed"], report
