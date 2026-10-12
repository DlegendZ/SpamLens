"""Real-corpus audit; skip only when local processed data are unavailable."""

import pandas as pd
import pytest
from test_build_dataset import fixtures

from smsspam.config import load_config
from smsspam.data.audit import audit_dataset
from smsspam.data.build import curate, write_splits


def test_all_eight_invariants_on_the_real_corpus():
    processed = load_config().path("processed")
    if not (processed / "sms_spam_curated_v1.csv").exists():
        pytest.skip("Rebuild local processed data before running the real-corpus audit")
    report = audit_dataset(processed)
    assert len(report["checks"]) == 8
    assert report["passed"], report


def test_split_label_tampering_is_detected(tmp_path):
    clean, stats = curate(*fixtures())
    clean["source"] = "uci_sms_spam_collection"
    write_splits(clean, stats, tmp_path)
    train_path = tmp_path / "train.csv"
    train = pd.read_csv(train_path)
    ham = train.index[train.label == 0][0]
    spam = train.index[train.label == 1][0]
    train.loc[ham, "label"], train.loc[spam, "label"] = 1, 0
    train.to_csv(train_path, index=False)
    report = audit_dataset(tmp_path)
    assert report["checks"]["balanced_classes"]
    assert not report["checks"]["unique_disjoint_keys"]
    assert not report["passed"]
