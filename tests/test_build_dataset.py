"""Reference curation filters, source precedence, and deterministic splits."""

import pandas as pd
import pytest

from smsspam.data.build import clean_text, curate, dedup_key, is_lemmatised, write_splits


def fixtures():
    names = "Amber Birch Cedar Dahlia Elm Fern Grove Hazel Iris Juniper Kiwi Laurel Maple Nectar Olive Pine Quartz Rose Sage Thyme Umber Violet Willow Xenon Yarrow Zinnia".split()
    uci = pd.DataFrame([
        {"text": f"Lunch with {name} tomorrow.", "label": 0, "source": "uci_sms_spam_collection"}
        for name in names
    ])
    sup = pd.DataFrame([
        {"text": f"Urgent claim voucher {name} NOW!", "label": 1, "source": "super_sms_dataset"}
        for name in names[:20]
    ])
    extras = [
        ("Hi", 0), ("lowercase only words", 1), ("This be a reminder", 1),
        ("你好朋友 Hello", 0), ("Account notice status.", 0), ("Account notice status!", 1),
        ("Lunch with Amber tomorrow.", 0),
    ]
    extra_frame = pd.DataFrame([{"text": text, "label": label, "source": "super_sms_dataset"} for text, label in extras])
    return uci, pd.concat([sup, extra_frame], ignore_index=True)


def test_ordered_filters_remove_the_intended_rows():
    clean, stats = curate(*fixtures())
    expected = {
        "too_short": 1, "format_stripped_artifact": 1, "lemmatisation_artifact": 1,
        "non_latin_script": 1, "conflicting_labels": 2, "near_duplicates": 1, "empty_key": 0,
    }
    assert {name: entry["removed"] for name, entry in stats["steps"].items()} == expected
    assert len(clean) == 46
    assert clean.loc[clean.text == "Lunch with Amber tomorrow.", "source"].item() == "uci_sms_spam_collection"


def test_build_is_balanced_disjoint_and_deterministic(tmp_path):
    hashes = []
    for attempt in range(2):
        clean, stats = curate(*fixtures())
        output = tmp_path / str(attempt)
        result = write_splits(clean, stats, output)
        hashes.append(result["sha256_output"])
        full = pd.read_csv(output / "sms_spam_curated_v1.csv")
        assert full.label.value_counts().to_dict() == {0: 20, 1: 20}
        assert not full.text.map(dedup_key).duplicated().any()
        split_counts = full.groupby("split").label.agg(["sum", "count"])
        assert (abs(2 * split_counts["sum"] - split_counts["count"]) <= 1).all()
        pool = pd.read_csv(output / "unused_ham_pool.csv")
        assert set(pool.text.map(dedup_key)).isdisjoint(full.text.map(dedup_key))
    assert hashes[0] == hashes[1]


@pytest.mark.parametrize(
    ("text", "expected"),
    [("Do n't click. It 's mine.", "Don't click. It's mine."), ("``Hello'' friend", '"Hello" friend'), ("Win Ã‚Â£100 today", "Win £100 today")],
)
def test_clean_text_repairs_publisher_artifacts(text, expected):
    assert clean_text(text) == expected


def test_legitimate_modal_is_not_lemmatization_artifact():
    assert not is_lemmatised("Please let me be happy")
    assert is_lemmatised("Your account have be suspend")


def test_dedup_key_collapses_numbers_and_urls():
    assert dedup_key("CLAIM 123 at www.example.com!") == dedup_key("claim 456 at www.other.com.")
