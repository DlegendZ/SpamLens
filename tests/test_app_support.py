"""Input limits, language heuristics, and CSV exports protect UI boundaries."""

import io

import pandas as pd
import pytest

from smsspam.app_support import (
    csv_bytes,
    guard_text,
    likely_non_english,
    parse_batch,
    predict_batch,
)
from smsspam.inference import load_predictor


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (b"", "could not be read"),
        (b"body\nhello\n", "column named text"),
        (b"text,text\none,two\n", "exactly one"),
        (b"text\n", "between 1"),
        (b"text\n\"   \"\n", "non-empty"),
        (b"text\n\xff\n", "UTF-8"),
        (b"text\n" + b"Hi\n" * 1001, "1,000"),
        (b"text\n" + b"a" * 2_000_000, "larger than 2 MB"),
        (b'text\n"unclosed', "could not be read"),
    ],
    ids=["empty", "missing_column", "duplicate_column", "no_rows", "blank_text", "encoding", "too_many_rows", "oversize_file", "malformed"],
)
def test_batch_rejects_invalid_uploads(payload, message):
    with pytest.raises(ValueError, match=message):
        parse_batch(payload)


def test_batch_sample_round_trip_matches_single_predictions():
    frame = parse_batch(b"\xef\xbb\xbftext\nHello friend\nWin a prize now\n")
    predictor = load_predictor("tfidf_lr")
    progress = []
    result = predict_batch(frame, predictor, progress=progress.append)
    assert len(result) == len(frame)
    assert progress[-1] == 1
    for text, probability in zip(frame.text, result.spam_probability, strict=True):
        assert probability == pytest.approx(predictor.predict_proba([text])[0])
    exported = pd.read_csv(io.BytesIO(csv_bytes(result)))
    assert exported.label.tolist() == result.label.tolist()


def test_oversized_messages_are_truncated_before_prediction():
    text, truncated = guard_text("x" * 1001)
    assert len(text) == 1000
    assert truncated
    assert guard_text(" \t ") == ("", False)


def test_language_warning_handles_indonesian_and_non_latin():
    vocabulary = {"hello", "please", "meet", "tomorrow", "morning"}
    assert likely_non_english("Saya sedang pergi bersama teman untuk makan", vocabulary)
    assert likely_non_english("مرحبا صديقي كيف حالك اليوم صباحا", vocabulary)
    assert not likely_non_english("Hello please meet tomorrow morning", vocabulary)
    assert not likely_non_english("Hi 🎉", vocabulary)


def test_csv_export_neutralizes_formulas_without_changing_input():
    frame = pd.DataFrame({"text": ["=1+1", " +SUM(1,2)", "ordinary text"]})
    exported = pd.read_csv(io.BytesIO(csv_bytes(frame)))
    assert exported.text.tolist() == ["'=1+1", "' +SUM(1,2)", "ordinary text"]
    assert frame.text.iloc[0] == "=1+1"
