"""Input guards and in-memory CSV handling shared by the UI and its tests."""

import csv
import io
import re
from collections.abc import Callable
from dataclasses import asdict

import pandas as pd

from smsspam.inference import Predictor
from smsspam.preprocessing import normalize

DISCLAIMER = (
    "This is an automated estimate and may be wrong. Do not click links or share "
    "codes/passwords in unexpected messages; verify with the official sender."
)
LANGUAGE_WARNING = (
    "This message may not be in English. The model was trained on English SMS, "
    "so this result is less reliable."
)
MODEL_LABELS = {
    "distilbert": "DistilBERT",
    "tfidf_lr": "TF-IDF + Logistic Regression",
    "tfidf_nb": "TF-IDF + Naive Bayes",
}
BAND_LABELS = {
    "likely_spam": "Likely spam",
    "suspicious": "Suspicious — verify before acting",
    "likely_ham": "Likely legitimate",
}
MAX_CSV_BYTES = 2_000_000


def guard_text(text: str, max_chars: int = 1000) -> tuple[str, bool]:
    """Trim blank input and enforce the configured limit before inference."""
    text = text.strip()
    return text[:max_chars], len(text) > max_chars


def likely_non_english(text: str, vocabulary: set[str]) -> bool:
    tokens = re.findall(r"\b[^\W\d_]{3,}\b", normalize(text).lower())
    return len(tokens) >= 5 and sum(token in vocabulary for token in tokens) / len(tokens) < 0.4


def parse_batch(payload: bytes, max_rows: int = 1000) -> pd.DataFrame:
    """Validate UTF-8 CSV before passing any content to a model."""
    if len(payload) > MAX_CSV_BYTES:
        raise ValueError("The CSV is larger than 2 MB. Upload a smaller file.")
    try:
        decoded = payload.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError("Save the CSV as UTF-8 before uploading it.") from error
    try:
        header = next(csv.reader(io.StringIO(decoded)))
        if header.count("text") != 1:
            raise ValueError("The CSV must contain exactly one column named text.")
        frame = pd.read_csv(io.StringIO(decoded), dtype=str, keep_default_na=False)
    except (StopIteration, pd.errors.EmptyDataError, pd.errors.ParserError, csv.Error) as error:
        raise ValueError("The file could not be read as a CSV with a text column.") from error
    if not 1 <= len(frame) <= max_rows:
        raise ValueError(f"Upload between 1 and {max_rows:,} message rows.")
    if frame.text.str.strip().eq("").any():
        raise ValueError("Every row needs a non-empty message in the text column.")
    return frame[["text"]].copy()


def predict_batch(frame: pd.DataFrame, predictor: Predictor, max_chars: int = 1000,
                  progress: Callable[[float], None] | None = None) -> pd.DataFrame:
    texts = [guard_text(text, max_chars)[0] for text in frame.text]
    predictions = []
    for start in range(0, len(texts), 32):
        predictions.extend(asdict(result) for result in predictor.predict(texts[start:start + 32]))
        if progress:
            progress(min((start + 32) / len(texts), 1.0))
    result = pd.DataFrame(predictions)
    result.insert(0, "text", texts)
    return result


def csv_bytes(frame: pd.DataFrame) -> bytes:
    """Keep exports in memory and neutralize spreadsheet formula prefixes."""
    output = frame.copy()
    if "text" in output:
        output["text"] = output.text.map(lambda text: "'" + text if text.lstrip().startswith(("=", "+", "-", "@")) else text)
    return output.to_csv(index=False, lineterminator="\n").encode("utf-8-sig")
