"""Shared text normaliser (reference implementation for EXECUTION_GUIDE.md section 6).

Applied identically to EVERY model, at training AND inference time.
Purpose: map dataset-specific anonymisation masks ("XXXX", "<#>", "<DECIMAL>") and real
values (digits, phone numbers, URLs) onto the SAME placeholders, so that a model cannot
learn "contains XXXX => spam" (an annotation artefact) and real user input looks like
training input.
"""
import re
import unicodedata

PLACEHOLDERS = ("[URL]", "[EMAIL]", "[PHONE]", "[MONEY]", "[NUM]")

_URL = re.compile(
    r"(?:https?://|www\.)\S+"
    r"|\b[a-z0-9-]+(?:\.[a-z0-9-]+)*\.(?:com|net|org|ly|me|co|info|io|uk|au|in|us|xyz|top|link|site|online|club|app|gl|biz)\b(?:/\S*)?",
    re.I,
)
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_MASK_TAGS = re.compile(r"<(?:#|DECIMAL|TIME)>", re.I)
_MASK_URL = re.compile(r"<URL>", re.I)
_MASK_EMAIL = re.compile(r"<EMAIL>", re.I)
# digit-or-mask character: real digits OR the uppercase X used by dataset anonymisers
_D = r"[0-9X]"
_PHONE = re.compile(rf"(?<![A-Za-z0-9])\+?{_D}(?:[\s\-()]?{_D}){{6,}}(?![A-Za-z0-9])")
_MONEY = re.compile(
    rf"(?:[£$€₹]|\b(?:rs|inr|usd|gbp|eur|php|ngn|ksh|aud|sgd)\.?)\s?{_D}[{_D[1:-1]},.]*(?:\s?(?:k|m|lakh|lac|lacs|crore|cr)\b)?",
    re.I,
)
_NUM = re.compile(rf"(?<![A-Za-z0-9]){_D}*\d{_D}*(?:[.,:/]{_D}+)*(?![A-Za-z0-9])|\bX{{2,}}\b|(?<![A-Za-z'\-])X(?![A-Za-z'\-])")


def normalize(text: str) -> str:
    """Return the normalised form of one SMS. Pure function, idempotent."""
    if text is None:
        return ""
    s = unicodedata.normalize("NFKC", str(text))
    s = _MASK_URL.sub(" [URL] ", s)
    s = _MASK_EMAIL.sub(" [EMAIL] ", s)
    s = _MASK_TAGS.sub(" [NUM] ", s)
    s = _EMAIL.sub(" [EMAIL] ", s)
    s = _URL.sub(" [URL] ", s)
    s = _MONEY.sub(" [MONEY] ", s)
    s = _PHONE.sub(" [PHONE] ", s)
    s = _NUM.sub(" [NUM] ", s)
    s = re.sub(r"(?:\[NUM\]\s*){2,}", "[NUM] ", s)   # collapse runs
    return re.sub(r"\s+", " ", s).strip()
