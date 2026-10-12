"""Build the curated, balanced SMS spam dataset (v1).

Reference implementation for EXECUTION_GUIDE.md, section 5 (Data Pipeline).

Inputs (raw, never modified):
  * UCI SMS Spam Collection  -> SMSSpamCollection (tab-separated: label<TAB>text), 5,574 rows
  * Super SMS Dataset        -> super_sms_dataset.csv (columns: SMSes, Labels), 67,010 rows, cp1252

Usage:
  python build_dataset.py --uci data/raw/SMSSpamCollection \
                          --super data/raw/super_sms_dataset.csv \
                          --out data/processed
Outputs:
  sms_spam_curated_v1.csv (text,label,source,split), train.csv, val.csv, test.csv,
  unused_ham_pool.csv, data_stats.json
"""
import hashlib
import json
import re
import unicodedata
from pathlib import Path

import ftfy
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42

URL_RE = re.compile(
    r"(https?://\S+|www\.\S+|\b[a-z0-9-]+\.(?:com|net|org|ly|me|co|info|io|uk|au|in|us|xyz|top|link|site)(?:/\S*)?)",
    re.I,
)
# --- Artifact 1: rows that were lower-cased AND stripped of all digits/punctuation upstream.
ARTIFACT_PUNCT = re.compile(r"[\d.,!?:;'\"/()\-@#$%&*+=<>£€]")
# --- Artifact 2: rows that were lemmatised upstream ("This be a reminder", "KYC have be suspend").
NOT_LEMMA_PREV = set(
    "to will would could should shld shd can may might must not cannot ll 'll ill i'll wil let gonna wanna "
    "gotta dont don't won't wont never also just always shall ever can't cant really still all definitely "
    "probably prob maybe must've so".split()
)
LEMMA_VERBS = (
    "deliver|attempt|suspend|lock|block|charge|process|verify|confirm|schedule|ship|update|cancel|detect|restrict|"
    "disable|approve|send|receive|hold|decline|flag|generate|expire|register|issue|reserve|credit|debit|select|"
    "activate|deactivate|terminate|cap|limit|sign|add|deduct|transfer|pend|complete|accept|reject|note|record|"
    "refund|pay|bill|grant|award|choose|notify|initiate|freeze|close|open|use|change|reset|link|start|stop|make|"
    "place|dispatch|return|mark|post|apply|qualify|submit|enable|renew|upgrade|offer|win|claim|allot|allocate|"
    "assign|book"
)
LEMMA_P1 = re.compile(
    r"\b(?:have|has|had) be\b|\b(?:this|there|which|who) be\b|\bname be\b"
    r"|\b(?:i|he|she|we|they|you|it|this|that|account|card|number|order|package|parcel|payment|message|offer|code|link)"
    r" be (?:a|an|the|your|my|our|from|now|eligible|due|valid|available|pending|requir\w*)\b"
    r"|\b(?:i|we|they) be (?:hire|hiring|recruit\w*|look\w*|offer\w*|pleased|glad|happy|writ\w*|contact\w*|inform\w*)\b",
    re.I,
)
# --- Artifact 3: Penn-Treebank tokenisation traces ("Do n't", "It 's", ``quotes'') found only in spam.
#     Fixed by DETOKENISING every row (not by dropping rows).
PTB_FIXES = [
    (re.compile(r"(\w) (n't|'s|'re|'m|'ve|'ll|'d)\b", re.I), r"\1\2"),
    (re.compile(r"``\s*|\s*''"), '"'),
    (re.compile(r"\b(wan|gon|got) na\b", re.I), r"\1na"),
]
LEMMA_P2 = re.compile(r"\b([a-z']+) be (?:" + LEMMA_VERBS + r")\b", re.I)


def clean_text(s: str) -> str:
    s = ftfy.fix_text(str(s))                     # repair multi-level mojibake (Ã‚Â£ -> £)
    s = s.replace("\ufffd", "")              # drop unrecoverable replacement chars
    s = re.sub(r"Â(?=[^A-Za-z£])", "", s)          # stray 'Â' left before apostrophes/punctuation
    s = unicodedata.normalize("NFKC", s)
    s = s.translate(str.maketrans({"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"', "\u2013": "-", "\u2014": "-", "\u00a0": " "}))
    s = re.sub(r"\s+", " ", s)                     # collapse first so PTB patterns see single spaces
    for pattern, repl in PTB_FIXES:
        s = pattern.sub(repl, s)
    return re.sub(r"\s+", " ", s).strip()


def dedup_key(s: str) -> str:
    """Near-duplicate key: templated spam that differs only in numbers/URLs/punctuation collapses."""
    s = s.lower()
    s = URL_RE.sub(" URL ", s)
    s = re.sub(r"\d+", "0", s)
    s = re.sub(r"[^a-z0 ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def is_raw_format(s: str) -> bool:
    return bool(re.search(r"[A-Z]", s) or ARTIFACT_PUNCT.search(s))


def is_lemmatised(s: str) -> bool:
    if LEMMA_P1.search(s):
        return True
    return any(m.group(1).lower() not in NOT_LEMMA_PREV for m in LEMMA_P2.finditer(s))


def is_latin_script(s: str) -> bool:
    letters = [c for c in s if c.isalpha()]
    if not letters:
        return False
    non_latin = sum(1 for c in letters if "LATIN" not in unicodedata.name(c, ""))
    return non_latin / len(letters) <= 0.10


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def curate(uci: pd.DataFrame, sup: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Apply Appendix A.2's cleaning and ordered filters, with UCI copies first."""
    stats = {"raw": {"uci": uci.label.value_counts().to_dict(), "super": sup.label.value_counts().to_dict()}}
    df = pd.concat([uci, sup], ignore_index=True)
    df["text"] = df.text.map(clean_text)
    steps = {}

    def step(name, keep_mask):
        nonlocal df
        dropped = df[~keep_mask]
        steps[name] = {
            "removed": int((~keep_mask).sum()),
            "removed_ham": int((dropped.label == 0).sum()),
            "removed_spam": int((dropped.label == 1).sum()),
        }
        df = df[keep_mask].copy()

    step("too_short", (df.text.str.len() >= 5) & (df.text.str.split().str.len() >= 2))
    step("format_stripped_artifact", df.text.map(is_raw_format))
    step("lemmatisation_artifact", ~df.text.map(is_lemmatised))
    step("non_latin_script", df.text.map(is_latin_script))
    df["key"] = df.text.map(dedup_key)
    n_labels = df.groupby("key").label.nunique()
    step("conflicting_labels", ~df.key.isin(n_labels[n_labels > 1].index))
    step("near_duplicates", ~df.key.duplicated(keep="first"))
    step("empty_key", df.key.str.len() > 0)
    stats["steps"] = steps
    stats["after_cleaning"] = df.groupby(["source", "label"]).size().unstack(fill_value=0).to_dict()
    return df, stats


def write_splits(df: pd.DataFrame, stats: dict, out: Path, seed: int = SEED) -> dict:
    """Balance and split exactly as the reference; emit portable UTF-8/LF CSVs."""
    out.mkdir(parents=True, exist_ok=True)
    spam = df[df.label == 1]
    if len(spam) < 10 or (df.label == 0).sum() < len(spam):
        raise ValueError("Need at least ten spam rows and at least as many clean ham rows")
    ham = df[df.label == 0].sample(n=len(spam), random_state=seed)
    bal = pd.concat([spam, ham]).sample(frac=1, random_state=seed)
    train, tmp = train_test_split(bal, test_size=0.30, stratify=bal.label, random_state=seed)
    val, test = train_test_split(tmp, test_size=0.50, stratify=tmp.label, random_state=seed)
    cols = ["text", "label", "source", "split"]
    parts = []
    for name, part in (("train", train), ("val", val), ("test", test)):
        part = part.assign(split=name)
        part[cols].to_csv(out / f"{name}.csv", index=False, lineterminator="\n")
        parts.append(part)
    full = pd.concat(parts)
    full[cols].to_csv(out / "sms_spam_curated_v1.csv", index=False, lineterminator="\n")
    unused = df[(df.label == 0) & (~df.index.isin(ham.index))]
    unused[["text", "label", "source"]].to_csv(out / "unused_ham_pool.csv", index=False, lineterminator="\n")
    stats["final"] = {
        "total": len(full),
        "split_label": full.groupby(["split", "label"]).size().unstack().to_dict(),
        "source_label": full.groupby(["source", "label"]).size().unstack(fill_value=0).to_dict(),
        "unused_ham_pool": len(unused),
    }
    stats["sha256_output"] = sha256(out / "sms_spam_curated_v1.csv")
    (out / "data_stats.json").write_text(json.dumps(stats, indent=2, default=str) + "\n", encoding="utf-8")
    return stats


def build_dataset(uci_path: Path, super_path: Path, out: Path, seed: int = SEED) -> dict:
    """Build the curated dataset from the verified, original raw source files."""
    uci = pd.read_csv(uci_path, sep="\t", header=None, names=["label", "text"], quoting=3)
    if len(uci) != 5574 or set(uci.label) != {"ham", "spam"}:
        raise ValueError("UCI input must have 5,574 rows and only ham/spam labels")
    uci = pd.DataFrame({"text": uci.text, "label": (uci.label == "spam").astype(int), "source": "uci_sms_spam_collection"})
    sup = pd.read_csv(super_path, encoding="cp1252", encoding_errors="replace").dropna(subset=["SMSes", "Labels"])
    if not sup.Labels.isin([0, 1]).all():
        raise ValueError("Super input contains labels outside {0, 1}")
    sup = pd.DataFrame({"text": sup.SMSes.astype(str), "label": sup.Labels.astype(int), "source": "super_sms_dataset"})
    df, stats = curate(uci, sup)
    stats["sha256_inputs"] = {"uci": sha256(uci_path), "super": sha256(super_path)}
    return write_splits(df, stats, out, seed)


def acceptance_errors(stats: dict) -> list[str]:
    """Stop if measured reference counts deviate by more than the allowed 0.5%."""
    errors = []
    expected = {"total": 19152, "unused_ham_pool": 29293}
    for name, target in expected.items():
        actual = stats["final"][name]
        if abs(actual - target) / target > 0.005:
            errors.append(f"{name}: {actual} differs from {target} by more than 0.5%")
    for name, target in {"train": 13406, "val": 2873, "test": 2873}.items():
        counts = stats["final"]["split_label"]
        actual = sum(labels[name] for labels in counts.values())
        if abs(actual - target) / target > 0.005:
            errors.append(f"{name}: {actual} differs from {target} by more than 0.5%")
    return errors

