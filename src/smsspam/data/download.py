"""Download original SMS sources with retries and pinned integrity checks."""

import hashlib
import io
import json
import time
import urllib.error
import urllib.request
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from smsspam.config import Config


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def fetch(url: str, attempts: int = 3, timeout: int = 60) -> bytes:
    """Retry transient network failures; integrity failures are never ignored."""
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "SpamLens/0.1"})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except (urllib.error.URLError, TimeoutError, OSError):
            if attempt == attempts - 1:
                raise
            time.sleep(2**attempt)
    raise RuntimeError("No download attempts configured")


def verify_hash(payload: bytes, expected: str, name: str) -> None:
    actual = sha256_bytes(payload)
    if actual != expected:
        raise ValueError(f"{name}: SHA-256 {actual} does not match pinned {expected}")


def verify_uci(payload: bytes) -> dict[str, int]:
    frame = pd.read_csv(io.BytesIO(payload), sep="\t", header=None, names=["label", "text"], quoting=3)
    counts = {str(label): int(count) for label, count in frame.label.value_counts().items()}
    if len(frame) != 5574 or counts != {"ham": 4827, "spam": 747} or frame.text.isna().any():
        raise ValueError(f"UCI row/class counts differ from the pinned specification: {counts}")
    return counts


def download_uci(settings: dict) -> tuple[bytes, dict]:
    """Prefer official UCI bytes; use the hash-pinned mirror on failure."""
    try:
        archive = fetch(settings["url"])
        with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
            payload = bundle.read("SMSSpamCollection")
        counts = verify_uci(payload)
        return payload, {
            "url": settings["url"], "fallback_used": False,
            "archive_sha256": sha256_bytes(archive), "class_counts": counts,
        }
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as error:
        payload = fetch(settings["fallback_url"])
        verify_hash(payload, settings["fallback_sha256"], "UCI mirror")
        counts = verify_uci(payload)
        return payload, {
            "url": settings["fallback_url"], "fallback_used": True,
            "primary_error": str(error), "class_counts": counts,
        }


def write_atomic(path: Path, payload: bytes) -> None:
    """Replace a local raw artifact only after all validation has passed."""
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_bytes(payload)
    temporary.replace(path)


def download_sources(config: Config) -> dict:
    """Fetch both sources, validate them, and record the download provenance."""
    raw = config.path("raw")
    raw.mkdir(parents=True, exist_ok=True)
    sources = config.values["sources"]
    uci, uci_metadata = download_uci(sources["uci"])
    sup = fetch(sources["super"]["url"])
    verify_hash(sup, sources["super"]["sha256"], "Super SMS Dataset")
    super_frame = pd.read_csv(io.BytesIO(sup), encoding="cp1252", encoding_errors="replace")
    if not {"SMSes", "Labels"}.issubset(super_frame.columns) or len(super_frame) != 67010:
        raise ValueError("Super SMS Dataset schema or row count does not match the pinned input")
    files = {
        "uci": ("SMSSpamCollection", uci, uci_metadata),
        "super": ("super_sms_dataset.csv", sup, {"url": sources["super"]["url"], "encoding": "cp1252"}),
    }
    manifest = {"downloaded_at": datetime.now(UTC).isoformat(), "sources": {}}
    for name, (filename, payload, metadata) in files.items():
        write_atomic(raw / filename, payload)
        manifest["sources"][name] = {
            **metadata, "filename": filename, "sha256": sha256_bytes(payload),
            "size_bytes": len(payload), "rows": 5574 if name == "uci" else len(super_frame),
        }
    (raw / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
