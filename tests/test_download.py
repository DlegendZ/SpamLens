"""Pinned integrity and recovery behavior without external network access."""

import io
import urllib.error
import zipfile

import pytest

from smsspam.data import download


def test_hash_mismatch_is_rejected():
    with pytest.raises(ValueError, match="does not match pinned"):
        download.verify_hash(b"corrupt", "0" * 64, "fixture")


def test_uci_malformed_data_is_rejected():
    with pytest.raises(ValueError, match="row/class counts"):
        download.verify_uci(b"spam\tWin a prize!\n")


def test_fetch_retries_transient_error(monkeypatch):
    calls = []

    def urlopen(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise urllib.error.URLError("temporary outage")
        return io.BytesIO(b"ok")

    monkeypatch.setattr(download.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(download.time, "sleep", lambda _: None)
    assert download.fetch("https://example.org/data") == b"ok"
    assert len(calls) == 2


def test_uci_falls_back_with_hash_check(monkeypatch):
    payload = b"test fixture"
    settings = {"url": "primary", "fallback_url": "mirror", "fallback_sha256": download.sha256_bytes(payload)}

    def fetch(url):
        if url == "primary":
            raise urllib.error.URLError("offline")
        return payload

    monkeypatch.setattr(download, "fetch", fetch)
    monkeypatch.setattr(download, "verify_uci", lambda _: {"ham": 4827, "spam": 747})
    result, metadata = download.download_uci(settings)
    assert result == payload
    assert metadata["fallback_used"] is True
    assert metadata["url"] == "mirror"


def test_uci_primary_records_archive_hash(monkeypatch):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("SMSSpamCollection", b"test fixture")
    monkeypatch.setattr(download, "fetch", lambda _: buffer.getvalue())
    monkeypatch.setattr(download, "verify_uci", lambda _: {"ham": 4827, "spam": 747})
    _, metadata = download.download_uci({"url": "primary"})
    assert metadata["archive_sha256"] == download.sha256_bytes(buffer.getvalue())
    assert metadata["fallback_used"] is False
