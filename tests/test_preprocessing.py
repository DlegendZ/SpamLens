"""Acceptance cases for the shared, reference normalizer (guide §6)."""

import pytest

from smsspam.preprocessing import normalize


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (None, ""),
        ("", ""),
        ("  \n\t  ", ""),
        ("Visit https://example.com/prize now", "Visit [URL] now"),
        ("Visit www.example.org", "Visit [URL]"),
        ("Visit claim.example.co.uk/pay", "Visit [URL]"),
        ("Email person@example.com now", "Email [EMAIL] now"),
        ("Call +44 7700 900123", "Call [PHONE]"),
        ("Call 07XXXXXXXXX", "Call [PHONE]"),
        ("Win £1,000 today", "Win [MONEY] today"),
        ("Win USD 500 today", "Win [MONEY] today"),
        ("Win £XXXX today", "Win [MONEY] today"),
        ("Meet at 12:30", "Meet at [NUM]"),
        ("Code 12XX expires", "Code [NUM] expires"),
        ("XXXX <#> <DECIMAL> <TIME>", "[NUM]"),
        ("Open <URL> email <EMAIL>", "Open [URL] email [EMAIL]"),
        ("BWTY0803 4G 1GB X-ray", "BWTY0803 4G 1GB X-ray"),
        ("X X's X-ray X", "[NUM] X's X-ray [NUM]"),
        ("  Hello\tthere\nfriend  ", "Hello there friend"),
        ("１２３", "[NUM]"),
        ("🎉 مرحبا 你好", "🎉 مرحبا 你好"),
        ("[URL] [EMAIL] [PHONE] [MONEY] [NUM]", "[URL] [EMAIL] [PHONE] [MONEY] [NUM]"),
    ],
)
def test_reference_normalization(text, expected):
    assert normalize(text) == expected
    assert normalize(normalize(text)) == expected


def test_normalize_does_not_modify_input():
    text = "Win £1000! Visit https://example.com"
    normalize(text)
    assert text == "Win £1000! Visit https://example.com"
