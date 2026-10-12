"""Streamlit AppTest coverage for single/batch flows and graceful degradation."""

from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from smsspam import inference
from smsspam.app_support import DISCLAIMER

APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"


@pytest.fixture
def app(monkeypatch):
    """Fast app checks simulate missing M3 while exercising real committed M2/M1."""
    original = inference.load_predictor

    def load(name):
        if name == "distilbert":
            raise OSError("Simulated offline model load")
        return original(name)

    st.cache_resource.clear()
    monkeypatch.setattr(inference, "load_predictor", load)
    yield AppTest.from_file(APP, default_timeout=30).run()
    st.cache_resource.clear()


def test_app_loads_four_tabs_and_forced_transformer_failure_falls_back(app):
    assert not app.exception
    assert app.selectbox(key="selected_model").value == "distilbert"
    assert [tab.label for tab in app.tabs] == ["Check a message", "Batch check", "Model performance", "About & limitations"]
    assert any("DistilBERT is unavailable" in warning.value for warning in app.warning)
    assert app.button(key="check_message").disabled
    assert any("Paste a message" in message.value for message in app.info)


def test_example_check_renders_verdict_and_explanation(app):
    app.button(key="example_0").click().run()
    assert app.text_area(key="sms_text").value
    app.button(key="check_message").click().run()
    assert not app.exception
    assert any("Likely spam" in title.value for title in app.subheader)
    assert any(DISCLAIMER == caption.value for caption in app.caption)
    assert app.session_state["history"][0]["Model"] == "TF-IDF + Logistic Regression"
    assert app.session_state["explanations"]


def test_non_english_warns_and_still_predicts(app):
    app.text_area(key="sms_text").set_value("Saya sedang pergi bersama teman untuk makan").run()
    assert any("may not be in English" in warning.value for warning in app.warning)
    app.button(key="check_message").click().run()
    assert not app.exception
    assert len(app.session_state["history"]) == 1


def test_oversized_text_shows_truncation_warning(app):
    app.text_area(key="sms_text").set_value("FREE " * 250).run()
    assert len(app.text_area(key="sms_text").value) == 1000
    assert any("was truncated" in warning.value for warning in app.warning)


def test_history_and_explanation_cache_are_bounded_and_can_be_cleared(app):
    for index in range(6):
        app.text_area(key="sms_text").set_value(f"Hello friend, see you at lunch {index}").run()
        app.button(key="check_message").click().run()
    assert len(app.session_state["history"]) == 5
    assert len(app.session_state["explanations"]) == 5
    app.button(key="clear_history").click().run()
    assert "history" not in app.session_state
    assert "explanations" not in app.session_state


def test_batch_upload_round_trip_and_download(app):
    payload = (APP.parent / "assets" / "sample_messages.csv").read_bytes()
    app.file_uploader(key="batch_file").set_value(("sample.csv", payload, "text/csv")).run()
    app.button(key="check_batch").click().run()
    assert not app.exception
    result = app.session_state["batch_result"][2]
    assert len(result) == 10
    assert {"text", "label", "spam_probability", "risk_band"}.issubset(result.columns)
    assert app.download_button(key="results_csv")
    assert any(DISCLAIMER == caption.value for caption in app.caption)


def test_batch_invalid_header_has_clear_error(app):
    app.file_uploader(key="batch_file").set_value(("bad.csv", b"body\nhello\n", "text/csv")).run()
    assert not app.exception
    assert any("column named text" in error.value for error in app.error)


def test_model_selection_updates_result_model(app):
    app.selectbox(key="selected_model").set_value("tfidf_nb").run()
    app.button(key="example_3").click().run()
    app.button(key="check_message").click().run()
    assert not app.exception
    assert app.session_state["history"][0]["Model"] == "TF-IDF + Naive Bayes"


@pytest.mark.parametrize("text", ["🎉", "مرحبا 你好", " \t ", '<script>alert("hello")</script>'])
def test_edge_inputs_do_not_crash(app, text):
    app.text_area(key="sms_text").set_value(text).run()
    if not app.button(key="check_message").disabled:
        app.button(key="check_message").click().run()
    assert not app.exception


@pytest.mark.slow
def test_actual_distilbert_app_prediction():
    if not (APP.parents[1] / "models" / "distilbert" / "config.json").exists():
        pytest.skip("Local DistilBERT absent")
    st.cache_resource.clear()
    app = AppTest.from_file(APP, default_timeout=90).run()
    app.button(key="example_0").click().run()
    app.button(key="check_message").click().run()
    assert not app.exception
    assert app.session_state["history"][0]["Model"] == "DistilBERT"
    st.cache_resource.clear()
