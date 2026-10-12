"""SpamLens Streamlit application; user messages stay in session memory only."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from smsspam import inference  # noqa: E402
from smsspam.app_support import (  # noqa: E402
    BAND_LABELS,
    DISCLAIMER,
    LANGUAGE_WARNING,
    MODEL_LABELS,
    csv_bytes,
    guard_text,
    likely_non_english,
    parse_batch,
    predict_batch,
)
from smsspam.config import load_config  # noqa: E402
from smsspam.explain import explain, highlighted_text  # noqa: E402

CONFIG = load_config()
REPORTS = CONFIG.path("reports")
REPO_URL = "https://github.com/DlegendZ/SpamLens"
MODEL_DESCRIPTIONS = {
    "distilbert": "A compact transformer that learns message context.",
    "tfidf_lr": "Word and character patterns with logistic regression.",
    "tfidf_nb": "A simple, interpretable word-pattern baseline.",
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


@st.cache_resource(show_spinner=False)
def cached_predictor(name: str):
    """Cache only model resources; no message content enters this shared cache."""
    return inference.load_predictor(name)


def resolve_predictor(requested: str):
    failed = st.session_state.setdefault("failed_models", set())
    try:
        if requested in failed:
            raise RuntimeError("Previously unavailable in this session")
        return cached_predictor(requested)
    except Exception:
        # UI errors intentionally omit exception text, which can contain sensitive input.
        if requested != "distilbert":
            st.error("This model could not load. Check the local model files and try again.")
            st.stop()
        failed.add(requested)
        st.warning("DistilBERT is unavailable. Using TF-IDF + Logistic Regression for this session.")
        try:
            return cached_predictor("tfidf_lr")
        except Exception:
            st.error("The fallback model could not load. Restore models/tfidf_lr.joblib and restart the app.")
            st.stop()


def sidebar():
    with st.sidebar:
        st.title("SMS Spam Checker")
        st.caption("SpamLens · English SMS")
        names = list(MODEL_LABELS)
        requested = st.selectbox("Model", names, index=names.index(CONFIG.values["app"]["default_model"]),
                                 format_func=MODEL_LABELS.get, key="selected_model")
        st.caption(MODEL_DESCRIPTIONS[requested])
        if "distilbert" in st.session_state.get("failed_models", set()):
            if st.button("Retry DistilBERT", key="retry_model"):
                st.session_state["failed_models"].discard("distilbert")
        with st.spinner("Loading model…"):
            predictor = resolve_predictor(requested)
        metrics = read_json(REPORTS / "metrics" / "main_results.json").get(predictor.name, {})
        st.divider()
        st.caption(f"Active model: {MODEL_LABELS[predictor.name]}")
        st.metric("Spam decision threshold", f"{predictor.settings['threshold']:.2f}")
        if metrics:
            st.metric("Test macro-F1", f"{metrics['macro_f1']:.3f}")
        st.link_button("View project on GitHub", REPO_URL)
        st.caption("Your messages stay in this session. They are never saved to files, logs, or analytics.")
    return predictor


def word_vocabulary() -> set[str]:
    model = cached_predictor("tfidf_lr").scorer.model
    word_vectorizer = dict(model.named_steps["features"].transformer_list)["word"]
    return set(word_vectorizer.vocabulary_)


def choose_example(text: str) -> None:
    st.session_state["sms_text"] = text
    st.session_state.pop("last_result", None)


def enforce_message_limit() -> None:
    """Enforce the cap before widget creation, retaining evidence for the warning."""
    raw = st.session_state.get("sms_text", "")
    if len(raw) > CONFIG.values["app"]["max_chars"]:
        st.session_state["sms_text"] = raw[:CONFIG.values["app"]["max_chars"]]
        st.session_state["text_truncated"] = True


def display_result(text, result, predictor):
    with st.container(border=True):
        symbols = {"likely_spam": "🚫", "suspicious": "⚠️", "likely_ham": "✅"}
        st.subheader(f"{symbols[result.risk_band]} {BAND_LABELS[result.risk_band]}")
        calibration = read_json(REPORTS / "metrics" / "test_calibration.json").get(result.model_name, {})
        wording = "Calibrated spam probability" if calibration.get("probability_wording_allowed", False) else "Spam score"
        st.metric(wording, f"{result.spam_probability:.1%}")
        st.progress(result.spam_probability)
        st.caption(f"{MODEL_LABELS[result.model_name]} · {result.latency_ms:.1f} ms")
        st.caption(DISCLAIMER)
    with st.expander("Why this result?"):
        cache = st.session_state.setdefault("explanations", {})
        key = (result.model_name, text)
        try:
            if key not in cache:
                cache[key] = explain(predictor, text, CONFIG.values["app"]["explain_top_k"], CONFIG.values["app"]["explain_max_words"])
                while len(cache) > 5:
                    del cache[next(iter(cache))]
            effects = cache[key]
            st.markdown("<div>" + highlighted_text(text, effects) + "</div>", unsafe_allow_html=True)
            st.dataframe(pd.DataFrame([{"Word": item.word, "Effect on spam probability (pp)": round(item.effect_pp, 2)} for item in effects]), hide_index=True, width="stretch")
            st.caption("Positive values raise the spam score; negative values lower it. This compares word removal and ignores word interactions.")
            if len(text.split()) > CONFIG.values["app"]["explain_max_words"]:
                st.caption("Explanation covers the first 60 words; the prediction uses the full message within the model's token limit.")
        except Exception:
            st.info("The prediction is ready, but its explanation is temporarily unavailable.")


def single_tab(predictor):
    st.subheader("Check an SMS before you act")
    st.write("Paste a message to estimate spam risk. Verify unexpected requests with the official sender.")
    examples = read_json(ROOT / "app" / "assets" / "examples.json").get("single_examples", [])
    if examples:
        with st.expander("Try an example", expanded=True):
            columns = st.columns(3)
            for index, example in enumerate(examples):
                columns[index % 3].button(example["title"], key=f"example_{index}", on_click=choose_example, args=(example["text"],), width="stretch")
    # Streamlit's max_chars deserializer truncates before callbacks can detect it.
    # Enforce the same 1,000-character cap here so FR-A8 can show a truthful warning.
    enforce_message_limit()
    text = st.text_area("Message", placeholder="Paste an SMS here…", key="sms_text", on_change=enforce_message_limit, height=150)
    st.caption(f"{len(text):,} / {CONFIG.values['app']['max_chars']:,} characters")
    clean, truncated = guard_text(text, CONFIG.values["app"]["max_chars"])
    if truncated or st.session_state.pop("text_truncated", False):
        st.warning("Your message was longer than 1,000 characters and was truncated.")
    if not clean:
        st.info("Paste a message or choose an example to begin.")
    if clean and likely_non_english(clean, word_vocabulary()):
        st.warning(LANGUAGE_WARNING)
    if st.button("Check message", type="primary", disabled=not clean, key="check_message"):
        with st.spinner("Checking message…"):
            try:
                result = predictor.predict([clean])[0]
                st.session_state["last_result"] = (clean, result)
                history = st.session_state.setdefault("history", [])
                history.insert(0, {"Message": clean, "Result": BAND_LABELS[result.risk_band], "Spam score": f"{result.spam_probability:.1%}", "Model": MODEL_LABELS[result.model_name]})
                del history[5:]
            except Exception:
                st.error("This message could not be checked. Please try again.")
    previous = st.session_state.get("last_result")
    if previous and previous[0] == clean and previous[1].model_name == predictor.name:
        display_result(*previous, predictor)
    history = st.session_state.get("history", [])
    if history:
        with st.expander("Recent checks · this session only"):
            st.dataframe(pd.DataFrame(history), hide_index=True, width="stretch")
            if st.button("Clear history", key="clear_history"):
                for key in ("history", "last_result", "explanations"):
                    st.session_state.pop(key, None)
                st.rerun()


def batch_tab(predictor):
    st.subheader("Check a batch of messages")
    st.write("Upload a UTF-8 CSV with a text column: up to 1,000 rows and 2 MB.")
    sample = ROOT / "app" / "assets" / "sample_messages.csv"
    if sample.exists():
        st.download_button("Download sample CSV", sample.read_bytes(), file_name="sample_messages.csv", mime="text/csv", key="sample_csv")
    upload = st.file_uploader("Messages CSV", type=["csv"], key="batch_file")
    if upload is None:
        return
    try:
        frame = parse_batch(upload.getvalue(), CONFIG.values["app"]["batch_max_rows"])
    except ValueError as error:
        st.error(str(error))
        return
    st.caption(f"{len(frame):,} messages ready · {MODEL_LABELS[predictor.name]}")
    if frame.text.str.len().gt(CONFIG.values["app"]["max_chars"]).any():
        st.warning("Messages longer than 1,000 characters will be truncated before checking.")
    if st.button("Check batch", type="primary", key="check_batch"):
        progress = st.progress(0.0)
        try:
            result = predict_batch(frame, predictor, CONFIG.values["app"]["max_chars"], progress.progress)
            st.session_state["batch_result"] = (upload.file_id, predictor.name, result)
        except Exception:
            st.error("The batch could not be checked. Please try again with a smaller file.")
        finally:
            progress.empty()
    previous = st.session_state.get("batch_result")
    if previous and previous[:2] == (upload.file_id, predictor.name):
        result = previous[2]
        counts = result.risk_band.value_counts()
        st.write(" · ".join(f"{BAND_LABELS[band]}: {int(counts.get(band, 0))}" for band in BAND_LABELS))
        st.dataframe(result[["text", "label", "spam_probability", "risk_band"]], hide_index=True, width="stretch")
        st.caption(DISCLAIMER)
        st.download_button("Download results CSV", csv_bytes(result), file_name="spamlens_predictions.csv", mime="text/csv", key="results_csv")


def performance_tab(predictor):
    st.subheader("Measured model performance")
    st.caption("Frozen models, held-out test data, and validation-selected thresholds. No training runs inside this app.")
    table = REPORTS / "tables" / "T4_main_results.csv"
    if table.exists():
        frame = pd.read_csv(table)
        st.dataframe(frame[["model", "macro_f1", "spam_precision", "spam_recall", "ham_fpr"]], hide_index=True, width="stretch")
    st.write(f"Selected model: **{MODEL_LABELS[predictor.name]}**")
    for filename, caption in [(f"F5_confusion_{predictor.name}.png", "Confusion matrix · balanced test set"),
                              (f"F7_reliability_{predictor.name}.png", "Reliability before and after calibration")]:
        path = REPORTS / "figures" / filename
        if path.exists():
            st.image(str(path), caption=caption, width="stretch")
    st.subheader("Realistic spam prevalence: 13.4%")
    natural = REPORTS / "tables" / "T5_natural_prevalence.csv"
    if natural.exists():
        frame = pd.read_csv(natural)
        st.dataframe(frame[["model", "spam_precision", "spam_recall", "ham_fpr", "false_alarms_per_1000_ham"]], hide_index=True, width="stretch")
    st.caption("A 50/50 test set inflates precision compared with real-world prevalence.")
    stats = read_json(REPORTS / "metrics" / "data_stats.json")
    if stats:
        st.subheader("Dataset summary")
        st.write(f"{stats['final']['total']:,} curated messages · equal ham/spam · seed 42 · train/validation/test 70/15/15")
        for filename in ("T1_sources_licenses.csv", "T2_curation.csv"):
            path = REPORTS / "tables" / filename
            if path.exists():
                st.dataframe(pd.read_csv(path), hide_index=True, width="stretch")


def about_tab():
    st.subheader("An assistive first-line SMS filter")
    st.write("SpamLens estimates whether an English SMS resembles spam. It cannot verify a sender or guarantee that a message is safe.")
    st.markdown("**1. Normalize text** → **2. Estimate spam risk** → **3. Explain and verify**")
    st.markdown("**Data sources and attribution**")
    st.markdown("[UCI SMS Spam Collection](https://doi.org/10.24432/C5CC84), Almeida & Hidalgo (2011), CC BY 4.0; Almeida et al., DocEng 2011, DOI 10.1145/2034691.2034742.")
    st.markdown("[Super SMS Dataset](https://doi.org/10.1109/ACCESS.2024.3364671), Salman, Ikram & Kaafar (IEEE Access, 2024). Author-published research data without an explicit license file; this project uses it for non-commercial academic work and does not redistribute it.")
    st.markdown("[DistilBERT](https://arxiv.org/abs/1910.01108), Sanh et al. (2019), Apache-2.0 pretrained model. Project code: MIT.")
    st.markdown("**Limitations**")
    st.write("English SMS only; uneven regional and dialect coverage; training sources end in 2024; spam changes over time. Published labels contain borderline promotional or transactional messages. No URL reputation or sender checks are performed.")
    st.write("Obfuscated wording and informal or mixed-language messages can cause errors. The score is an automated estimate; verify unexpected requests independently.")
    st.markdown("**Privacy**")
    st.write("Messages and uploads are processed in memory and are never written to files, logs, or analytics. Recent checks and explanations belong only to this session and clear when its connection ends. Use Clear history to remove recent single-message checks immediately.")
    st.caption(DISCLAIMER)
    st.markdown("**Team · COMP6885001 — Natural Language Processing**")
    team = pd.read_csv(ROOT / "app" / "assets" / "team.csv", dtype=str, keep_default_na=False)
    st.dataframe(team, hide_index=True, width="stretch")


def main():
    st.set_page_config(page_title="SpamLens · SMS Spam Checker", page_icon="✉️", layout="centered")
    predictor = sidebar()
    st.caption("SPAMLENS / ENGLISH SMS")
    st.title("Pause. Check. Verify.")
    tabs = st.tabs(["Check a message", "Batch check", "Model performance", "About & limitations"])
    for tab, render in zip(tabs[:3], (single_tab, batch_tab, performance_tab), strict=True):
        with tab:
            render(predictor)
    with tabs[3]:
        about_tab()


if __name__ == "__main__":
    main()
