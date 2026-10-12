"""Train-only NLP ablation and source-drift comparisons with frozen settings."""

import json
import string

import joblib
import nltk
import pandas as pd
import spacy
from sklearn.base import clone
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.preprocessing import FunctionTransformer

from smsspam.config import Config
from smsspam.preprocessing import normalize
from smsspam.viz import plt, save_figure


def drift_experiment(config: Config, train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """E3 fixes raw threshold 0.5 for both regimes to isolate training-corpus drift."""
    rows = []
    for name in ("tfidf_nb", "tfidf_lr"):
        curated = joblib.load(config.path("models") / f"{name}.joblib")
        uci_train = train[train.source == "uci_sms_spam_collection"]
        uci_only = clone(curated).fit(uci_train.text, uci_train.label)
        for regime, model in [("uci_only", uci_only), ("curated", curated)]:
            for source, subset in test.groupby("source"):
                predicted = model.predict_proba(subset.text)[:, 1] >= 0.5
                rows.append({"model": name, "training_corpus": regime, "test_source": source, "n": len(subset), "threshold": 0.5,
                             "macro_f1": f1_score(subset.label, predicted, average="macro"), "spam_recall": recall_score(subset.label, predicted), "spam_precision": precision_score(subset.label, predicted, zero_division=0)})
    results = pd.DataFrame(rows)
    results.to_csv(config.path("reports") / "tables" / "T6_drift.csv", index=False, lineterminator="\n")
    figure, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    sources = ["uci_sms_spam_collection", "super_sms_dataset"]
    for axis, name in zip(axes, ("tfidf_nb", "tfidf_lr"), strict=True):
        for regime, offset, hatch in [("uci_only", -0.18, "//"), ("curated", 0.18, "xx")]:
            values = [results.loc[(results.model == name) & (results.training_corpus == regime) & (results.test_source == source), "spam_recall"].item() for source in sources]
            axis.bar([offset, 1 + offset], values, width=0.36, label=regime.replace("_", " "), color="white", edgecolor="black", hatch=hatch)
        axis.set(xticks=[0, 1], xticklabels=["UCI test", "Super test"], ylabel="Spam recall", ylim=(0, 1.05), title=name)
        axis.legend()
    save_figure(figure, config.path("reports") / "figures" / "F8_drift.png")
    return results


def ablation_experiment(config: Config, train: pd.DataFrame) -> pd.DataFrame:
    """E4 compares fixed NLP variants with CV exclusively within train.csv."""
    nltk.data.path.insert(0, str(config.root / ".cache" / "nltk"))
    stops = set(nltk.corpus.stopwords.words("english"))
    normalized = train.text.map(normalize).tolist()
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    variants = {
        "raw_lowercase": train.text.str.lower().tolist(),
        "normalize": normalized,
        "normalize_stopwords": [" ".join(word for word in text.split() if word.strip(string.punctuation).lower() not in stops) for text in normalized],
        "normalize_lemmas": [" ".join(token.lemma_ for token in document) for document in nlp.pipe(normalized, batch_size=512)],
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.seed)
    rows = []
    for name in ("tfidf_nb", "tfidf_lr"):
        original = joblib.load(config.path("models") / f"{name}.joblib")
        for variant, texts in variants.items():
            model = clone(original)
            model.set_params(normalize=FunctionTransformer(validate=False))
            with joblib.parallel_backend("loky", inner_max_num_threads=1):
                scores = cross_val_score(model, texts, train.label, cv=cv, scoring="f1_macro", n_jobs=config.values["classical"]["n_jobs"], error_score="raise")
            rows.append({"model": name, "variant": variant, "mean_macro_f1": scores.mean(), "std_macro_f1": scores.std(), "selection_split": "train", "cv_folds": 5})
            print(f"E4 {name} {variant}: {scores.mean():.4f}", flush=True)
    result = pd.DataFrame(rows)
    result.to_csv(config.path("reports") / "tables" / "T7_ablation.csv", index=False, lineterminator="\n")
    return result


def suspect_labels(config: Config, train: pd.DataFrame) -> pd.DataFrame:
    """Flag train-only out-of-fold disagreements at fixed confidence ≥0.99."""
    model = clone(joblib.load(config.path("models") / "tfidf_lr.joblib"))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.seed)
    with joblib.parallel_backend("loky", inner_max_num_threads=1):
        probabilities = cross_val_predict(model, train.text, train.label, cv=cv, method="predict_proba", n_jobs=config.values["classical"]["n_jobs"])[:, 1]
    mask = ((train.label == 0) & (probabilities >= 0.99)) | ((train.label == 1) & (probabilities <= 0.01))
    result = train.loc[mask, ["text", "label", "source"]].copy()
    result["probability"] = probabilities[mask]
    result["id"] = [f"train_{index}" for index in result.index]
    report = {"n_train": len(train), "n_suspects": len(result), "suspect_fraction": len(result) / len(train), "confidence_cutoff": 0.99, "method": "train-only five-fold out-of-fold M2 confident disagreement; published labels unchanged"}
    (config.path("reports") / "metrics" / "label_noise_audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return result
