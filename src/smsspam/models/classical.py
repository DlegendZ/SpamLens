"""Train M1/M2 using only the train split and five-fold stratified CV."""

import json
from collections.abc import Sequence
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer

from smsspam.config import Config
from smsspam.preprocessing import normalize


def normalize_many(texts: Sequence[str]) -> list[str]:
    return [normalize(text) for text in texts]


def make_nb() -> Pipeline:
    return Pipeline([
        ("normalize", FunctionTransformer(normalize_many, validate=False)),
        ("tfidf", TfidfVectorizer(lowercase=True, sublinear_tf=True)),
        ("classifier", MultinomialNB()),
    ])


def make_lr(max_features: int = 200_000, seed: int = 42) -> Pipeline:
    features = FeatureUnion([
        ("word", TfidfVectorizer(lowercase=True, sublinear_tf=True, ngram_range=(1, 2), min_df=2)),
        ("char", TfidfVectorizer(lowercase=True, sublinear_tf=True, analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_features=max_features)),
    ])
    return Pipeline([
        ("normalize", FunctionTransformer(normalize_many, validate=False)),
        ("features", features),
        ("classifier", LogisticRegression(max_iter=3000, solver="liblinear", random_state=seed)),
    ])


def coefficient_figure(model: Pipeline, reports: Path) -> None:
    """F4 uses coefficients from the selected, training-only fitted M2."""
    from smsspam.viz import plt, save_figure

    names = model.named_steps["features"].get_feature_names_out()
    coefficients = model.named_steps["classifier"].coef_[0]
    low, high = np.argsort(coefficients)[:20], np.argsort(coefficients)[-20:]
    rows = []
    figure, axes = plt.subplots(1, 2, figsize=(12, 8))
    for axis, indices, title, hatch in [(axes[0], low, "Ham-indicative features", "//"), (axes[1], high, "Spam-indicative features", "xx")]:
        axis.barh(np.arange(len(indices)), coefficients[indices], color="white", edgecolor="black", hatch=hatch)
        axis.set(yticks=np.arange(len(indices)), yticklabels=[str(names[i]).replace("__", ": ") for i in indices], xlabel="Logistic regression coefficient", title=title)
        rows.extend({"feature": names[i], "coefficient": coefficients[i], "direction": "ham" if coefficients[i] < 0 else "spam"} for i in indices)
    figure.suptitle("F4: Selected M2 feature coefficients")
    save_figure(figure, reports / "figures" / "F4_model_coefficients.png")
    pd.DataFrame(rows).to_csv(reports / "tables" / "feature_coefficients.csv", index=False, lineterminator="\n")


def fit_grid(name: str, model: Pipeline, grid: dict, train: pd.DataFrame, config: Config) -> tuple[Pipeline, dict]:
    settings = config.values["classical"]
    model.memory = joblib.Memory(config.root / ".cache" / "sklearn", verbose=0)
    cv = StratifiedKFold(n_splits=settings["cv_folds"], shuffle=True, random_state=config.seed)
    search = GridSearchCV(model, grid, cv=cv, scoring="f1_macro", n_jobs=settings["n_jobs"], refit=True, error_score="raise", verbose=1)
    started = perf_counter()
    with joblib.parallel_backend("loky", inner_max_num_threads=1):
        search.fit(train.text, train.label)
    selected = search.best_estimator_
    selected.memory = None
    report = {
        "model": name, "selection_split": "train", "seed": config.seed,
        "cv_folds": settings["cv_folds"], "best_params": search.best_params_,
        "best_macro_f1": float(search.best_score_), "training_seconds": perf_counter() - started,
        "candidates": [
            {"params": params, "mean_macro_f1": float(mean), "std_macro_f1": float(std)}
            for params, mean, std in zip(search.cv_results_["params"], search.cv_results_["mean_test_score"], search.cv_results_["std_test_score"], strict=True)
        ],
    }
    return selected, report


def train_classical(config: Config) -> dict:
    from smsspam.data.build import sha256

    train_path = config.path("processed") / "train.csv"
    train = pd.read_csv(train_path)
    if not train.split.eq("train").all():
        raise ValueError("Classical training accepts only rows explicitly marked train")
    settings = config.values["classical"]
    configs = {
        "tfidf_nb": (make_nb(), {"tfidf__ngram_range": [(1, 1), (1, 2)], "tfidf__min_df": [1, 2], "classifier__alpha": settings["nb_alpha"]}),
        "tfidf_lr": (make_lr(settings["char_max_features"], config.seed), {"classifier__C": settings["lr_C"]}),
    }
    models_dir = config.path("models")
    metrics_dir = config.path("reports") / "metrics"
    models_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    reports = {}
    for name, (model, grid) in configs.items():
        selected, report = fit_grid(name, model, grid, train, config)
        path = models_dir / f"{name}.joblib"
        joblib.dump(selected, path, compress=3)
        size = path.stat().st_size
        if size >= 50_000_000:
            raise ValueError(f"{name} artifact exceeds the 50 MB limit; reduce char_max_features and document it")
        report.update({"artifact_bytes": size, "sha256_train": sha256(train_path), "sha256_artifact": sha256(path)})
        (metrics_dir / f"cv_{name}.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        reports[name] = report
        print(f"{name}: CV macro-F1 {report['best_macro_f1']:.4f}; artifact {size / 1_000_000:.2f} MB", flush=True)
        if name == "tfidf_lr":
            coefficient_figure(selected, config.path("reports"))
    rows = [{"model": name, "cv_macro_f1": report["best_macro_f1"], "selected_parameters": json.dumps(report["best_params"]), "training_seconds": report["training_seconds"], "artifact_bytes": report["artifact_bytes"]} for name, report in reports.items()]
    pd.DataFrame(rows).to_csv(config.path("reports") / "tables" / "T3_hyperparameters.csv", index=False, lineterminator="\n")
    return reports
