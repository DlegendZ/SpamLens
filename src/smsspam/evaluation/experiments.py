"""Run frozen-model E1–E8 and export every measured result to reports/."""

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import precision_recall_curve, roc_curve

from smsspam.calibration import calibration_metrics
from smsspam.config import Config
from smsspam.data.build import sha256
from smsspam.evaluation.classical_experiments import (
    ablation_experiment,
    drift_experiment,
    suspect_labels,
)
from smsspam.evaluation.metrics import bootstrap_intervals, classification_metrics, mcnemar
from smsspam.evaluation.perturb import PERTURBATIONS, perturb_many
from smsspam.evaluation.review import export_review
from smsspam.inference import load_predictor
from smsspam.models.runtime import MODEL_NAMES
from smsspam.viz import plt, save_figure


def write_json(path: Path, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


def confusion_figure(result: dict, name: str, folder: Path) -> None:
    matrix = np.asarray(result["confusion_matrix"])
    figure, axis = plt.subplots(figsize=(4, 4))
    axis.imshow(matrix, cmap="Greys")
    for (row, column), value in np.ndenumerate(matrix):
        axis.text(column, row, str(value), ha="center", va="center", color="white" if value > matrix.max() / 2 else "black")
    axis.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Ham", "Spam"], yticklabels=["Ham", "Spam"], xlabel="Predicted label", ylabel="Actual label", title=f"F5: {name}")
    save_figure(figure, folder / f"F5_confusion_{name}.png")


def main_comparison(config: Config, test, predictors) -> tuple[dict, dict, dict]:
    reports = config.path("reports")
    results, probabilities, predicted = {}, {}, {}
    rows = []
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for index, (name, predictor) in enumerate(predictors.items()):
        probability = predictor.predict_proba(test.text)
        threshold = predictor.settings["threshold"]
        probabilities[name] = probability
        predicted[name] = probability >= threshold
        result = classification_metrics(test.label, probability, threshold)
        result["confidence_intervals"] = bootstrap_intervals(test.label, predicted[name], seed=config.seed)
        results[name] = result
        row = {"model": name, **{key: value for key, value in result.items() if not isinstance(value, (dict, list))}}
        for metric, interval in result["confidence_intervals"].items():
            row[f"{metric}_ci_low"], row[f"{metric}_ci_high"] = interval["lower"], interval["upper"]
        rows.append(row)
        confusion_figure(result, name, reports / "figures")
        fpr, tpr, _ = roc_curve(test.label, probability)
        precision, recall, _ = precision_recall_curve(test.label, probability)
        style = ["-", "--", ":"][index]
        axes[0].plot(fpr, tpr, color="black", linestyle=style, label=name)
        axes[1].plot(recall, precision, color="black", linestyle=style, label=name)
        print(f"E1 {name}: test macro-F1={result['macro_f1']:.4f}, FPR={result['ham_fpr']:.4f}", flush=True)
    axes[0].set(xlabel="Ham false-positive rate", ylabel="Spam recall", title="ROC curves")
    axes[1].set(xlabel="Spam recall", ylabel="Spam precision", title="Precision–recall curves")
    for axis in axes:
        axis.legend()
    save_figure(figure, reports / "figures" / "F6_roc_pr.png")
    pd.DataFrame(rows).to_csv(reports / "tables" / "T4_main_results.csv", index=False, lineterminator="\n")
    write_json(reports / "metrics" / "main_results.json", results)
    paired = {f"{a}_vs_{b}": mcnemar(test.label, predicted[a], predicted[b]) for a, b in itertools.combinations(predictors, 2)}
    write_json(reports / "metrics" / "mcnemar.json", paired)
    return results, probabilities, paired


def natural_prevalence(config: Config, test, predictors) -> dict:
    spam = test[test.label == 1]
    ham = test[test.label == 0]
    target = int(round(len(spam) * (1 - config.values["natural_prevalence"]) / config.values["natural_prevalence"]))
    pool = pd.read_csv(config.path("processed") / "unused_ham_pool.csv")
    combined = pd.concat([spam, ham, pool.sample(n=target - len(ham), random_state=config.seed)], ignore_index=True)
    results = {}
    for name, predictor in predictors.items():
        result = classification_metrics(combined.label, predictor.predict_proba(combined.text), predictor.settings["threshold"])
        result.update({"spam_prevalence": float(combined.label.mean()), "false_alarms_per_1000_ham": result["ham_fpr"] * 1000})
        results[name] = result
        print(f"E2 {name}: spam precision={result['spam_precision']:.4f}", flush=True)
    pd.DataFrame([{"model": name, **{key: value for key, value in result.items() if not isinstance(value, (dict, list))}} for name, result in results.items()]).to_csv(config.path("reports") / "tables" / "T5_natural_prevalence.csv", index=False, lineterminator="\n")
    write_json(config.path("reports") / "metrics" / "natural_prevalence.json", results)
    return results


def reliability_axis(axis, before: dict, after: dict, title: str) -> None:
    axis.plot([0, 1], [0, 1], "k:", label="Perfect calibration")
    for result, label, style in [(before, "Before", "k--o"), (after, "After", "k-s")]:
        bins = [row for row in result["bins"] if row["count"]]
        axis.plot([row["mean_probability"] for row in bins], [row["spam_frequency"] for row in bins], style, markersize=4, label=label)
    axis.set(xlabel="Mean spam score", ylabel="Observed spam fraction", title=title, xlim=(0, 1), ylim=(0, 1))
    axis.legend()


def calibration_experiment(config: Config, test, predictors, probabilities) -> dict:
    figure, axes = plt.subplots(1, 3, figsize=(13, 4))
    results, rows = {}, []
    for axis, (name, predictor) in zip(axes, predictors.items(), strict=True):
        raw = expit(predictor.scorer.logits(test.text))
        before = calibration_metrics(test.label, raw)
        after = calibration_metrics(test.label, probabilities[name])
        results[name] = {"before": before, "after": after, "probability_wording_allowed": after["ece"] <= 0.03}
        rows.extend({"model": name, "stage": stage, **{key: value for key, value in result.items() if key != "bins"}} for stage, result in [("before", before), ("after", after)])
        reliability_axis(axis, before, after, name)
        individual, individual_axis = plt.subplots(figsize=(5, 4))
        reliability_axis(individual_axis, before, after, name)
        save_figure(individual, config.path("reports") / "figures" / f"F7_reliability_{name}.png")
    save_figure(figure, config.path("reports") / "figures" / "F7_reliability.png")
    pd.DataFrame(rows).to_csv(config.path("reports") / "tables" / "T8_calibration.csv", index=False, lineterminator="\n")
    write_json(config.path("reports") / "metrics" / "test_calibration.json", results)
    return results


def robustness_experiment(config: Config, test, predictors, main_results, baseline_probabilities) -> pd.DataFrame:
    rows = []
    for kind in PERTURBATIONS:
        texts = perturb_many(test.text, kind, config.seed)
        for name, predictor in predictors.items():
            probability = predictor.predict_proba(texts)
            threshold = predictor.settings["threshold"]
            result = classification_metrics(test.label, probability, threshold)
            rows.append({"model": name, "perturbation": kind, "spam_recall": result["spam_recall"], "ham_fpr": result["ham_fpr"],
                         "recall_change": result["spam_recall"] - main_results[name]["spam_recall"], "fpr_change": result["ham_fpr"] - main_results[name]["ham_fpr"],
                         "flip_rate": float(((probability >= threshold) != (baseline_probabilities[name] >= threshold)).mean())})
        print(f"E6 completed: {kind}", flush=True)
    result = pd.DataFrame(rows)
    result.to_csv(config.path("reports") / "tables" / "T9_robustness.csv", index=False, lineterminator="\n")
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    for axis, column, title in [(axes[0], "recall_change", "Spam recall change"), (axes[1], "fpr_change", "Ham FPR change")]:
        for name, marker in zip(MODEL_NAMES, ("o", "s", "^"), strict=True):
            values = result[result.model == name].set_index("perturbation").loc[list(PERTURBATIONS), column]
            axis.plot(range(len(PERTURBATIONS)), values, color="black", marker=marker, linestyle={"o": "-", "s": "--", "^": ":"}[marker], label=name)
        axis.axhline(0, color="gray", linewidth=0.8)
        axis.set(xticks=range(len(PERTURBATIONS)), xticklabels=[name.replace("_", " ") for name in PERTURBATIONS], ylabel="Change (absolute fraction)", title=title)
        axis.tick_params(axis="x", labelrotation=35)
        axis.legend()
    save_figure(figure, config.path("reports") / "figures" / "F9_robustness.png")
    return result


def efficiency_experiment(config: Config) -> dict:
    reports = config.path("reports")
    latency = json.loads((reports / "metrics" / "cpu_latency.json").read_text(encoding="utf-8"))
    rows = []
    for name in MODEL_NAMES:
        path = config.path("models") / ("distilbert" if name == "distilbert" else f"{name}.joblib")
        size = sum(file.stat().st_size for file in path.rglob("*") if file.is_file()) if path.is_dir() else path.stat().st_size
        training = json.loads((reports / "metrics" / ("training_distilbert.json" if name == "distilbert" else f"cv_{name}.json")).read_text(encoding="utf-8"))
        rows.append({"model": name, "model_bytes": size, "training_seconds": training["training_seconds"], **latency[name]})
    pd.DataFrame(rows).to_csv(reports / "tables" / "T11_efficiency.csv", index=False, lineterminator="\n")
    return latency


def success_criteria(config: Config, main, natural, paired, calibration, latency) -> dict:
    name = config.values["app"]["default_model"]
    tests = {
        "SC1": (max(row["macro_f1"] for row in main.values()), 0.98, ">="),
        "SC2": (main[name]["ham_fpr"], 0.01, "<="),
        "SC3": (natural[name]["spam_precision"], 0.93, ">="),
        "SC4": (paired["tfidf_nb_vs_distilbert"]["p_value"], 0.05, "<"),
        "SC5": (calibration[name]["after"]["ece"], 0.03, "<="),
        "SC6": (latency[name]["p95_ms"], 500, "<="),
    }
    results = {}
    for criterion, (value, target, operator) in tests.items():
        passed = value >= target if operator == ">=" else value <= target if operator == "<=" else value < target
        if criterion == "SC4":
            passed = passed and main[name]["macro_f1"] > main["tfidf_nb"]["macro_f1"]
        results[criterion] = {"value": value, "target": target, "operator": operator, "passed": bool(passed)}
    results["SC7"] = {"passed": None, "reason": "Application and public deployment await T15/T18; no publication is authorized"}
    write_json(config.path("reports") / "metrics" / "success_criteria.json", results)
    return results


def run_experiments(config: Config) -> dict:
    artifacts = [config.path("models") / name for name in ("tfidf_nb.joblib", "tfidf_lr.joblib", "calibration.json", "distilbert/model.safetensors")]
    frozen = {str(path.relative_to(config.root)): sha256(path) for path in artifacts}
    processed = config.path("processed")
    train, test = pd.read_csv(processed / "train.csv"), pd.read_csv(processed / "test.csv")
    predictors = {name: load_predictor(name) for name in MODEL_NAMES}
    main, probabilities, paired = main_comparison(config, test, predictors)
    natural = natural_prevalence(config, test, predictors)
    drift_experiment(config, train, test)
    print("E3 completed", flush=True)
    ablation_experiment(config, train)
    calibration = calibration_experiment(config, test, predictors, probabilities)
    print("E5 completed", flush=True)
    robustness_experiment(config, test, predictors, main, probabilities)
    suspects = suspect_labels(config, train)
    deployed = config.values["app"]["default_model"]
    review = export_review(config, test, probabilities[deployed], predictors[deployed].settings["threshold"], suspects)
    latency = efficiency_experiment(config)
    criteria = success_criteria(config, main, natural, paired, calibration, latency)
    after = {str(path.relative_to(config.root)): sha256(path) for path in artifacts}
    if frozen != after:
        raise RuntimeError("A frozen artifact changed during evaluation")
    write_json(config.path("reports") / "metrics" / "evaluation_manifest.json", {"seed": config.seed, "sha256_artifacts": frozen, "sha256_train": sha256(processed / "train.csv"), "sha256_test": sha256(processed / "test.csv"), "models_changed": False, "E7_status": "human categories pending; T10 is intentionally not fabricated"})
    print(json.dumps({"success_criteria": criteria, "manual_review": review}, indent=2))
    return criteria
