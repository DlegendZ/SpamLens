"""Generate measured corpus figures F1–F3 and source/curation tables T1–T2."""

import json

import nltk
import numpy as np
import pandas as pd

from smsspam.config import Config
from smsspam.preprocessing import normalize
from smsspam.viz import plt, save_figure


def generate_eda(config: Config) -> None:
    processed = config.path("processed")
    reports = config.path("reports")
    figures, tables = reports / "figures", reports / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    stats = json.loads((processed / "data_stats.json").read_text(encoding="utf-8"))
    full = pd.read_csv(processed / "sms_spam_curated_v1.csv")
    counts = [stats["raw"]["uci"], stats["raw"]["super"], full.label.value_counts().to_dict()]
    x = np.arange(3)
    figure, axis = plt.subplots(figsize=(7, 4))
    for label, name, offset, hatch in [(0, "Ham", -0.18, "//"), (1, "Spam", 0.18, "xx")]:
        values = [count.get(str(label), count.get(label, 0)) for count in counts]
        axis.bar(x + offset, values, width=0.36, label=name, color="white", edgecolor="black", hatch=hatch)
    axis.set(xticks=x, xticklabels=["Raw UCI", "Raw Super", "Curated"], ylabel="Messages", title="F1: Class distribution by corpus")
    axis.legend()
    save_figure(figure, figures / "F1_class_distribution.png")

    steps = pd.DataFrame.from_dict(stats["steps"], orient="index")
    steps.index.name = "curation_step"
    steps.to_csv(tables / "T2_curation.csv", lineterminator="\n")
    figure, axis = plt.subplots(figsize=(8, 5))
    y = np.arange(len(steps))
    for column, name, offset, hatch in [("removed_ham", "Ham", -0.18, "//"), ("removed_spam", "Spam", 0.18, "xx")]:
        axis.barh(y + offset, steps[column], height=0.36, label=name, color="white", edgecolor="black", hatch=hatch)
    axis.set(yticks=y, yticklabels=[name.replace("_", " ") for name in steps.index], xlabel="Messages removed", title="F2: Curation removals by class")
    axis.invert_yaxis()
    axis.legend()
    save_figure(figure, figures / "F2_curation_funnel.png")

    lengths = full.text.str.split().str.len()
    figure, axis = plt.subplots(figsize=(7, 4))
    for label, name, hatch in [(0, "Ham", "//"), (1, "Spam", "xx")]:
        axis.hist(lengths[full.label == label], bins=np.arange(0, 165, 5), histtype="stepfilled", facecolor="none", edgecolor="black", hatch=hatch, label=name)
    axis.set(xlabel="Words per message", ylabel="Messages", title="F3: Message length by class")
    axis.legend()
    save_figure(figure, figures / "F3_message_lengths.png")
    full.assign(words=lengths).groupby("label").words.describe(percentiles=[0.5, 0.95, 0.99]).to_csv(tables / "message_lengths.csv", lineterminator="\n")

    source_rows = [
        {"source": "uci_sms_spam_collection", "license": "CC BY 4.0", "citation": "Almeida and Hidalgo (2011), DOI 10.24432/C5CC84", "url": config.values["sources"]["uci"]["url"]},
        {"source": "super_sms_dataset", "license": "Research release; no explicit license; academic policy; no redistribution", "citation": "Salman et al. (2024), DOI 10.1109/ACCESS.2024.3364671", "url": config.values["sources"]["super"]["url"]},
    ]
    pd.DataFrame(source_rows).to_csv(tables / "T1_sources_licenses.csv", index=False, lineterminator="\n")
    # Vocabulary summaries are descriptive only; the final model grid remains fixed.
    token_rows = []
    for label in (0, 1):
        tokens = (token.lower() for text in full.loc[full.label == label, "text"] for token in normalize(text).split())
        token_rows.extend({"label": label, "token": token, "count": count} for token, count in nltk.FreqDist(tokens).most_common(20))
    pd.DataFrame(token_rows).to_csv(tables / "top_tokens.csv", index=False, lineterminator="\n")
