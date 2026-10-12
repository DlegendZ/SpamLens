"""The human-review gate rejects incomplete or accidentally deleted rows."""

import numpy as np
import pandas as pd
import pytest

from smsspam.config import Config
from smsspam.evaluation.review import export_review, finalize_review


@pytest.fixture
def review_project(tmp_path):
    config = Config(tmp_path, {"paths": {"reports": "reports"}, "app": {"default_model": "distilbert"}})
    (tmp_path / "reports" / "metrics").mkdir(parents=True)
    (tmp_path / "reports" / "tables").mkdir()
    test = pd.DataFrame({"text": ["A notice.", "A claim."], "label": [0, 1], "source": ["fixture", "fixture"]})
    suspects = pd.DataFrame(columns=["id", "text", "label", "source", "probability"])
    export_review(config, test, np.array([0.8, 0.2]), 0.5, suspects)
    return config


def test_manual_categories_are_required(review_project):
    with pytest.raises(ValueError, match="need a valid manual category"):
        finalize_review(review_project)


def test_deleted_error_rows_are_rejected(review_project):
    path = review_project.path("reports") / "review" / "errors_private.csv"
    frame = pd.read_csv(path, keep_default_na=False).iloc[:1].copy()
    frame["category"] = "probable_label_noise"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="row count changed"):
        finalize_review(review_project)


def test_complete_human_review_generates_taxonomy(review_project):
    path = review_project.path("reports") / "review" / "errors_private.csv"
    frame = pd.read_csv(path, keep_default_na=False)
    frame["category"] = "probable_label_noise"
    frame["anonymized_example"] = ["An anonymized notice.", "An anonymized claim."]
    frame.to_csv(path, index=False)
    status = finalize_review(review_project)
    assert status["manual_review_complete"]
    taxonomy = pd.read_csv(review_project.path("reports") / "tables" / "T10_error_taxonomy.csv")
    assert taxonomy["count"].sum() == 2
