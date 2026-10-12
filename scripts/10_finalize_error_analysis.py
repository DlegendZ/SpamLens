"""Validate team annotations and generate the final T10 taxonomy and analysis."""

import json

import _bootstrap  # noqa: F401

from smsspam.config import load_config
from smsspam.evaluation.review import finalize_review

if __name__ == "__main__":
    try:
        print(json.dumps(finalize_review(load_config()), indent=2))
    except ValueError as error:
        raise SystemExit(str(error)) from error
