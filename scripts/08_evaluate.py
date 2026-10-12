"""Evaluate frozen models on the untouched held-out test data; never select parameters."""

import _bootstrap  # noqa: F401

from smsspam.config import load_config
from smsspam.evaluation.experiments import run_experiments

if __name__ == "__main__":
    run_experiments(load_config())
