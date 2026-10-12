"""Fit both classical grids on train.csv; do not load validation or test data."""

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from smsspam.config import load_config
from smsspam.models.classical import train_classical

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    train_classical(load_config(args.config))
