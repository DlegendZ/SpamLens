"""Fine-tune M3 with train/validation data and save the best local checkpoint."""

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from smsspam.config import load_config
from smsspam.models.transformer import train_distilbert

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    train_distilbert(load_config(args.config))
