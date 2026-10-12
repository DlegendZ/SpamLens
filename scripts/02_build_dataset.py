"""Reproduce the curated corpus with the guide's reference curation logic."""

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from smsspam.config import load_config
from smsspam.data.build import acceptance_errors, build_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    raw = config.path("raw")
    stats = build_dataset(raw / "SMSSpamCollection", raw / "super_sms_dataset.csv", config.path("processed"), config.seed)
    print(json.dumps(stats, indent=2, default=str))
    errors = acceptance_errors(stats)
    if errors:
        raise SystemExit("\n".join(errors))


if __name__ == "__main__":
    main()
