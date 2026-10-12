"""Fail if any of the eight curated-dataset acceptance invariants is violated."""

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from smsspam.config import load_config
from smsspam.data.audit import save_audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    report = save_audit(config.path("processed"), config.path("reports") / "metrics" / "data_audit.json", config.seed)
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
