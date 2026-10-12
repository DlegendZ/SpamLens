"""Download and verify the two original public SMS datasets."""

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from smsspam.config import load_config
from smsspam.data.download import download_sources


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    print(json.dumps(download_sources(load_config(args.config)), indent=2))


if __name__ == "__main__":
    main()
