"""Record runtime versions and verify the local CUDA training requirement."""

import argparse
import json
import platform
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from zoneinfo import ZoneInfo

import _bootstrap  # noqa: F401
import torch

from smsspam.config import load_config

PACKAGES = (
    "torch", "pandas", "numpy", "scikit-learn", "scipy", "ftfy", "joblib",
    "transformers", "accelerate", "datasets", "huggingface_hub", "safetensors",
    "nltk", "spacy", "matplotlib", "pyyaml", "streamlit", "pytest", "ruff", "jupyter",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()
    packages = {}
    for package in PACKAGES:
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    cuda = torch.cuda.is_available()
    report = {
        "recorded_at": datetime.now(ZoneInfo("Asia/Jakarta")).isoformat(),
        "python": platform.python_version(),
        "os": platform.platform(),
        "cuda_available": cuda,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if cuda else None,
        "packages": packages,
        "seed": load_config().seed,
    }
    output = load_config().path("reports") / "metrics" / "environment.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if args.require_cuda and not cuda:
        raise SystemExit("CUDA is required for T02 on this machine; install a CUDA PyTorch build.")
    if any(value is None for value in packages.values()):
        raise SystemExit("Training dependencies are missing; see environment.json.")


if __name__ == "__main__":
    main()
