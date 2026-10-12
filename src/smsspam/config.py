"""Load project configuration independently of the caller's working directory."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Config:
    """Configuration values and paths resolved relative to the repository."""

    root: Path
    values: dict[str, Any]

    @property
    def seed(self) -> int:
        return int(self.values["seed"])

    def path(self, name: str) -> Path:
        return self.root / self.values["paths"][name]


def load_config(path: Path | None = None) -> Config:
    """Read YAML configuration; custom configurations use the same project root."""
    config_path = path or PROJECT_ROOT / "configs" / "config.yaml"
    with config_path.open(encoding="utf-8") as handle:
        values = yaml.safe_load(handle)
    return Config(root=PROJECT_ROOT, values=values)
