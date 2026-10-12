"""Generate corpus EDA; F4 is generated after fitting the selected M2 at T09."""

import _bootstrap  # noqa: F401

from smsspam.config import load_config
from smsspam.data.eda import generate_eda

if __name__ == "__main__":
    generate_eda(load_config())
    print("Generated F1–F3 and T1–T2. F4 follows selected M2 training (T09).")
