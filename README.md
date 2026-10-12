# SpamLens

English SMS spam classification with artefact-aware data curation, classical
baselines, and fine-tuned DistilBERT. The implementation follows
[docs/EXECUTION_GUIDE.md](docs/EXECUTION_GUIDE.md).

## Current status

Repository scaffold is ready. The data and training pipeline are being implemented
in the guide's task order. See [implementation status](docs/IMPLEMENTATION_STATUS.md)
for measured completion evidence. No deployment results are available yet.

## Local setup (PowerShell)

Use Python 3.11 or 3.12. Install the appropriate PyTorch build from the
[official selector](https://pytorch.org/get-started/locally/), then install training
dependencies in a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-train.txt
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\ruff.exe check .
.\.venv\Scripts\python.exe -m pytest -q
```

Raw/processed datasets and transformer weights stay outside Git. All commits in
this implementation session remain local; no push or publication is performed.
