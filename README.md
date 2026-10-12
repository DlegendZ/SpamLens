# SpamLens

English SMS spam classification with artefact-aware data curation, classical
baselines, and fine-tuned DistilBERT. The implementation follows
[docs/EXECUTION_GUIDE.md](docs/EXECUTION_GUIDE.md).

## Current status

T01–T12 are implemented and verified. The pinned rebuild matches the reference
SHA-256 exactly (19,152 balanced messages), and all three trained models have
validation-only calibration. Quantitative E1–E8 results are generated. T14 requires
the team's manual error categories before T15 application work.

See [implementation status](docs/IMPLEMENTATION_STATUS.md) and
[manual review instructions](docs/MANUAL_REVIEW.md). The Streamlit app and public
deployment are pending; no public URL is available.

## Measured results

These values come from `reports/tables/T4_main_results.csv` and
`reports/tables/T5_natural_prevalence.csv`, not the planning guide.

| Model | Test macro-F1 | Ham FPR | Spam precision at 13.4% prevalence |
|---|---:|---:|---:|
| M1: TF-IDF + Naive Bayes | 0.9673 | 0.70% | 94.61% |
| M2: Word + char TF-IDF + Logistic Regression | 0.9854 | 0.90% | 95.71% |
| M3: DistilBERT | 0.9916 | 0.63% | 96.27% |

Model comparison includes 1,000 seeded bootstrap resamples and exact paired
McNemar tests. SC1–SC6 pass; SC7 awaits application and deployment verification.

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

## Run the implemented pipeline

```powershell
.\.venv\Scripts\python.exe scripts\00_environment.py --require-cuda
.\.venv\Scripts\python.exe scripts\01_download_data.py
.\.venv\Scripts\python.exe scripts\02_build_dataset.py
.\.venv\Scripts\python.exe scripts\03_audit_dataset.py
.\.venv\Scripts\python.exe scripts\03_document_data.py
.\.venv\Scripts\python.exe scripts\04_eda.py
.\.venv\Scripts\python.exe scripts\05_train_classical.py
.\.venv\Scripts\python.exe scripts\06_train_distilbert.py
.\.venv\Scripts\python.exe scripts\07_calibrate.py
.\.venv\Scripts\python.exe scripts\12_verify_predictors.py
.\.venv\Scripts\python.exe scripts\08_evaluate.py
.\.venv\Scripts\python.exe scripts\12_validate_runtime.py
```

For E4, install `en_core_web_sm` with `python -m spacy download en_core_web_sm`
and NLTK resources with `python -c "import nltk; nltk.download('stopwords')"`
inside the venv. Local training logs record Python 3.12.8 and CUDA PyTorch
2.6.0+cu124; library versions are in `reports/metrics/environment.json`.

After the team fills both private review queues, finalize T14:

```powershell
.\.venv\Scripts\python.exe scripts\10_finalize_error_analysis.py
```

Raw/processed datasets and transformer weights stay outside Git. All commits in
this implementation session remain local; no push or publication is performed.

The six team identities are recorded in `docs/contribution_log.md`. Roles remain
blank at the project owner's request. Attribution is in [DATA_LICENSES.md](DATA_LICENSES.md).
