# SMS Spam Detection — Software Requirements & Execution Specification (SRES)

| Field | Value |
|---|---|
| Project title | **Robust English SMS Spam Detection with Classical and Transformer Models on a Curated, Balanced Multi-Source Corpus** |
| Course | COMP6885001 — Natural Language Processing (Final Project) |
| Document type | Software Requirements & Execution Specification (SRD + implementation plan + evaluation protocol) |
| Version | 1.0 (planning baseline) |
| Audience | Coding agent (primary), team members, reviewer |
| Language of all code, comments, UI, report | English |
| Hardware assumed | Local NVIDIA RTX 3050 (6 GB VRAM) for fine-tuning; Google Colab T4 as fallback |
| Deployment | Local Streamlit + Streamlit Community Cloud (public URL) |

---

## 0. How to use this document (read first)

This document is the single source of truth for building, evaluating, deploying, and documenting the project. It is written so that a coding agent can execute it top to bottom without re-planning.

**Rules of engagement for the coding agent**

1. **Execute the task list in §17 in order.** Each task has inputs, outputs, and a Definition of Done (DoD). Do not start a task until the previous task's DoD is met, unless the task is explicitly marked *parallel*.
2. **Do not change scope.** Anything not in §3.1 is out of scope. If something in this spec is impossible (for example, a library API changed), choose the closest option that keeps the requirement's intent, record it in `docs/DECISIONS.md` (date, what, why, impact), and continue.
3. **The test split is sacred.** Never use `test.csv` (or the natural-prevalence test set) for training, hyperparameter search, threshold selection, calibration, early stopping, or feature selection. All tuning uses `train.csv` (cross-validation) and `val.csv`.
4. **Every number in the report must come from a file the code generated** (`reports/metrics/*.json`, `reports/tables/*.csv`). Never type results by hand. Never invent or "estimate" numbers.
5. **Determinism.** Global seed `42` everywhere (Python `random`, NumPy, PyTorch, scikit-learn `random_state`, HF `set_seed`). Record library versions in `reports/metrics/environment.json`.
6. **Do not commit datasets** (raw or processed) or the fine-tuned DistilBERT weights to Git. Commit code, small classical-model files, configs, figures, tables, and metrics.
7. **Plagiarism policy.** All code must be written for this project. When adapting a documented pattern (for example, the Hugging Face `Trainer` example), add a comment with the source URL. Cite every dataset, model, and library in the report.
8. **When unsure, prefer the simpler option that satisfies the requirement**, and log the choice in `docs/DECISIONS.md`.

**Document map**

| § | Content |
|---|---|
| 1 | Context, objectives, grading alignment |
| 2 | Audit of the original idea against the guideline, and what changed |
| 3 | Scope, users, functional and non-functional requirements |
| 4 | Data specification (sources, licensing, defects found, curation, splits) |
| 5 | Data pipeline implementation |
| 6 | Text normalisation and NLP preprocessing |
| 7 | Models (M1, M2, M3), training setup, hyperparameters |
| 8 | Calibration, thresholds, risk bands |
| 9 | Evaluation protocol and experiments E1–E10 |
| 10 | Application (Streamlit) specification |
| 11 | Repository structure, environment, configuration |
| 12 | Testing and quality assurance |
| 13 | Deployment (Hugging Face Hub + Streamlit Community Cloud) |
| 14 | Written report specification |
| 15 | Demo video specification |
| 16 | Ethics, privacy, licensing |
| 17 | Ordered task list with Definitions of Done |
| 18 | Timeline (Week 6 → Week 14) |
| 19 | Risks and mitigations |
| 20 | Final submission checklist |
| A–D | Appendices: reference code, reference results, references, glossary |

---

## 1. Context and objectives

### 1.1 Problem statement

Unwanted SMS (promotions, scams, phishing / "smishing") disrupt users and can cause financial loss. The project builds an NLP system that classifies an English SMS as **spam** or **ham** (legitimate). It returns a calibrated spam probability, a risk band, and a short explanation of which words drove the decision. The application is an **assistive first-line filter, not a security guarantee**, and the UI must say so.

### 1.2 Objectives

| ID | Objective | Measured by |
|---|---|---|
| O1 | Build a clean, balanced, licensed, documented SMS spam corpus from public sources | Data card + automated data-quality tests (§12) |
| O2 | Compare a classical baseline, a strong classical model, and a fine-tuned transformer | E1 results table with 95% CIs and McNemar test |
| O3 | Show *why* curation matters (concept drift, annotation artefacts) | E3 drift experiment + artefact audit |
| O4 | Ship a stable, user-friendly Streamlit application with clear inputs/outputs | App requirements FR-A* and NFR (§3), smoke tests |
| O5 | Produce an academic report (10–15 pages) and a demo video (< 5 min) | §14, §15 checklists |

### 1.3 Grading alignment (from the course guideline)

| Deliverable (weight) | Rubric points | What in this project earns them |
|---|---|---|
| **Application (40%)** | Functionality 20 | Single + batch prediction, 3 selectable models, calibrated probability, risk band, word-level explanation, model-performance tab |
| | Usability 10 | Example messages, clear result card, input validation messages, disclaimers, About tab, works on phone width |
| | Reliability 10 | Cached model loading, graceful fallback to classical model, input guards, unit + app smoke tests, public deployment |
| **Written report (40%)** | Methodology 15 | Data curation with quantified artefact removal, leakage controls, 3 models, calibration, rigorous protocol |
| | Analysis 15 | 10 experiments: main comparison with CIs, natural prevalence, drift, ablation, calibration, robustness, error taxonomy, efficiency |
| | Writing quality 10 | Required structure (§14), IEEE references, figures generated by code, 10–15 pages |
| **Demo video (20%)** | Clarity 10, Presentation 10 | Scripted 4:30 video (§15): problem → data → method → results → live demo → limitations |

Learning outcomes: LO1/LO2 (report background + methodology), LO3 (real-world spam filtering application), LO4 (constructed, deployed application).

---

## 2. Audit of the original idea against the guideline

### 2.1 Requirement compliance

| Guideline requirement | Original research PDF | Status | Action in this spec |
|---|---|---|---|
| Theme: text classification | Binary SMS spam/ham | ✅ | Kept |
| Language: Python | Python | ✅ | Kept |
| Libraries: NLTK, spaCy, HF Transformers | HF Transformers only | ⚠️ partial | NLTK + spaCy used in the preprocessing ablation (E4) and EDA |
| Deployment: Streamlit/Flask/FastAPI | Streamlit | ✅ | Kept; adds public Streamlit Cloud URL |
| Proposal: title, problem, dataset + licence, method + baseline, metrics + success criteria, ≥ 2–3 references | Mostly present; no numeric success criteria | ⚠️ | Proposal PDF provides all items with numeric success criteria |
| Dataset policy: legal, sufficient size/quality, disclose imbalance and mitigation | UCI only; imbalance noted but not fixed | ❌ | Multi-source curated corpus, balanced 1:1; all limitations disclosed (§4) |
| Ethics: bias, transparency, privacy | Disclaimer only | ⚠️ | Explanations, calibration, privacy-by-design, documented limitations (§16) |
| Report structure and 10–15 pages | Not yet | — | §14 |
| Demo video < 5 min | Not yet | — | §15 |
| Team contribution statement | Not yet | — | §14.4 and `docs/contribution_log.md` |

### 2.2 Problems found with "UCI SMS Spam Collection only"

Verified by downloading and profiling the data:

* **Imbalance:** 4,827 ham vs 747 spam (13.4% spam).
* **Duplicates:** 403 exact duplicate texts; only 5,159 unique messages after whitespace/case normalisation (4,517 ham, 642 spam).
* **Age / concept drift:** collected up to 2012. Modern spam (delivery scams, KYC/bank smishing, crypto, job scams) is missing. In our reference run, a TF-IDF + Naive Bayes model trained only on UCI messages caught **59.2%** of modern spam in the curated test set, versus **97.3%** when trained on the curated corpus (Appendix B).

### 2.3 Decision: curated multi-source corpus (v1)

**Use the UCI SMS Spam Collection plus the Super SMS Dataset (Salman et al., IEEE Access 2024), cleaned of the defects listed in §4.3, de-duplicated across sources, and balanced 1:1 by under-sampling ham.**

Result: **19,152 messages (9,576 spam / 9,576 ham)**, split 70/15/15. The artefact-only classifier scores 53.9% (chance is 50%), and the estimated label-noise rate is about 0.09%. An extra **29,293 unused clean ham messages** are kept for realistic-prevalence evaluation.

### 2.4 Sources considered and rejected

| Source | Why rejected |
|---|---|
| ExAIS SMS Spam Dataset (Onashoga et al., 2015; GitHub) | Label noise: 183 of the user's own *outgoing* replies ("STOP", "1", "ACTIVATE") are labelled SPAM, 42 texts have conflicting labels, carrier balance notices are labelled SPAM while promotions are labelled HAM, CSV columns are misaligned, and there is no licence file. |
| Mishra & Soni SMS Phishing Dataset (Mendeley, 2022) | Ham largely re-uses UCI messages; spam/smishing obtained by OCR from internet images; adds 3-class complexity that is out of scope. |
| "Balanced Dataset for Spam and Smishing Detection using LLMs" (Mendeley, 2025) | **Synthetic** (LLM-generated) messages; not real SMS; would inflate scores and weaken validity. |
| SmishTank (Timko & Rahman, 2024), SpamHunter (Tang et al., 2022) | Spam-only (no ham), small or multilingual, unclear redistribution terms. Can be cited in Related Work. |
| Kaggle re-uploads / Hugging Face commercial samples | Unclear provenance and licensing. |

---

## 3. Scope and requirements

### 3.1 In scope

* English SMS binary classification (spam vs ham).
* Three models: M1 TF-IDF + Multinomial Naive Bayes (baseline), M2 TF-IDF word+char n-grams + Logistic Regression (strong classical), M3 fine-tuned DistilBERT (`distilbert-base-uncased`).
* Post-hoc probability calibration (temperature scaling on validation) and validation-chosen decision thresholds.
* Streamlit web app (single and batch prediction, explanations, model performance, about/limitations).
* Report, figures, tables, demo video, deployment.

### 3.2 Out of scope

* Non-English messages, multi-class (smishing vs promo), URL reputation lookups, sender metadata, on-device/mobile app, user accounts, storing user messages, LLM APIs.

### 3.3 Users and use cases

| Actor | Use case |
|---|---|
| End user | Pastes one suspicious SMS and sees a verdict, probability, risk band, and the words that drove it |
| Analyst / lecturer | Uploads a CSV of messages, downloads predictions; reviews model performance tab |
| Team / grader | Runs the pipeline from scratch and reproduces all reported numbers |

### 3.4 Functional requirements

**Data and ML pipeline**

| ID | Requirement |
|---|---|
| FR-D1 | `scripts/01_download_data.py` downloads both raw sources to `data/raw/`, verifies SHA-256 where pinned (§4.2), and writes `data/raw/MANIFEST.json` (URL, date, SHA-256, size). |
| FR-D2 | `scripts/02_build_dataset.py` reproduces the curated corpus exactly as specified in §5 and writes `data/processed/*` and `data_stats.json`. |
| FR-D3 | `scripts/03_audit_dataset.py` runs the data-quality audit (§5.4) and fails (exit code 1) if any invariant is violated. |
| FR-D4 | `scripts/04_eda.py` generates all EDA figures and tables in §9.4. |
| FR-M1 | `scripts/05_train_classical.py` trains M1 and M2 with 5-fold stratified CV grid search on `train.csv`, refits on full train, and saves `models/tfidf_nb.joblib`, `models/tfidf_lr.joblib`, plus `reports/metrics/cv_*.json`. |
| FR-M2 | `scripts/06_train_distilbert.py` fine-tunes M3 on train, early-stops on val macro-F1, and saves `models/distilbert/` (weights, tokenizer, config, `training_log.json`). |
| FR-M3 | `scripts/07_calibrate.py` fits temperature T and thresholds (τ, τ_low) per model on `val.csv` and writes `models/calibration.json`. |
| FR-M4 | `scripts/08_evaluate.py` runs experiments E1–E8 (E9–E10 optional) on the frozen models and writes all metrics, tables, and figures to `reports/`. |
| FR-M5 | `src/smsspam/inference.py` exposes one `load_predictor(name)` factory and a uniform `Predictor` interface (§10.5) used by both the evaluation scripts and the app. |

**Application**

| ID | Requirement |
|---|---|
| FR-A1 | Text area for one SMS (max 1,000 characters), a **Check message** button, and 6 example buttons (3 spam, 3 ham, taken from the **test** split and anonymised). |
| FR-A2 | Result card: verdict (Likely spam / Suspicious / Likely legitimate), calibrated spam probability (%), model used, latency (ms). |
| FR-A3 | "Why this result?" panel: top words that pushed the score up/down (leave-one-word-out occlusion, §10.6), shown as highlighted text plus a small table. |
| FR-A4 | Model selector in the sidebar: DistilBERT (default), TF-IDF + Logistic Regression, TF-IDF + Naive Bayes. |
| FR-A5 | Batch tab: upload CSV (UTF-8, column `text`, ≤ 1,000 rows, ≤ 2 MB) → table with label/probability/band → **Download results CSV**. |
| FR-A6 | Model performance tab: metrics table (E1), confusion matrix, reliability diagram, and dataset summary loaded from `reports/` files. |
| FR-A7 | About & limitations tab: purpose, data sources with licences and citations, limitations, ethics, privacy statement, team. |
| FR-A8 | Input guards: empty input → info message; > 1,000 chars → truncate with warning; likely non-English (§10.7) → warning banner (still predicts). |
| FR-A9 | Graceful degradation: if DistilBERT cannot load (no internet, Hub down, RAM), the app shows a warning and falls back to M2 automatically. |
| FR-A10 | Persistent disclaimer under every result: "This is an automated estimate and may be wrong. Do not click links or share codes/passwords in unexpected messages; verify with the official sender." |

### 3.5 Non-functional requirements

| ID | Category | Requirement |
|---|---|---|
| NFR-1 | Performance | Single prediction on CPU: M2 p95 ≤ 50 ms; M3 p95 ≤ 500 ms (excluding first model load). Explanation ≤ 2 s on CPU. |
| NFR-2 | Startup | Cold start on Streamlit Cloud ≤ 90 s, including DistilBERT download (cached with `st.cache_resource`). |
| NFR-3 | Memory | App peak RAM ≤ 1.5 GB (Streamlit Community Cloud limit is up to 2.7 GB). |
| NFR-4 | Reliability | No unhandled exception for any input string (including emoji, RTL scripts, 1,000-char spam, whitespace only). Verified by tests. |
| NFR-5 | Reproducibility | `python scripts/run_all.py` reproduces data, classical models, calibration, and evaluation; DistilBERT is reproducible within ±0.3 pp macro-F1 across reruns. |
| NFR-6 | Privacy | The app never writes user input to disk, logs, or analytics. Stated in the UI. |
| NFR-7 | Maintainability | Type hints, docstrings, `ruff` clean, functions ≤ ~50 lines where practical, config in `configs/config.yaml`. |
| NFR-8 | Usability | All UI text in English; readable at 375 px width; meaning never conveyed by colour alone (labels + icons/text). |
| NFR-9 | Portability | Runs on Windows 10/11 and Linux with Python 3.11 (3.12 accepted). |

---

## 4. Data specification

### 4.1 Final corpus at a glance (v1, reference build)

| Item | Value |
|---|---|
| Total messages | **19,152** |
| Class balance | **9,576 spam / 9,576 ham (50.0% / 50.0%)** |
| Splits (stratified, seed 42) | train 13,406 (6,703 / 6,703); val 2,873 (1,436 ham / 1,437 spam); test 2,873 (1,437 ham / 1,436 spam) |
| Composition by source | ham: Super 8,460, UCI 1,116; spam: Super 8,980, UCI 596 |
| Unused clean ham pool (for E2) | 29,293 messages |
| Message length (words) | mean 18.4, median 18, p95 38, p99 56, max 157; spam median 22 vs ham median 10 words |
| Time span | 2012 (UCI) to 2024 (Super, per repository README) |
| Output file SHA-256 | `cda5cbb3fb8bf44619803a549ea229c67f41701489454789878969f085fa19aa` (`sms_spam_curated_v1.csv`) |

Columns of every processed CSV: `text` (cleaned string), `label` (int: 1 = spam, 0 = ham), `source` (`uci_sms_spam_collection` | `super_sms_dataset`), `split` (`train` | `val` | `test`; absent in `unused_ham_pool.csv`).

### 4.2 Sources, provenance, licensing

| # | Dataset | Content | Licence / terms | Download (pinned) | Integrity |
|---|---|---|---|---|---|
| S1 | **SMS Spam Collection** — Almeida & Hidalgo, UCI ML Repository, DOI 10.24432/C5CC84 | 5,574 SMS (4,827 ham / 747 spam), English, up to 2012, tab-separated `label\ttext` | **CC BY 4.0** (attribution required) | Primary: `https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip` (file `SMSSpamCollection`). Fallback mirror: `https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/5339710562cc46c4a32a369d63f1e8751a18a54e/data/sms.tsv` | Must parse to exactly 5,574 rows with 4,827 ham / 747 spam. Mirror SHA-256 `7d039a24a6083ed9ef0f806ebad56bbb976e3aeb8de05669173bfdc4996c239d` |
| S2 | **Super SMS Dataset** — Salman, Ikram & Kaafar, "Investigating Evasive Techniques in SMS Spam Filtering…", IEEE Access 12:24306–24324, 2024, DOI 10.1109/ACCESS.2024.3364671 | 67,010 rows (`SMSes`, `Labels` 0/1), 2012–2024. Aggregates UCI, the NUS SMS corpus, SpamHunter (Twitter-reported spam), volunteers and scam observatories | Publicly released by the authors for research; README requests citation of the paper; **no explicit licence file** → use for non-commercial academic purposes with citation; do **not** redistribute raw or processed files in the repo | `https://raw.githubusercontent.com/smspamresearch/spstudy/a0889f36fb59e23a7f853a6932485d93d18b8afc/Data/super_sms_dataset.csv` | SHA-256 `ba825b468b59fb229cc56c15a6c2d14d593a88051d54ea2f402166bf2ab894d3` (strict). Encoding **cp1252** |

> If the official UCI zip's `SMSSpamCollection` differs byte-wise from the mirror (for example, line endings), that is acceptable as long as the row and class counts match. Log both hashes in `MANIFEST.json`. The cleaned text is whitespace-normalised, so the processed output is expected to be identical.

### 4.3 Data defects found and how they are handled

All counts below come from the reference build (`data_stats_v1.json`, Appendix B).

| # | Defect | Evidence | Risk if ignored | Handling |
|---|---|---|---|---|
| D1 | **Class imbalance** (UCI 13.4% spam) | 747 / 5,574 | Biased model; misleading accuracy | Combine sources, then under-sample ham to 1:1. Keep unused ham for realistic-prevalence evaluation (E2). |
| D2 | **Exact and near duplicates**, including templated spam differing only by numbers/URLs | UCI 403 exact duplicates; 6,335 near duplicates removed across sources | Train/test leakage; inflated scores | Near-duplicate key = lowercase → URL→`URL` → digits→`0` → strip non-alphanumerics → collapse spaces; keep the first (UCI rows win). |
| D3 | **Format-stripping artefact**: lowercased and all digits/punctuation removed upstream, mostly spam | 13,327 spam vs 1,549 ham rows; a formatting-only classifier reaches 85.8% accuracy on raw Super data | Model learns "no punctuation ⇒ spam"; real user input has punctuation | Drop rows with no uppercase letter **and** no digit/punctuation (applies to both classes). |
| D4 | **Lemmatisation artefact** ("This be a reminder…", "KYC have be suspend") almost only in spam | 2,634 spam vs 14 ham rows removed; detector false-positive rate on genuine UCI ham ≈ 0.06% (3 of 4,827) | Grammar shortcut, unrealistic text | Drop rows matched by the lemmatisation detector (Appendix A.1). |
| D5 | **Penn-Treebank tokenisation traces** ("Do n't", "It 's", ``quotes'') only in spam | 8.6% of spam vs 0% of ham before the fix | Shortcut feature | **Detokenise** every row (rows are kept). |
| D6 | **Anonymisation masks** differ by class: `XXXX` in 27.5% of spam / 0% of ham; `<#>` in 5.4% of ham / 0.2% of spam | Measured on the curated set | Model learns "XXXX ⇒ spam"; real messages contain real digits | Shared normaliser maps masks **and** real numbers to the same placeholders (`[NUM]`, `[PHONE]`, `[MONEY]`, …) for all models, at train and inference time (§6). After normalisation a mask-only classifier scores 50.3%. |
| D7 | **Mojibake** (`Ã‚Â£`, `WeÂ've`) | Hundreds of rows | Noise tokens | `ftfy.fix_text` + targeted cleanup + NFKC normalisation. |
| D8 | **Conflicting labels** for the same near-duplicate key | 11 rows | Label noise | Drop all rows of conflicting keys. |
| D9 | **Very short / empty** messages | 266 rows (< 5 chars or < 2 words) | Unlearnable noise | Drop. |
| D10 | **Residual label noise** (for example, OTP or ticket notices labelled spam) | Cross-validated confident-disagreement estimate: 17 rows (0.09%) | Small error floor | Keep labels as published (no silent relabelling). Run the audit in E7 and report the estimated noise. |
| D11 | **Residual case difference**: 8.1% of ham vs 0.3% of spam are fully lowercase | Measured | Minor shortcut | Neutralised by design: M1/M2 lowercase their input and M3 is *uncased*. Verified by the robustness test (E6, lowercase perturbation). |
| D12 | **Language / region mix** (UK, Singapore, India, US, Australia, and others; Singlish/texting style in ham) | Inspection | Domain shift | English-variety diversity is kept on purpose (automatic language ID wrongly flagged Singlish ham as foreign). Only messages with > 10% non-Latin letters are dropped (1 row). Disclosed as a limitation. |

**Bias check built into the plan:** source must not predict label. Reference result: a source-only classifier reaches 52.7% (chance 50%). Re-checked by the audit script.

### 4.4 Balancing rationale (for the report)

* Balanced **training** data prevents majority-class bias without synthetic oversampling (no SMOTE on text, no LLM generation).
* Balanced **val/test** make macro-F1 and per-class metrics easy to read. Because a 50/50 test set inflates precision relative to the real world, **E2 re-evaluates at a realistic 13.4% spam prevalence** using the held-out clean ham pool. This is the honest number to quote for deployment.
* Ham was under-sampled (not spam over-sampled) because clean ham is abundant (38,869 after cleaning) and spam is the limiting class (9,576).

---

## 5. Data pipeline implementation

### 5.1 Steps (must match the reference implementation in Appendix A.2 exactly)

1. Load S1 (tab-separated, `quoting=3`, no header) and assert 5,574 rows. Map `spam→1`, `ham→0`, `source="uci_sms_spam_collection"`.
2. Load S2 with `encoding="cp1252", encoding_errors="replace"`, drop rows with missing `SMSes`/`Labels` (3 rows), `source="super_sms_dataset"`.
3. Concatenate **UCI first** (so UCI copies win de-duplication).
4. `clean_text`: `ftfy.fix_text` → drop U+FFFD → remove stray `Â` → NFKC → curly quotes/dashes/NBSP → ASCII equivalents → collapse whitespace → PTB detokenisation → collapse whitespace.
5. Filters in this order (record removed counts per class): `too_short` → `format_stripped_artifact` → `lemmatisation_artifact` → `non_latin_script` → `conflicting_labels` (by near-duplicate key) → `near_duplicates` (keep first) → `empty_key`.
6. Balance: keep all spam; sample the same number of ham with `random_state=42`; shuffle with `random_state=42`.
7. Split: `train_test_split(test_size=0.30, stratify=label, random_state=42)`, then split the 30% in half (val/test), stratified, `random_state=42`.
8. Write `train.csv`, `val.csv`, `test.csv`, `sms_spam_curated_v1.csv`, `unused_ham_pool.csv`, `data_stats.json` (raw counts, per-step removals, after-cleaning counts, final counts, input and output SHA-256).

### 5.2 Expected outputs (acceptance values)

| Check | Expected |
|---|---|
| After-cleaning counts | ham 38,869 (Super 34,398 + UCI 4,471); spam 9,576 (Super 8,980 + UCI 596) |
| Final total | 19,152 (exactly 50/50) |
| Split sizes | 13,406 / 2,873 / 2,873 |
| Removed per step (ham/spam) | too_short 264/2 · format_stripped 1,549/13,327 · lemmatisation 14/2,634 · non_latin 1/0 · conflicting 6/5 · near_duplicates 4,954/1,381 |
| Output SHA-256 | `cda5cbb3…fa19aa` with the pinned inputs and the same library behaviour |

If the hash differs but the counts match within ±0.5% and every audit invariant (§5.4) passes, proceed and log the difference in `docs/DECISIONS.md`. Typical causes: a different `ftfy`/pandas version. If counts differ by more than 0.5%, stop and debug.

### 5.3 Library requirements for the pipeline

`pandas`, `scikit-learn`, `ftfy` (≥ 6.0). No language-ID library (deliberately; see D12).

### 5.4 Data-quality audit (automated; `scripts/03_audit_dataset.py` + `tests/test_data_quality.py`)

All must pass:

1. Exactly 50/50 class balance overall and in each split (±1 row for odd splits).
2. No near-duplicate key appears in more than one split; no key is duplicated within the corpus.
3. Zero rows match the format-stripped, lemmatisation-P1, PTB-contraction, PTB-quote, or mojibake (`Ã`, `Â`, `â€`) patterns.
4. Artefact-only classifier (features: lower_only, no_punct, stripped_fmt, ptb_contraction, ptb_quotes, lemma_be_det, lemma_have_be, mojibake; logistic regression; 5-fold CV accuracy) ≤ **0.56**.
5. After `normalize()`, a classifier using only "raw mask present" features (`X{2,}` or lone `X`, `<#>`) scores ≤ **0.55**.
6. Source-only accuracy ≤ **0.56**.
7. No empty `text`; labels ∈ {0, 1}; `split` ∈ {train, val, test}.
8. `unused_ham_pool.csv` shares no near-duplicate key with train/val/test.

Write the audit results to `reports/metrics/data_audit.json`.

---

## 6. Text normalisation and NLP preprocessing

### 6.1 Shared normaliser (mandatory for all models, train and inference)

Implemented in `src/smsspam/preprocessing.py` exactly as in Appendix A.1 (`normalize(text) -> str`). Order of operations:

1. NFKC.
2. Dataset masks: `<URL>`→`[URL]`, `<EMAIL>`→`[EMAIL]`, `<#>`/`<DECIMAL>`/`<TIME>`→`[NUM]`.
3. E-mails → `[EMAIL]`; URLs (scheme, `www.`, or bare domains with common TLDs) → `[URL]`.
4. Money (currency symbol or code + amount, including `X` masks) → `[MONEY]`.
5. Phone-like sequences (≥ 7 digits/`X`, optional `+`, spaces, dashes, brackets) → `[PHONE]`.
6. Other numbers (digits mixed with `X` masks, decimals, times, dates), `XX+` masks, and lone `X` → `[NUM]`; collapse repeated `[NUM]`.
7. Collapse whitespace.

Properties to unit-test: idempotent (`normalize(normalize(x)) == normalize(x)`); alphanumeric codes such as `BWTY0803`, `4G`, `1GB` stay unchanged; `X-ray` is unchanged; `None`/empty → `""`.

**Why:** it removes the D6 artefact and makes real user input (real digits/links) look like training data (masked digits/links). It is also a classic, explainable NLP step that is easy to justify in the report.

### 6.2 Model-specific preprocessing

| Model | After `normalize()` |
|---|---|
| M1, M2 | `TfidfVectorizer(lowercase=True, sublinear_tf=True)` with the default token pattern (placeholders become the tokens `url`, `email`, `phone`, `money`, `num`; this was the configuration verified in Appendix B). No stop-word removal or lemmatisation in the final models (decided by E4). |
| M3 | DistilBERT uncased tokenizer, `max_length=128`, truncation on, dynamic padding. 128 sub-word tokens covers > 99.5% of messages (p99 = 56 words). Placeholders are tokenised as ordinary text (`[`, `url`, `]`); no vocabulary changes. |

### 6.3 Classical NLP components (NLTK and spaCy) used in E4 and EDA

* **NLTK:** `nltk.corpus.stopwords` (English) for the stop-word-removal ablation; `nltk.FreqDist` for top-token EDA.
* **spaCy:** `en_core_web_sm` with `disable=["parser","ner"]` for the lemmatisation ablation, run with `nlp.pipe(batch_size=512)`.
* Both are listed in `requirements-train.txt` only (not needed by the app).

---

## 7. Models and training

### 7.1 M1 — Baseline: TF-IDF + Multinomial Naive Bayes

* Pipeline: `normalize` → `TfidfVectorizer` → `MultinomialNB`.
* Grid (5-fold `StratifiedKFold(shuffle=True, random_state=42)`, scoring `f1_macro`): `ngram_range ∈ {(1,1),(1,2)}`, `min_df ∈ {1,2}`, `alpha ∈ {0.01, 0.05, 0.1, 0.5, 1.0}`.
* Expected test macro-F1 ≈ 0.97 (reference 0.972).

### 7.2 M2 — Strong classical: TF-IDF word + character n-grams + Logistic Regression

* Features: `FeatureUnion` of word TF-IDF (`ngram_range=(1,2)`, `min_df=2`) and `char_wb` TF-IDF (`ngram_range=(3,5)`, `min_df=3`, `max_features=200_000`).
* Classifier: `LogisticRegression(max_iter=3000, solver="liblinear")`; grid `C ∈ {1, 4, 10, 30}`.
* Explainable through coefficients (used in EDA figure F4); robust to obfuscation thanks to character n-grams.
* Expected test macro-F1 ≈ 0.98 (reference 0.984).
* Save with `joblib.dump(..., compress=3)`; the file must be < 50 MB (if not, lower `max_features` to 100,000 and log it).

### 7.3 M3 — Transformer: fine-tuned DistilBERT

| Setting | Value |
|---|---|
| Checkpoint | `distilbert-base-uncased` (Apache-2.0) |
| Head | `AutoModelForSequenceClassification`, `num_labels=2`, `id2label={0:"ham",1:"spam"}` |
| Input | `normalize(text)`, `max_length=128`, dynamic padding (`DataCollatorWithPadding`) |
| Optimiser / schedule | AdamW, lr `2e-5`, weight decay `0.01`, linear decay, warmup ratio `0.1` |
| Batch size | 32 (RTX 3050 6 GB with fp16). On OOM: 16 with `gradient_accumulation_steps=2` |
| Epochs | max 4; early stopping patience 1 on **val macro-F1**; `load_best_model_at_end=True` |
| Precision | `fp16=True` when CUDA is available |
| Seed | 42 (`transformers.set_seed`) |
| Evaluation / save | every epoch; keep the best checkpoint only (`save_total_limit=1`) |
| Logging | `training_log.json` with loss and val metrics per epoch, wall-clock time, GPU name |

* Use the `Trainer` API. Check the installed `transformers` version: newer versions use `eval_strategy` (older ones used `evaluation_strategy`). Use whichever the installed version accepts and log it.
* Expected training time on an RTX 3050: about 10–15 minutes in total. Expected test macro-F1 ≥ 0.985.
* **Colab fallback:** `notebooks/colab_train_distilbert.ipynb` runs the same `src/` code (clone repo → install → run script); data is rebuilt inside Colab with the same scripts.
* Optional E10: train with seeds {13, 42, 2026} and report mean ± std; the app uses seed 42.

### 7.4 Model artefacts

| Path | Committed to Git? |
|---|---|
| `models/tfidf_nb.joblib`, `models/tfidf_lr.joblib` | Yes (small) |
| `models/calibration.json` | Yes |
| `models/distilbert/` | **No** (gitignored) → uploaded to Hugging Face Hub (§13.1) |
| `models/MODEL_CARD.md` | Yes (also uploaded as the Hub README) |

---

## 8. Calibration, thresholds, and risk bands

### 8.1 Temperature scaling (all three models, fitted on `val.csv`)

* Let `p` = the model's raw P(spam), clipped to [1e-6, 1 − 1e-6]; `z = logit(p)`.
* Calibrated probability `p_cal = sigmoid(z / T)`; T > 0 found by minimising validation negative log-likelihood (`scipy.optimize.minimize_scalar`, bounds [0.05, 20]).
* For M3, `z` is the logit difference `logit_spam − logit_ham`, which is equivalent.
* Report ECE (15 equal-width bins), Brier score, and NLL before and after (E5).
* **Wording rule:** the UI may call the score a "calibrated spam probability" only if E5 shows test ECE ≤ 0.03 for that model; otherwise call it a "spam score".

### 8.2 Decision threshold τ and suspicious band

On `val.csv` with calibrated probabilities:

* **τ** = the threshold in a 0.01-step grid over [0.05, 0.95] that maximises macro-F1 subject to ham false-positive rate ≤ 1.0%. If no threshold meets the constraint, take the one with the lowest FPR and log it.
* **τ_low** = the largest threshold ≤ τ at which val spam recall ≥ 0.99. If τ_low ≥ τ, there is no suspicious band (τ_low = τ).
* Bands: `p ≥ τ` → **Likely spam**; `τ_low ≤ p < τ` → **Suspicious — verify before acting**; `p < τ_low` → **Likely legitimate**.
* Stored in `models/calibration.json`:

```json
{
  "distilbert": {"temperature": 1.37, "threshold": 0.62, "threshold_low": 0.31, "val_ece_before": 0.021, "val_ece_after": 0.008},
  "tfidf_lr":   {"temperature": 0.91, "threshold": 0.55, "threshold_low": 0.28, "...": "..."},
  "tfidf_nb":   {"temperature": 2.40, "threshold": 0.58, "threshold_low": 0.22, "...": "..."}
}
```

(The values above are illustrative placeholders; the real ones are produced by the script.)

---

## 9. Evaluation protocol and experiments

### 9.1 Metrics (always on the untouched test set unless stated otherwise)

Primary: **macro-F1**. Also: accuracy, spam precision/recall/F1, ham precision/recall/F1, **ham false-positive rate (FPR)**, ROC-AUC, PR-AUC (average precision), confusion matrix, ECE, Brier. Report **95% bootstrap confidence intervals** (1,000 resamples, seed 42) for macro-F1, spam recall, and FPR. Pairwise **McNemar test** (exact binomial on discordant pairs) between M1–M2, M1–M3, and M2–M3.

### 9.2 Success criteria (stated in the proposal)

| ID | Criterion | Target |
|---|---|---|
| SC1 | Best model macro-F1 on the balanced test set | ≥ 0.98 |
| SC2 | Ham FPR of the deployed model at τ (test) | ≤ 1.0% |
| SC3 | Spam precision at realistic 13.4% prevalence (E2) | ≥ 0.93 |
| SC4 | DistilBERT vs NB baseline | Significantly better (McNemar p < 0.05) |
| SC5 | Calibration of the deployed model (test ECE after temperature scaling) | ≤ 0.03 |
| SC6 | App latency (CPU, single message, M3) | p95 ≤ 0.5 s |
| SC7 | All automated tests pass; app runs locally and at a public URL | 100% |

If a criterion is missed, do **not** tune on test. Report it honestly in Discussion with a cause analysis.

### 9.3 Experiments

| ID | Name | Design | Outputs |
|---|---|---|---|
| **E1** | Main comparison | M1, M2, M3 at their calibrated τ on test; all metrics in §9.1 + CIs + McNemar | `tables/T4_main_results.csv`, `figures/F5_confusion_*.png`, `figures/F6_roc_pr.png` |
| **E2** | Realistic prevalence | Test spam (1,436) + test ham (1,437) + ham sampled from `unused_ham_pool.csv` (seed 42) so that spam = 13.4% (≈ 9,280 ham in total). Report precision, recall, F1, FPR, and number of false alarms per 1,000 ham | `tables/T5_natural_prevalence.csv` |
| **E3** | Concept drift / value of curation | Train M1 and M2 on (a) **UCI-only** rows of train and (b) the full curated train. Evaluate on test, split by source (UCI part vs Super part). Reference: UCI-only M1 spam recall on the Super part 0.592 vs curated 0.973 | `tables/T6_drift.csv`, `figures/F8_drift.png` |
| **E4** | Preprocessing ablation (M1 and M2, 5-fold CV on train) | (i) raw lowercase; (ii) + `normalize`; (iii) + NLTK stop-word removal; (iv) + spaCy lemmatisation. Mean ± std macro-F1 | `tables/T7_ablation.csv` |
| **E5** | Calibration | Reliability diagrams before/after; ECE, Brier, NLL on test for each model | `figures/F7_reliability.png`, `tables/T8_calibration.csv` |
| **E6** | Robustness / invariance | Apply to test messages (seeded): lowercase; strip punctuation; leetspeak (a→4, e→3, i→1, o→0, s→5 on 30% of eligible characters); character spacing of one random word ("F R E E"); swap URL domain; append benign sentence "See you soon." For each model: recall change on spam, FPR change on ham, flip rate | `tables/T9_robustness.csv`, `figures/F9_robustness.png` |
| **E7** | Error analysis | All test FP/FN of the deployed model, each manually tagged into: transactional/OTP looks like promo; short/ambiguous; conversational spam (romance/job scams); obfuscated; non-English or code-mixed; probable label noise. Counts + 2 anonymised examples per category. Also review the 17 cross-validated suspect labels (D10) | `tables/T10_error_taxonomy.csv`, `reports/error_analysis.md` |
| **E8** | Efficiency | Model size on disk, CPU latency (mean/p95 over 200 test messages, batch size 1, after warm-up), training time | `tables/T11_efficiency.csv` |
| E9 *(optional)* | Balanced vs imbalanced training | M2 trained on all 48,445 cleaned messages (minus val/test keys) with `class_weight="balanced"` vs balanced train; evaluate on test and E2 set | `tables/T12_balance.csv` |
| E10 *(optional)* | Seed variance for M3 | 3 seeds, mean ± std | `tables/T13_seeds.csv` |

### 9.4 EDA outputs (`scripts/04_eda.py`)

| ID | Figure / table |
|---|---|
| F1 | Class distribution: raw UCI vs raw Super vs curated (bar) |
| F2 | Curation funnel: rows removed per step, by class (horizontal bar) |
| F3 | Message length (words) by class (histogram/violin) |
| F4 | Top 20 spam-indicative and ham-indicative n-grams (M2 coefficients, after normalisation) |
| F10 | DistilBERT training curves (train loss, val macro-F1 per epoch) |
| T1 | Dataset sources and licences (from §4.2) |
| T2 | Curation step table (from `data_stats.json`) |
| T3 | Hyperparameters and selected values (from CV results) |

Figure style: matplotlib only, 300 dpi PNG, readable in grayscale (use hatching/markers, not colour alone), consistent fonts, titles and axis labels in English.

### 9.5 Reference sanity ranges (a deviation means: investigate)

| Model | Expected test macro-F1 | Investigate if |
|---|---|---|
| M1 | 0.96–0.98 | < 0.95 or > 0.99 |
| M2 | 0.975–0.99 | < 0.965 |
| M3 | 0.985–0.995 | < 0.975, or > 0.998 (possible leakage) |

---

## 10. Application specification (Streamlit)

### 10.1 Layout

* **Sidebar:** app title "SMS Spam Checker", model selector (FR-A4), short model description, the deployed model's threshold and test macro-F1 (read from `reports/metrics/`), link to the GitHub repo, privacy note.
* **Tabs:** `Check a message` · `Batch check` · `Model performance` · `About & limitations`.

### 10.2 "Check a message" tab

1. Example buttons row (3 spam, 3 ham). Clicking fills the text area.
2. `st.text_area` (placeholder: "Paste an SMS here…", `max_chars=1000`), character counter.
3. **Check message** primary button (disabled while empty).
4. Result card:
   * Big verdict with icon and text: 🚫 *Likely spam* / ⚠️ *Suspicious — verify before acting* / ✅ *Likely legitimate* (the icon is never the only signal).
   * "Calibrated spam probability: 97.4%" (or "spam score" per §8.1 rule), progress bar, model name, latency.
   * Disclaimer (FR-A10).
5. Expander **Why this result?** (FR-A3): the message with the top-k influential words highlighted (k ≤ 8; spam-pushing vs ham-pushing words distinguished by a label, not colour alone) and a table `word | effect on spam probability (pp)`.
6. Session history of the last 5 checks held only in `st.session_state` (cleared on refresh, never written to disk).

### 10.3 "Batch check" tab

Upload a CSV with a `text` column (validate encoding, column name, row limit 1,000, size 2 MB; show clear errors). Show a progress bar, a results table (`text`, `label`, `spam_probability`, `risk_band`), a summary count per band, and a **Download results CSV** button. A sample CSV (10 test rows) can be downloaded from the tab.

### 10.4 "Model performance" and "About & limitations" tabs

* Performance: T4 main results table, confusion matrix and reliability figure for the selected model, E2 realistic-prevalence numbers, and a dataset summary (T1/T2), all loaded from `reports/`. **No retraining in the app.**
* About: purpose, how it works (3-step diagram in text), data sources with licences/citations, limitations (§16.3), privacy statement, team names, course code.

### 10.5 Inference interface (`src/smsspam/inference.py`)

```python
@dataclass(frozen=True)
class Prediction:
    label: Literal["spam", "ham"]          # thresholded at tau
    spam_probability: float                # calibrated, 0..1
    risk_band: Literal["likely_spam", "suspicious", "likely_ham"]
    model_name: str
    latency_ms: float

class Predictor(Protocol):
    name: str
    def predict_proba(self, texts: Sequence[str]) -> np.ndarray: ...   # calibrated P(spam), shape (n,)
    def predict(self, texts: Sequence[str]) -> list[Prediction]: ...

def load_predictor(name: Literal["distilbert", "tfidf_lr", "tfidf_nb"]) -> Predictor: ...
```

* Every predictor applies `normalize()` internally. Callers pass raw text.
* DistilBERT loads from the local `models/distilbert/` if present, else from the Hub repo id in `configs/config.yaml` (`hf_model_id`), on CPU in the app, with `torch.inference_mode()` and batch size 32 for batches.
* The app wraps `load_predictor` in `@st.cache_resource`.

### 10.6 Explanation method (model-agnostic leave-one-word-out occlusion)

For a message with words w₁…wₙ (n ≤ 60; beyond that, explain the first 60 words): for each i, delete wᵢ and compute Δᵢ = p(full) − p(without wᵢ). Do all n variants in **one batch**. Show the top-k |Δᵢ|. Cache by `(model, text)`. Document in the report as an occlusion-based local explanation and state its limits (word interactions are ignored).

### 10.7 Non-English warning heuristic (no extra dependency)

Compute the share of alphabetic tokens (length ≥ 3, after `normalize`, lowercased) that are **in the M2 word vocabulary**. If < 40% and the message has ≥ 5 such tokens, show: "This message may not be in English. The model was trained on English SMS, so this result is less reliable." Still predict.

### 10.8 Theming and assets

Use `.streamlit/config.toml` with a neutral light theme. No external CDNs or trackers. App file: `app/streamlit_app.py`, which inserts `Path(__file__).resolve().parents[1] / "src"` into `sys.path` before importing `smsspam`.

---

## 11. Repository, environment, configuration

### 11.1 Repository layout

```
sms-spam-detection/
├── README.md                      # quick start, results summary, live URL, citations
├── LICENSE                        # MIT (code only)
├── DATA_LICENSES.md               # S1 CC BY 4.0 attribution; S2 citation + academic-use note
├── requirements.txt               # APP RUNTIME (Streamlit Cloud reads this; CPU torch)
├── requirements-train.txt         # training/eval/test extras (no torch line; torch installed separately)
├── pyproject.toml                 # ruff + pytest config (package metadata optional)
├── .gitignore                     # data/, models/distilbert/, .venv/, __pycache__/, *.ipynb_checkpoints
├── .streamlit/config.toml
├── configs/config.yaml            # paths, seed, URLs, hashes, hyperparameters, hf_model_id
├── app/streamlit_app.py
├── src/smsspam/
│   ├── __init__.py
│   ├── config.py                  # loads config.yaml → dataclass
│   ├── preprocessing.py           # normalize() (Appendix A.1)
│   ├── data/download.py           # FR-D1
│   ├── data/build.py              # FR-D2 (Appendix A.2 logic)
│   ├── data/audit.py              # FR-D3
│   ├── models/classical.py        # M1, M2 pipelines + grid search
│   ├── models/transformer.py      # M3 fine-tuning + loading
│   ├── calibration.py             # temperature scaling, thresholds, ECE
│   ├── inference.py               # Prediction, Predictor, load_predictor
│   ├── explain.py                 # occlusion explanations
│   ├── evaluation/metrics.py      # metrics, bootstrap CI, McNemar
│   ├── evaluation/experiments.py  # E1–E10
│   ├── evaluation/perturb.py      # E6 perturbations
│   └── viz.py                     # figure helpers (grayscale-safe)
├── scripts/
│   ├── 01_download_data.py  02_build_dataset.py  03_audit_dataset.py  04_eda.py
│   ├── 05_train_classical.py  06_train_distilbert.py  07_calibrate.py  08_evaluate.py
│   ├── 09_push_to_hub.py      # uploads models/distilbert + MODEL_CARD.md
│   └── run_all.py             # 01→05, 07, 08 (06 only with --with-transformer)
├── notebooks/
│   ├── 01_eda_walkthrough.ipynb   # narrative EDA for report screenshots (calls src/)
│   └── colab_train_distilbert.ipynb
├── data/                          # gitignored; README.md inside explains how to rebuild
├── models/                        # tfidf_*.joblib, calibration.json, MODEL_CARD.md (distilbert/ ignored)
├── reports/
│   ├── figures/  tables/  metrics/
│   └── error_analysis.md
├── tests/                         # see §12
└── docs/
    ├── DECISIONS.md   DATA_CARD.md   contribution_log.md   demo_script.md
```

### 11.2 Environment setup (local, RTX 3050)

1. Install Python **3.11** (64-bit). Create the environment: `python -m venv .venv` and activate it (Windows: `.venv\Scripts\activate`; Linux/macOS: `source .venv/bin/activate`).
2. Install the **CUDA build of PyTorch** using the selector at pytorch.org/get-started (choose the CUDA version supported by the installed NVIDIA driver; check with `nvidia-smi`).
3. `pip install -r requirements-train.txt`
4. `python -m spacy download en_core_web_sm` and `python -c "import nltk; nltk.download('stopwords')"`
5. Verify: `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"` → `True NVIDIA GeForce RTX 3050 …`

**`requirements-train.txt`** (lower bounds; after a successful full run, pin the exact installed versions and also save `pip freeze > requirements-lock.txt`):

```
pandas>=2.1
numpy>=1.26
scikit-learn>=1.4
scipy>=1.11
ftfy>=6.1
joblib>=1.3
transformers>=4.44
accelerate>=0.33
datasets>=2.20
huggingface_hub>=0.24
safetensors>=0.4
nltk>=3.8
spacy>=3.7
matplotlib>=3.8
pyyaml>=6.0
streamlit>=1.36
pytest>=8.0
ruff>=0.5
jupyter
```

**`requirements.txt`** (app runtime for Streamlit Cloud; pin exact versions verified locally):

```
--extra-index-url https://download.pytorch.org/whl/cpu
torch
transformers
huggingface_hub
safetensors
scikit-learn
pandas
numpy
scipy
joblib
pyyaml
streamlit
```

> The `scikit-learn` version in `requirements.txt` **must equal** the version used to train and save the joblib files; otherwise unpickling may warn or fail. Pin it exactly.

### 11.3 `configs/config.yaml` (keys required)

```yaml
seed: 42
paths: {raw: data/raw, processed: data/processed, models: models, reports: reports}
sources:
  uci: {url: "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip",
        fallback_url: "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/5339710562cc46c4a32a369d63f1e8751a18a54e/data/sms.tsv",
        fallback_sha256: "7d039a24a6083ed9ef0f806ebad56bbb976e3aeb8de05669173bfdc4996c239d",
        expected_rows: 5574}
  super: {url: "https://raw.githubusercontent.com/smspamresearch/spstudy/a0889f36fb59e23a7f853a6932485d93d18b8afc/Data/super_sms_dataset.csv",
          sha256: "ba825b468b59fb229cc56c15a6c2d14d593a88051d54ea2f402166bf2ab894d3", encoding: cp1252}
split: {val: 0.15, test: 0.15}
natural_prevalence: 0.134
classical: {cv_folds: 5, nb_alpha: [0.01, 0.05, 0.1, 0.5, 1.0], lr_C: [1, 4, 10, 30], char_max_features: 200000}
distilbert: {checkpoint: distilbert-base-uncased, max_length: 128, lr: 2.0e-5, batch_size: 32,
             epochs: 4, weight_decay: 0.01, warmup_ratio: 0.1, patience: 1, fp16: true}
calibration: {max_ham_fpr: 0.01, min_spam_recall_low: 0.99, ece_bins: 15}
app: {default_model: distilbert, max_chars: 1000, batch_max_rows: 1000, explain_max_words: 60, explain_top_k: 8}
hf_model_id: "<hf-username>/distilbert-sms-spam-curated"
```

---

## 12. Testing and quality assurance

### 12.1 Automated tests (`pytest -q`)

| File | What it checks |
|---|---|
| `tests/test_preprocessing.py` | Placeholder mapping for URL, e-mail, phone, money, numbers, masks; idempotency; codes untouched (`BWTY0803`, `4G`); `None`/empty; emoji and non-Latin input do not crash |
| `tests/test_build_dataset.py` | On a tiny synthetic fixture (≈ 30 rows covering each defect): every filter removes exactly the intended rows; PTB detokenisation; conflicting keys dropped; output is balanced; splits are disjoint; the run is deterministic (same hash twice) |
| `tests/test_data_quality.py` | Runs the §5.4 audit on the real processed data; `pytest.skip` if `data/processed/` is absent |
| `tests/test_calibration.py` | Temperature scaling recovers T on synthetic logits (±10%); ECE of perfectly calibrated synthetic data < 0.02; threshold search respects the FPR constraint |
| `tests/test_inference.py` | `Predictor` contract: probabilities in [0,1], length matches input, deterministic, batch equals single, bands consistent with τ/τ_low; M3 tests marked `@pytest.mark.slow` and skipped when the model is absent |
| `tests/test_explain.py` | Occlusion returns ≤ k items; removing a highly spammy word such as "[URL]" from an obvious spam lowers the probability |
| `tests/test_app.py` | `streamlit.testing.v1.AppTest`: app loads without exception; entering an example and clicking **Check message** renders a verdict; empty input shows info; forced DistilBERT load failure triggers the M2 fallback warning |
| `tests/test_metrics.py` | Bootstrap CI contains the point estimate; McNemar on a known table matches the hand-computed p-value |

### 12.2 Continuous integration (recommended)

`.github/workflows/ci.yml`: Ubuntu, Python 3.11, install CPU torch + `requirements-train.txt`, run `ruff check .` and `pytest -q -m "not slow"`. The classical models are committed, so the app smoke test runs in CI.

### 12.3 Manual QA checklist (before the video and before submission)

* [ ] Fresh clone → follow README → app runs locally in ≤ 10 minutes of setup.
* [ ] Public URL loads in an incognito window; first prediction works; app wakes from sleep.
* [ ] All 6 examples behave sensibly; the batch sample CSV round-trips.
* [ ] Phone-width layout is readable (browser dev tools at 375 px).
* [ ] Inputs tried: empty, whitespace, 1,000-char spam, emoji-only, Indonesian sentence (warning shown), a URL only, a real recent scam text typed by a team member (anonymised).

---

## 13. Deployment

### 13.1 Hugging Face Hub (DistilBERT weights)

1. Create a free account and a **model** repository named `distilbert-sms-spam-curated`.
2. `huggingface-cli login` (or set `HF_TOKEN`), then `python scripts/09_push_to_hub.py` uploads `models/distilbert/` and `models/MODEL_CARD.md` as `README.md`. The model card must include the intended use, data sources and licences, metrics from E1/E2/E5, limitations, and the "academic, non-commercial use" note for S2.
3. Set `hf_model_id` in `configs/config.yaml`.

### 13.2 Streamlit Community Cloud

1. Push the repo to GitHub (public, or private with access granted to the deployer).
2. At share.streamlit.io: **Create app** → repo/branch → main file `app/streamlit_app.py` → **Advanced settings → Python 3.11** (or 3.12). The platform default may be newer, and torch wheels might not exist for it yet.
3. Deploy. If the build fails on torch, check the Python version first, then the CPU index line in `requirements.txt`.
4. Smoke test the public URL. Put the URL in README, report, and video.
5. Free apps sleep after 12 hours without traffic. Open the URL shortly before grading, and mention in the README that the first load may take about a minute.

### 13.3 Fallback if cloud deployment is impossible

The app still runs locally (`streamlit run app/streamlit_app.py`). Record a full screen capture of the local app for the video. If DistilBERT cannot load in the cloud, the app runs M2 and shows the fallback notice (FR-A9). Document this in `DECISIONS.md` and the report.

---

## 14. Written report specification (10–15 pages, academic style, IEEE references)

### 14.1 Structure and page budget

| Section | Pages | Must contain (all numbers from `reports/`) |
|---|---|---|
| Title page | 1 | Project title, team members + student IDs, course code COMP6885001, semester/academic year |
| Abstract | ≤ 0.3 | Problem, data (19,152 balanced), models, best macro-F1, realistic-prevalence precision, app |
| 1 Introduction | 1 | Background (SMS spam and smishing harm), problem statement, objectives O1–O5, significance, contributions |
| 2 Related Work | 1.5 | UCI collection (Almeida et al.) and its validity; classical ML on SMS spam; Super SMS Dataset and evasion/drift findings (Salman et al.); SpamHunter (Tang et al.); SmishTank (Timko & Rahman); transformers (BERT, DistilBERT); recent NLP + ML study (Abdel-Jaber 2025). End with the gap: artefact-aware curation + calibrated deployment |
| 3 Methodology | 3 | 3.1 Data sources & licences (T1); 3.2 curation pipeline & defects D1–D12 (F2, T2); 3.3 balancing & splits; 3.4 normalisation; 3.5 models M1–M3 (T3); 3.6 training setup; 3.7 calibration & thresholds; 3.8 metrics & statistics |
| 4 Implementation & Results | 4 | 4.1 System architecture (pipeline + app diagram); 4.2 EDA (F1, F3, F4); 4.3 main results E1 (T4, F5, F6); 4.4 realistic prevalence E2 (T5); 4.5 drift E3 (F8); 4.6 ablation E4 (T7); 4.7 calibration E5 (F7); 4.8 robustness E6 (F9); 4.9 efficiency E8 (T11); 4.10 application screenshots |
| 5 Discussion & Limitations | 1.5 | Interpretation, error taxonomy E7 (T10), trade-offs (accuracy vs latency vs explainability), threats to validity (label definitions, residual artefacts, English-only, 2024 cut-off, 50/50 test inflation), ethical considerations |
| 6 Conclusion & Future Work | 0.5 | Summary vs success criteria SC1–SC7; future work: multilingual (Indonesian) SMS, smishing sub-classes, URL features, adversarial training, on-device model |
| References | 0.5–1 | IEEE style (Appendix C) |
| Appendix | — | Team contribution statement, app screenshots, key code snippets, hyperparameter grids, data card |

### 14.2 Writing rules

* Past tense for what was done; present tense for results shown in tables.
* Every figure/table is numbered, captioned, and referenced in the text.
* Report precision/recall/F1 with 3 decimals and CIs in brackets.
* Say "calibrated probability" only when SC5 is met.
* Disclose all data limitations and mitigations (guideline Dataset Policy).

### 14.3 Required figures/tables checklist

F1, F2, F3, F4, F5, F6, F7, F8, F9, F10, app screenshots ×3; T1–T11.

### 14.4 Team contribution statement (template; fill in real names)

| Member | Student ID | Main contributions | Shared tasks |
|---|---|---|---|
| [Name 1] | [NIM] | Data pipeline, audit, EDA | Report §3, video narration |
| [Name 2] | [NIM] | Classical models, calibration, evaluation | Report §4 |
| [Name 3] | [NIM] | DistilBERT, deployment | Report §2, §5 |
| [Name 4] | [NIM] | Streamlit app, tests, UX | Video editing |

Coordination: GitHub (feature branches + pull requests), GitHub Issues/Projects board, weekly log in `docs/contribution_log.md` (date, member, task, hours, commit links).

---

## 15. Demo video specification (< 5:00; target 4:30; MP4 1080p or YouTube unlisted)

| Time | Segment | Content |
|---|---|---|
| 0:00–0:25 | Hook & problem | Real-looking (fictional, anonymised) scam SMS on screen; the problem; the goal |
| 0:25–1:10 | Data | Sources and licences; the three artefacts found (show F2 funnel); balanced 19,152 messages |
| 1:10–1:50 | Method | Pipeline diagram: normalise → M1/M2/M3 → calibrate → threshold → app |
| 1:50–2:50 | Results | T4 (best macro-F1 with CI), E2 realistic precision, E3 drift bar chart, one robustness finding |
| 2:50–4:05 | Live demo | Public URL: spam example → verdict + explanation; ham example; tricky example (suspicious band); batch CSV; performance tab |
| 4:05–4:30 | Limitations & ethics, close | English-only, not a security guarantee, privacy; thank you + repo URL |

Production: scripted narration (`docs/demo_script.md`), clear microphone, 1080p screen capture, captions recommended, every team member speaks at least once if required by the lecturer.

---

## 16. Ethics, privacy, licensing

### 16.1 Privacy

* Datasets are public and already anonymised by their publishers (phone numbers/amounts masked). Never try to de-anonymise. Examples shown in the app, report, and video must come from the test split and contain no personal names, numbers, or links (replace with placeholders if needed).
* The app stores nothing: no logging of inputs, no analytics, session state only.

### 16.2 Fairness and transparency

* Region/dialect coverage is uneven (UK/SG/IN/US-heavy). Informal or non-native English ham may face a higher false-positive rate: E7 must report FP examples of this type, if any.
* Explanations (occlusion), calibrated scores, the suspicious band, and a visible disclaimer support appropriate reliance.

### 16.3 Limitations to state in the app and report

English only; trained on data up to 2024; spammers adapt (concept drift); label definitions differ across sources (some promotional/transactional messages are borderline); a 50/50 test set inflates precision (see E2); not a security product; no URL reputation checks.

### 16.4 Licensing and attribution

* S1 (CC BY 4.0): cite Almeida & Hidalgo (UCI, DOI 10.24432/C5CC84) and Almeida et al. (DocEng 2011) in README, app About tab, and report.
* S2: cite Salman et al. (IEEE Access 2024); use for academic, non-commercial purposes; do not redistribute data files; the repo provides scripts that download from the authors' repository.
* `distilbert-base-uncased`: Apache-2.0; cite Sanh et al. (2019).
* Code: MIT licence.

---

## 17. Ordered task list for the coding agent (with Definitions of Done)

| # | Task | Output | Definition of Done |
|---|---|---|---|
| T01 | Scaffold repo per §11.1; `.gitignore`; `pyproject.toml` (ruff, pytest markers `slow`); `configs/config.yaml`; empty `docs/*` | Repo skeleton | `ruff check .` passes; `pytest` collects 0 errors |
| T02 | Environment setup per §11.2; `reports/metrics/environment.json` (Python, OS, GPU, library versions) | Working venv | CUDA available = True on the RTX 3050 machine |
| T03 | `preprocessing.normalize` (Appendix A.1) + `tests/test_preprocessing.py` | Module + tests | Tests pass, including idempotency |
| T04 | `data/download.py` + `scripts/01_download_data.py` (FR-D1), with retry and fallback | `data/raw/*`, `MANIFEST.json` | Super SHA-256 matches; UCI 5,574 rows |
| T05 | `data/build.py` + `scripts/02_build_dataset.py` (Appendix A.2) + `tests/test_build_dataset.py` | `data/processed/*` | Counts match §5.2; build is deterministic |
| T06 | `data/audit.py` + `scripts/03_audit_dataset.py` + `tests/test_data_quality.py` | `reports/metrics/data_audit.json` | All 8 invariants pass |
| T07 | `docs/DATA_CARD.md` (sources, licences, curation, counts, limitations) and `DATA_LICENSES.md` | Docs | Reviewed against §4 |
| T08 | `scripts/04_eda.py` → F1–F4, T1–T2 | Figures/tables | Files exist; grayscale-readable |
| T09 | `models/classical.py` + `scripts/05_train_classical.py` (M1, M2, grid CV) | joblib files, `cv_*.json`, T3 | CV macro-F1 within §9.5 ranges; files < 50 MB |
| T10 | `models/transformer.py` + `scripts/06_train_distilbert.py` (M3) + Colab notebook | `models/distilbert/`, `training_log.json`, F10 | Val macro-F1 ≥ 0.98; training log saved |
| T11 | `calibration.py` + `scripts/07_calibrate.py` + `tests/test_calibration.py` | `models/calibration.json` | T, τ, τ_low for all 3 models; val FPR ≤ 1% at τ |
| T12 | `inference.py` + `explain.py` + tests | Uniform predictors | Contract tests pass; CPU latency measured |
| T13 | `evaluation/*` + `scripts/08_evaluate.py`: E1–E8 (+ E9/E10 optional) | `reports/metrics/*.json`, T4–T11, F5–F9 | All files generated; SC1–SC6 evaluated in `reports/metrics/success_criteria.json` |
| T14 | E7 manual tagging support: export FP/FN CSV with model probabilities → team tags → script builds T10 + `error_analysis.md` | Error taxonomy | Every FP/FN tagged |
| T15 | `app/streamlit_app.py` per §10 + `tests/test_app.py` | Working app | All FR-A1…A10 demonstrably met; AppTest passes |
| T16 | README (quick start, results table, live URL, screenshots, citations), `MODEL_CARD.md` | Docs | A fresh clone works by following the README alone |
| T17 | Push M3 to Hugging Face Hub (§13.1) | Hub model repo | App loads M3 from the Hub with a clean cache |
| T18 | Deploy to Streamlit Cloud (§13.2); pin `requirements.txt` | Public URL | Manual QA §12.3 complete |
| T19 | CI workflow (§12.2) | `.github/workflows/ci.yml` | Green run on main |
| T20 | `scripts/run_all.py` end-to-end reproduction check on a clean clone | Log | Classical metrics reproduce exactly; M3 within ±0.3 pp |
| T21 | Report assets: export final tables to `reports/tables/*.csv` + LaTeX/Word-ready versions; copy figures | Report-ready assets | §14.3 checklist complete |
| T22 | `docs/demo_script.md` filled with real numbers | Script | Timed read-through ≤ 4:40 |

---

## 18. Timeline (course weeks)

| Week | Milestone | Tasks |
|---|---|---|
| 6 | **Proposal submitted** (checkpoint) | Proposal PDF; T01–T03 |
| 7 | Data ready | T04–T08 |
| 8 | Classical models | T09; begin T10 |
| 9 | Transformer + calibration | T10–T12 |
| 10 | Evaluation complete | T13–T14 |
| 11 | Application | T15–T16 |
| 12 | Deployment + QA | T17–T20 |
| 13 | Report + video | T21–T22; write report; record video |
| 14 | **Submit by Sunday 23:59 WIB** | Final checklist §20; submit GitHub repo link, PDF report, MP4/YouTube link on BINUSMAYA. Late penalty: −10% per day |

Buffer: aim to finish all code by the end of Week 12 so that Weeks 13–14 are for writing and polishing.

---

## 19. Risks and mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Source URL changes or goes offline | Low | High | Pinned commit URLs; UCI mirror fallback; keep a private local copy of `data/raw/` (not committed) |
| Library version drift changes outputs | Medium | Medium | Pin versions after the first full run; `requirements-lock.txt`; count tolerance §5.2 |
| CUDA/torch install problems on Windows | Medium | Medium | pytorch.org selector; Colab fallback notebook |
| GPU out of memory (6 GB) | Low | Low | fp16, batch 16 + gradient accumulation, `max_length=128` |
| Streamlit Cloud Python/torch incompatibility | Medium | High | Choose Python 3.11 in Advanced settings; CPU index; FR-A9 fallback to M2 |
| Scores near ceiling make comparison uninformative | Medium | Medium | CIs + McNemar; E2/E3/E6 provide harder, more informative tests |
| Hidden leakage | Low | High | Near-duplicate split check; artefact audit; sanity ranges §9.5 |
| Licence questions about S2 | Low | Medium | Cite; no redistribution; academic use; disclose in report |
| Over-claiming in UI | Low | Medium | Mandatory disclaimer; wording rule §8.1 |
| Team coordination | Medium | Medium | Issues board, PR reviews, weekly log |

---

## 20. Final submission checklist

* [ ] GitHub repo public (or accessible), README complete, licence files present, no data or large weights committed, CI green.
* [ ] App: local run verified; public URL verified the same day as submission.
* [ ] Report PDF: 10–15 pages, required structure, IEEE references, team contribution statement, screenshots, all numbers traceable to `reports/`.
* [ ] Video: < 5:00, MP4 or YouTube link (unlisted, viewable without sign-in).
* [ ] Submitted on BINUSMAYA before Week 14 Sunday 23:59 WIB.

---

## Appendix A — Reference code (verified)

These two files were run end to end on the real data during planning and produced the numbers in this document. Port them into `src/smsspam/` unchanged in logic (style changes are fine).

### A.1 `preprocessing.py`

```python
"""Shared text normaliser (reference implementation for EXECUTION_GUIDE.md section 6).

Applied identically to EVERY model, at training AND inference time.
Purpose: map dataset-specific anonymisation masks ("XXXX", "<#>", "<DECIMAL>") and real
values (digits, phone numbers, URLs) onto the SAME placeholders, so that a model cannot
learn "contains XXXX => spam" (an annotation artefact) and real user input looks like
training input.
"""
import re
import unicodedata

PLACEHOLDERS = ("[URL]", "[EMAIL]", "[PHONE]", "[MONEY]", "[NUM]")

_URL = re.compile(
    r"(?:https?://|www\.)\S+"
    r"|\b[a-z0-9-]+(?:\.[a-z0-9-]+)*\.(?:com|net|org|ly|me|co|info|io|uk|au|in|us|xyz|top|link|site|online|club|app|gl|biz)\b(?:/\S*)?",
    re.I,
)
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_MASK_TAGS = re.compile(r"<(?:#|DECIMAL|TIME)>", re.I)
_MASK_URL = re.compile(r"<URL>", re.I)
_MASK_EMAIL = re.compile(r"<EMAIL>", re.I)
# digit-or-mask character: real digits OR the uppercase X used by dataset anonymisers
_D = r"[0-9X]"
_PHONE = re.compile(rf"(?<![A-Za-z0-9])\+?{_D}(?:[\s\-()]?{_D}){{6,}}(?![A-Za-z0-9])")
_MONEY = re.compile(
    rf"(?:[£$€₹]|\b(?:rs|inr|usd|gbp|eur|php|ngn|ksh|aud|sgd)\.?)\s?{_D}[{_D[1:-1]},.]*(?:\s?(?:k|m|lakh|lac|lacs|crore|cr)\b)?",
    re.I,
)
_NUM = re.compile(rf"(?<![A-Za-z0-9]){_D}*\d{_D}*(?:[.,:/]{_D}+)*(?![A-Za-z0-9])|\bX{{2,}}\b|(?<![A-Za-z'\-])X(?![A-Za-z'\-])")


def normalize(text: str) -> str:
    """Return the normalised form of one SMS. Pure function, idempotent."""
    if text is None:
        return ""
    s = unicodedata.normalize("NFKC", str(text))
    s = _MASK_URL.sub(" [URL] ", s)
    s = _MASK_EMAIL.sub(" [EMAIL] ", s)
    s = _MASK_TAGS.sub(" [NUM] ", s)
    s = _EMAIL.sub(" [EMAIL] ", s)
    s = _URL.sub(" [URL] ", s)
    s = _MONEY.sub(" [MONEY] ", s)
    s = _PHONE.sub(" [PHONE] ", s)
    s = _NUM.sub(" [NUM] ", s)
    s = re.sub(r"(?:\[NUM\]\s*){2,}", "[NUM] ", s)   # collapse runs
    return re.sub(r"\s+", " ", s).strip()
```

### A.2 `build_dataset.py`

```python
"""Build the curated, balanced SMS spam dataset (v1).

Reference implementation for EXECUTION_GUIDE.md, section 5 (Data Pipeline).

Inputs (raw, never modified):
  * UCI SMS Spam Collection  -> SMSSpamCollection (tab-separated: label<TAB>text), 5,574 rows
  * Super SMS Dataset        -> super_sms_dataset.csv (columns: SMSes, Labels), 67,010 rows, cp1252

Usage:
  python build_dataset.py --uci data/raw/SMSSpamCollection \
                          --super data/raw/super_sms_dataset.csv \
                          --out data/processed
Outputs:
  sms_spam_curated_v1.csv (text,label,source,split), train.csv, val.csv, test.csv,
  unused_ham_pool.csv, data_stats.json
"""
import argparse
import hashlib
import json
import re
import unicodedata
from pathlib import Path

import ftfy
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42

URL_RE = re.compile(
    r"(https?://\S+|www\.\S+|\b[a-z0-9-]+\.(?:com|net|org|ly|me|co|info|io|uk|au|in|us|xyz|top|link|site)(?:/\S*)?)",
    re.I,
)
# --- Artifact 1: rows that were lower-cased AND stripped of all digits/punctuation upstream.
ARTIFACT_PUNCT = re.compile(r"[\d.,!?:;'\"/()\-@#$%&*+=<>£€]")
# --- Artifact 2: rows that were lemmatised upstream ("This be a reminder", "KYC have be suspend").
NOT_LEMMA_PREV = set(
    "to will would could should shld shd can may might must not cannot ll 'll ill i'll wil let gonna wanna "
    "gotta dont don't won't wont never also just always shall ever can't cant really still all definitely "
    "probably prob maybe must've so".split()
)
LEMMA_VERBS = (
    "deliver|attempt|suspend|lock|block|charge|process|verify|confirm|schedule|ship|update|cancel|detect|restrict|"
    "disable|approve|send|receive|hold|decline|flag|generate|expire|register|issue|reserve|credit|debit|select|"
    "activate|deactivate|terminate|cap|limit|sign|add|deduct|transfer|pend|complete|accept|reject|note|record|"
    "refund|pay|bill|grant|award|choose|notify|initiate|freeze|close|open|use|change|reset|link|start|stop|make|"
    "place|dispatch|return|mark|post|apply|qualify|submit|enable|renew|upgrade|offer|win|claim|allot|allocate|"
    "assign|book"
)
LEMMA_P1 = re.compile(
    r"\b(?:have|has|had) be\b|\b(?:this|there|which|who) be\b|\bname be\b"
    r"|\b(?:i|he|she|we|they|you|it|this|that|account|card|number|order|package|parcel|payment|message|offer|code|link)"
    r" be (?:a|an|the|your|my|our|from|now|eligible|due|valid|available|pending|requir\w*)\b"
    r"|\b(?:i|we|they) be (?:hire|hiring|recruit\w*|look\w*|offer\w*|pleased|glad|happy|writ\w*|contact\w*|inform\w*)\b",
    re.I,
)
# --- Artifact 3: Penn-Treebank tokenisation traces ("Do n't", "It 's", ``quotes'') found only in spam.
#     Fixed by DETOKENISING every row (not by dropping rows).
PTB_FIXES = [
    (re.compile(r"(\w) (n't|'s|'re|'m|'ve|'ll|'d)\b", re.I), r"\1\2"),
    (re.compile(r"``\s*|\s*''"), '"'),
    (re.compile(r"\b(wan|gon|got) na\b", re.I), r"\1na"),
]
LEMMA_P2 = re.compile(r"\b([a-z']+) be (?:" + LEMMA_VERBS + r")\b", re.I)


def clean_text(s: str) -> str:
    s = ftfy.fix_text(str(s))                     # repair multi-level mojibake (Ã‚Â£ -> £)
    s = s.replace("\ufffd", "")              # drop unrecoverable replacement chars
    s = re.sub(r"Â(?=[^A-Za-z£])", "", s)          # stray 'Â' left before apostrophes/punctuation
    s = unicodedata.normalize("NFKC", s)
    s = s.translate(str.maketrans({"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"', "\u2013": "-", "\u2014": "-", "\u00a0": " "}))
    s = re.sub(r"\s+", " ", s)                     # collapse first so PTB patterns see single spaces
    for pattern, repl in PTB_FIXES:
        s = pattern.sub(repl, s)
    return re.sub(r"\s+", " ", s).strip()


def dedup_key(s: str) -> str:
    """Near-duplicate key: templated spam that differs only in numbers/URLs/punctuation collapses."""
    s = s.lower()
    s = URL_RE.sub(" URL ", s)
    s = re.sub(r"\d+", "0", s)
    s = re.sub(r"[^a-z0 ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def is_raw_format(s: str) -> bool:
    return bool(re.search(r"[A-Z]", s) or ARTIFACT_PUNCT.search(s))


def is_lemmatised(s: str) -> bool:
    if LEMMA_P1.search(s):
        return True
    return any(m.group(1).lower() not in NOT_LEMMA_PREV for m in LEMMA_P2.finditer(s))


def is_latin_script(s: str) -> bool:
    letters = [c for c in s if c.isalpha()]
    if not letters:
        return False
    non_latin = sum(1 for c in letters if "LATIN" not in unicodedata.name(c, ""))
    return non_latin / len(letters) <= 0.10


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--uci", type=Path, required=True)
    ap.add_argument("--super", dest="sup", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    uci = pd.read_csv(args.uci, sep="\t", header=None, names=["label", "text"], quoting=3)
    assert len(uci) == 5574, f"UCI row count {len(uci)} != 5574"
    uci = pd.DataFrame({"text": uci.text, "label": (uci.label == "spam").astype(int),
                        "source": "uci_sms_spam_collection"})
    sup = pd.read_csv(args.sup, encoding="cp1252", encoding_errors="replace").dropna(subset=["SMSes", "Labels"])
    sup = pd.DataFrame({"text": sup.SMSes.astype(str), "label": sup.Labels.astype(int),
                        "source": "super_sms_dataset"})

    stats = {"raw": {"uci": uci.label.value_counts().to_dict(), "super": sup.label.value_counts().to_dict()}}
    df = pd.concat([uci, sup], ignore_index=True)  # UCI first: its copy wins during de-duplication
    df["text"] = df.text.map(clean_text)

    steps = {}

    def step(name, keep_mask):
        nonlocal df
        dropped = df[~keep_mask]
        steps[name] = {"removed": int((~keep_mask).sum()),
                       "removed_ham": int((dropped.label == 0).sum()),
                       "removed_spam": int((dropped.label == 1).sum())}
        df = df[keep_mask].copy()

    step("too_short", (df.text.str.len() >= 5) & (df.text.str.split().str.len() >= 2))
    step("format_stripped_artifact", df.text.map(is_raw_format))
    step("lemmatisation_artifact", ~df.text.map(is_lemmatised))
    step("non_latin_script", df.text.map(is_latin_script))
    df["key"] = df.text.map(dedup_key)
    n_labels = df.groupby("key").label.nunique()
    step("conflicting_labels", ~df.key.isin(n_labels[n_labels > 1].index))
    step("near_duplicates", ~df.key.duplicated(keep="first"))
    step("empty_key", df.key.str.len() > 0)
    stats["steps"] = steps
    stats["after_cleaning"] = df.groupby(["source", "label"]).size().unstack(fill_value=0).to_dict()

    # Balance 1:1 by random under-sampling of ham (spam is the minority class).
    spam = df[df.label == 1]
    ham = df[df.label == 0].sample(n=len(spam), random_state=SEED)
    bal = pd.concat([spam, ham]).sample(frac=1, random_state=SEED)

    train, tmp = train_test_split(bal, test_size=0.30, stratify=bal.label, random_state=SEED)
    val, test = train_test_split(tmp, test_size=0.50, stratify=tmp.label, random_state=SEED)
    cols = ["text", "label", "source", "split"]
    parts = []
    for name, part in (("train", train), ("val", val), ("test", test)):
        part = part.assign(split=name)
        part[cols].to_csv(args.out / f"{name}.csv", index=False)
        parts.append(part)
    full = pd.concat(parts)
    full[cols].to_csv(args.out / "sms_spam_curated_v1.csv", index=False)
    unused = df[(df.label == 0) & (~df.index.isin(ham.index))]
    unused[["text", "label", "source"]].to_csv(args.out / "unused_ham_pool.csv", index=False)

    stats["final"] = {
        "total": len(full),
        "split_label": full.groupby(["split", "label"]).size().unstack().to_dict(),
        "source_label": full.groupby(["source", "label"]).size().unstack(fill_value=0).to_dict(),
        "unused_ham_pool": len(unused),
    }
    stats["sha256_inputs"] = {"uci": sha256(args.uci), "super": sha256(args.sup)}
    stats["sha256_output"] = sha256(args.out / "sms_spam_curated_v1.csv")
    (args.out / "data_stats.json").write_text(json.dumps(stats, indent=2, default=str))
    print(json.dumps(stats, indent=2, default=str))


if __name__ == "__main__":
    main()
```

---

## Appendix B — Reference results from the planning run (sanity checks only; not report numbers)

| Check | Result |
|---|---|
| Raw UCI | 5,574 (4,827 ham / 747 spam); 403 exact duplicate texts |
| Raw Super | 67,010 rows; 67,008 labelled (40,830 ham / 26,178 spam) |
| Formatting-only classifier on **raw** Super | 85.8% accuracy (majority baseline 60.9%) → artefact confirmed |
| Artefact-only classifier on **curated** v1 | 53.9% (chance 50%) |
| Mask-only classifier after `normalize()` | 50.3% |
| Source-only classifier | 52.7% |
| Cross-validated suspect labels | 17 (0.09%) |
| M1 TF-IDF(1–2) + MNB (with normalize), test | spam P 0.973, R 0.970, macro-F1 0.972 |
| M2 TF-IDF word+char + LR (with normalize), test | spam P 0.988, R 0.980, macro-F1 0.984 |
| TF-IDF word+char + LinearSVC (raw text), test | macro-F1 0.986; at 13.4% prevalence: spam P 0.948, R 0.981 |
| Drift (M1, raw text): train UCI-only → test Super part | spam recall 0.592 vs 0.973 with curated train |
| Normaliser speed | ~0.8 s for 19,152 messages |
| Planning environment | Python 3.13.16, pandas 3.0.5, scikit-learn 1.9.1, numpy 2.5.3, ftfy 6.3.1 (the project itself targets Python 3.11/3.12) |
| All 8 audit invariants (§5.4) | Pass |

---

## Appendix C — References (IEEE style; use in proposal, report, README)

[1] T. A. Almeida and J. M. G. Hidalgo, "SMS Spam Collection," UCI Machine Learning Repository, 2011. [Online]. Available: https://doi.org/10.24432/C5CC84

[2] T. A. Almeida, J. M. G. Hidalgo, and A. Yamakami, "Contributions to the study of SMS spam filtering: New collection and results," in *Proc. 11th ACM Symp. Document Engineering (DocEng)*, 2011, pp. 259–262, doi: 10.1145/2034691.2034742.

[3] M. Salman, M. Ikram, and M. A. Kaafar, "Investigating evasive techniques in SMS spam filtering: A comparative analysis of machine learning models," *IEEE Access*, vol. 12, pp. 24306–24324, 2024, doi: 10.1109/ACCESS.2024.3364671.

[4] S. Tang, X. Mi, Y. Li, X. Wang, and K. Chen, "Clues in tweets: Twitter-guided discovery and analysis of SMS spam," in *Proc. ACM SIGSAC Conf. Computer and Communications Security (CCS)*, 2022, doi: 10.1145/3548606.3559351.

[5] D. Timko and M. L. Rahman, "Smishing dataset I: Phishing SMS dataset from Smishtank.com," in *Proc. 14th ACM Conf. Data and Application Security and Privacy (CODASPY)*, 2024, doi: 10.1145/3626232.3653282.

[6] H. Abdel-Jaber, "Detecting spam and ham SMS messages using natural language processing and machine learning algorithms," *PeerJ Computer Science*, vol. 11, e3232, 2025, doi: 10.7717/peerj-cs.3232.

[7] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, "BERT: Pre-training of deep bidirectional transformers for language understanding," in *Proc. NAACL-HLT*, 2019, pp. 4171–4186.

[8] V. Sanh, L. Debut, J. Chaumond, and T. Wolf, "DistilBERT, a distilled version of BERT: Smaller, faster, cheaper and lighter," arXiv:1910.01108, 2019.

[9] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, "On calibration of modern neural networks," in *Proc. 34th Int. Conf. Machine Learning (ICML)*, 2017, pp. 1321–1330.

[10] T. Wolf *et al.*, "Transformers: State-of-the-art natural language processing," in *Proc. EMNLP: System Demonstrations*, 2020, pp. 38–45.

---

## Appendix D — Glossary

* **Ham:** a legitimate (non-spam) message.
* **Macro-F1:** the unweighted mean of the per-class F1 scores.
* **FPR (ham):** the share of legitimate messages wrongly flagged as spam.
* **ECE:** Expected Calibration Error; the average gap between predicted probability and observed frequency across bins.
* **Annotation artefact:** a non-semantic pattern (formatting, masks, tokenisation) that correlates with the label because of how the data was collected or processed, not because of what the message says.
* **Near duplicate:** messages that become identical after lowercasing and replacing digits/URLs/punctuation.
