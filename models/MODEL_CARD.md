---
language: en
tags:
- text-classification
- spam-detection
- academic-research
---

# SpamLens model card

## Intended use

Assistive English SMS spam/ham classification for a non-commercial academic
NLP project (COMP6885001). Output is an automated estimate, not a security
guarantee. Verify unexpected requests with the official sender. Never use the
model to decide whether a URL, payment request, password request, or OTP is safe.

## Models and provenance

M1: word TF-IDF + Multinomial Naive Bayes. M2: word/character TF-IDF + Logistic
Regression. M3: fine-tuned `distilbert-base-uncased` (Sanh et al., 2019,
[arXiv:1910.01108](https://arxiv.org/abs/1910.01108); pretrained model Apache-2.0).
The shared reference normalizer is applied at training and inference.

Data: [UCI SMS Spam Collection](https://doi.org/10.24432/C5CC84), Almeida & Hidalgo
(2011), CC BY 4.0; and Super SMS Dataset, Salman, Ikram & Kaafar (IEEE Access, 2024),
[DOI 10.1109/ACCESS.2024.3364671](https://doi.org/10.1109/ACCESS.2024.3364671).
Super is an author-published research release without an explicit dataset
license file. This project limits use to academic, non-commercial work and
does not redistribute raw or processed datasets. No broader license grant is claimed.

Curated data: 19,152 messages, balanced 1:1, seed 42, train/validation/test
70/15/15. Counts, filtering, source hashes, audit evidence, and limits are in
`docs/DATA_CARD.md`, `DATA_LICENSES.md`, and `reports/metrics/data_stats.json`.

## Training and selection

Classical hyperparameters were selected with five-fold stratified CV on train only.
M3 used dynamic padding, max length 128, AdamW (2e-5), four maximum epochs,
batch 32/fp16 on RTX 3050 6 GB, and best validation macro-F1 selection.
Temperature scaling and thresholds use validation only. Test data were never
used for model fitting, early stopping, or threshold selection.

Recorded Python: 3.12.8; scikit-learn: 1.9.1;
Transformers: 4.57.6; PyTorch: 2.6.0+cu124.

## Measured held-out results

Values are generated from reports/metrics/main_results.json and natural_prevalence.json.

| Model | Macro-F1 (95% CI) | Spam recall | Ham FPR | E2 spam precision | Test ECE |
|---|---:|---:|---:|---:|---:|
| tfidf_nb | 0.9673 [0.9607, 0.9732] | 0.9415 | 0.0070 | 0.9461 | 0.0111 |
| tfidf_lr | 0.9854 [0.9809, 0.9892] | 0.9798 | 0.0090 | 0.9571 | 0.0066 |
| distilbert | 0.9916 [0.9882, 0.9948] | 0.9896 | 0.0063 | 0.9627 | 0.0042 |

E2 uses measured spam prevalence approximately 13.4%, including a held-out ham pool.
Intervals use 1,000 seeded paired-row bootstrap resamples. Exact McNemar comparisons
are in reports/metrics/mcnemar.json. Success criteria SC1–SC6 pass; the public-app
part of SC7 is pending publication. See docs/MEASURED_FINDINGS.md for robustness limits.

## Validation-selected calibration

| Model | Temperature | Spam threshold | Suspicious-band lower threshold |
|---|---:|---:|---:|
| tfidf_nb | 0.6796 | 0.83 | 0.16 |
| tfidf_lr | 0.8417 | 0.59 | 0.49 |
| distilbert | 1.4552 | 0.45 | 0.15 |

The app uses 'calibrated spam probability' only when that model's test ECE ≤0.03.

## Limitations, privacy, and ethics

English-only; regional/dialect coverage is uneven; sources end in 2024; future spam
can drift. Promotional/transactional boundaries differ by source and some published
labels are doubtful. Informal or mixed-language ham can be falsely flagged.
M3 ham FPR rises under leetspeak and punctuation stripping (T9); unperturbed success
criteria do not imply adversarial robustness. Balanced-test precision is optimistic
relative to real-world prevalence; use E2 for that comparison.

The application keeps input, upload results, and explanations in session memory only,
never on disk, in logs, or in analytics. Model resource caches contain only models.
The explanation uses leave-one-word-out occlusion (first 60 words, top 8 effects),
which ignores word interactions and is not a causal explanation.

## Availability

Classical artifacts and calibration are included in the Git repository. M3 weights remain in
gitignored models/distilbert/. Hugging Face model publication is pending.
A checkout without local M3 automatically uses M2 and displays a fallback notice.
