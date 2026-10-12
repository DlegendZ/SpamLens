# Curated SMS dataset card

## Purpose and provenance

English SMS binary classification (0 = ham, 1 = spam). Seed 42. Sources are
UCI SMS Spam Collection (CC BY 4.0) and the author-published Super SMS Dataset
(research release without an explicit license file; academic use only; no redistribution).
Attribution and source links are in [DATA_LICENSES.md](../DATA_LICENSES.md).

## Measured build

All counts below are generated from reports/metrics/data_stats.json.

| Partition | Ham | Spam | Total |
|---|---:|---:|---:|
| train | 6,703 | 6,703 | 13,406 |
| val | 1,436 | 1,437 | 2,873 |
| test | 1,437 | 1,436 | 2,873 |

Total: 19,152 messages. Unused held-out ham pool: 29,293.

Curated file SHA-256: `cda5cbb3fb8bf44619803a549ea229c67f41701489454789878969f085fa19aa`.

## Curation

UCI copies take precedence. Apply ftfy, Unicode NFKC, whitespace/quote repair
and PTB detokenization, then the following ordered filters:

| Filter | Ham removed | Spam removed |
|---|---:|---:|
| too_short | 264 | 2 |
| format_stripped_artifact | 1,549 | 13,327 |
| lemmatisation_artifact | 14 | 2,634 |
| non_latin_script | 1 | 0 |
| conflicting_labels | 6 | 5 |
| near_duplicates | 4,954 | 1,381 |
| empty_key | 0 | 0 |

Balance by seeded random ham undersampling; retain all clean spam. Split 70/15/15
with stratification. Apply the same reference normalizer to all model inputs.

## Audit evidence

All 8 invariants pass. Diagnostic five-fold CV accuracies:

| Diagnostic | Accuracy | Limit |
|---|---:|---:|
| Formatting artefacts | 0.5482 | 0.56 |
| Raw masks after normalization | 0.5000 | 0.55 |
| Source only | 0.5272 | 0.56 |

No near-duplicate keys cross splits, and the unused ham pool shares no key with the balanced corpus.

## Limits and privacy

English-only, with regional UK/SG/IN/US skew and informal/Singlish ham.
Sources cover data up to 2024; future spam can drift. Labels are kept as published
and borderline promotional/transactional messages may contain residual label noise.
A balanced test set inflates deployment precision; E2 must also evaluate at 13.4% spam prevalence.
The corpus is already anonymized by publishers. Never de-anonymize or redistribute it.
The model is an assistive estimate, not a security guarantee. Keep user input out of logs and files.
