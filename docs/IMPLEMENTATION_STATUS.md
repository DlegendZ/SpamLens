# Implementation status

The execution order and acceptance checks are defined in [EXECUTION_GUIDE.md](EXECUTION_GUIDE.md).
Never mark a task complete without its required evidence.

| Task | Status | Evidence / next requirement |
|---|---|---|
| T01 | Complete | Ruff passes; pytest collection has no errors (no functional tests yet). |
| T02 | Complete | environment.json: Python 3.12.8, CUDA 12.4, RTX 3050; spaCy/NLTK resources installed. |
| T03 | Complete | 23 normalizer tests pass, including idempotency and preserved alphanumeric codes. |
| T04 | Complete | Both source hashes match; UCI has 5,574 rows and Super 67,010; 28 tests pass. |
| T05 | Complete | Counts/removals match exactly; 19,152 rows; output SHA-256 matches cda5cbb3…fa19aa; 7 build tests pass. |
| T06 | Complete | data_audit.json: all eight invariants pass. |
| T07 | Complete | DATA_CARD.md generated from measured statistics; DATA_LICENSES.md reviewed against §4. |
| T08 | Complete with dependency exception | F1–F3 and T1–T2 generated; F4 comes from selected M2 at T09 (DECISIONS.md). |
| T09 | Complete | Train-only CV macro-F1: M1 0.9702, M2 0.9868; artifacts 0.92/1.26 MB; F4 and T3 generated. |
| T10 | In progress | DistilBERT fine-tuning on train, early stopping on validation only. |
| T11–T22 | Pending | Execute sequentially after the preceding acceptance check passes. |

Git policy: local commits only. No push or external publication.
