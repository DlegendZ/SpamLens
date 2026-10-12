# Implementation status

The execution order and acceptance checks are defined in [EXECUTION_GUIDE.md](EXECUTION_GUIDE.md).
Never mark a task complete without its required evidence.

| Task | Status | Evidence / next requirement |
|---|---|---|
| T01 | Complete | Ruff passes; pytest collection has no errors (no functional tests yet). |
| T02 | Complete | environment.json: Python 3.12.8, CUDA 12.4, RTX 3050; spaCy/NLTK resources installed. |
| T03 | Complete | 23 normalizer tests pass, including idempotency and preserved alphanumeric codes. |
| T04 | Complete | Both source hashes match; UCI has 5,574 rows and Super 67,010; 28 tests pass. |
| T05 | In progress | Port reference curation, verify counts and deterministic output. |
| T06–T22 | Pending | Execute sequentially after the preceding acceptance check passes. |

Git policy: local commits only. No push or external publication.
