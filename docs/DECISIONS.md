# Implementation decisions

| Date (WIB) | Decision | Reason | Impact |
|---|---|---|---|
| 2026-10-12 | Keep all Git commits local; do not push or publish. | Explicit user instruction. | T17/T18 and a remote CI run remain pending until publication is authorized. |
| 2026-10-12 | Use the available Python 3.12 interpreter. | Accepted by NFR-9; Python 3.11 is not installed. | Record exact versions and verify all acceptance checks. |
| 2026-10-12 | Exclude the supplied dataset/reference ZIP from Git. | It contains processed S2 data; the guide prohibits redistribution. | Preserve the ZIP locally and rebuild data from pinned sources. |
| 2026-10-12 | Bootstrap a project venv with system site packages. | A working CUDA PyTorch 2.6.0+cu124 already exists locally. | Avoid replacing global packages; record inherited versions and provide a lock file for clean installs. |
| 2026-10-12 | Remove the CPU-only torch pulled transitively into the local venv. | It would hide the existing CUDA build. | The venv now uses verified system PyTorch 2.6.0+cu124; global packages were not changed. |
| 2026-10-12 | Generate F4 immediately after T09 rather than before model fitting in T08. | F4 requires the selected M2 coefficients, which do not exist until T09; the task table has a dependency cycle. | T08 produces F1–F3 and T1–T2; T09 also produces F4. No provisional or test-tuned model is introduced. |
| 2026-10-12 | Record 2 missing-label/text Super rows rather than the prose's 3. | The pinned source has 67,010 raw rows and 67,008 usable rows, matching Appendix B and the reference code. | All removal counts and the curated output hash match exactly. |
