# Implementation decisions

| Date (WIB) | Decision | Reason | Impact |
|---|---|---|---|
| 2026-10-12 | Keep all Git commits local; do not push or publish. | Explicit user instruction. | T17/T18 and a remote CI run remain pending until publication is authorized. |
| 2026-10-12 | Use the available Python 3.12 interpreter. | Accepted by NFR-9; Python 3.11 is not installed. | Record exact versions and verify all acceptance checks. |
| 2026-10-12 | Exclude the supplied dataset/reference ZIP from Git. | It contains processed S2 data; the guide prohibits redistribution. | Preserve the ZIP locally and rebuild data from pinned sources. |
| 2026-10-12 | Bootstrap a project venv with system site packages. | A working CUDA PyTorch 2.6.0+cu124 already exists locally. | Avoid replacing global packages; record inherited versions and provide a lock file for clean installs. |
| 2026-10-12 | Remove the CPU-only torch pulled transitively into the local venv. | It would hide the existing CUDA build. | The venv now uses verified system PyTorch 2.6.0+cu124; global packages were not changed. |
