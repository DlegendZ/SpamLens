# Local app verification (2026-10-12 WIB)

Run: `.\.venv\Scripts\python.exe -m streamlit run app\streamlit_app.py --server.port 8502`.
URL: http://127.0.0.1:8502/ (local only; no public deployment).

Automated evidence: reports/metrics/test_summary.json and test_results.xml.
Clean runtime evidence: reports/metrics/clean_runtime_check.json.

| Requirement | Evidence |
|---|---|
| FR-A1 | Six anonymized held-out examples, input cap enforced before inference, Check message button |
| FR-A2 | Verdict, calibrated probability gated by measured ECE, actual model, latency |
| FR-A3 | Batched first-60-word occlusion, top eight effects, escaped highlights and effect table |
| FR-A4 | Three sidebar models; default DistilBERT; actual-model indicator |
| FR-A5 | UTF-8/schema/size/row guards; 10-row sample uploaded in browser; predicted rows and downloaded CSV verified |
| FR-A6 | T4 comparison, selected-model confusion/reliability, E2 results, T1/T2 dataset summary |
| FR-A7 | Sources/licenses/citations, ethics/limitations/privacy, six team identities with blank roles |
| FR-A8 | Empty/whitespace disabled with info, oversized paste truncation warning, non-English warning with prediction |
| FR-A9 | Forced M3 failure tested; clean checkout without M3 automatically falls back to committed M2 |
| FR-A10 | Mandatory disclaimer under single/batch results; also present in About |

Browser verification used only anonymized examples: local DistilBERT single prediction,
word-effect expander, batch CSV upload/results/download, performance tab, and readable
375 px layout. Screenshots are in reports/screenshots/.

App data handling: model resources use the shared resource cache. User message history
and explanations are bounded to the last five checks in session state. Batch results
stay in that session. Input is not written to server files, logs, or analytics. A user
can explicitly download their result CSV to their own device.

Cloud cold start, cloud RAM, public-URL behavior, and public QA remain pending T18.
Local CPU latency measurements are not claims about an untested cloud deployment.
