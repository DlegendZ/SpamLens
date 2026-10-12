# Measured findings for the report

All numbers are read from generated files in reports/. Do not use Appendix B's
planning results as final measured values.

## Evaluation status

SC1–SC6 pass in `reports/metrics/success_criteria.json`. SC7 is pending the
application and deployment. The three models and their calibration/thresholds
remained frozen throughout evaluation; the manifest records their SHA-256s.

M3's balanced-test macro-F1 is 0.9916, with a 95% bootstrap interval of
[0.9882, 0.9948]. Its test ham FPR is 0.0063, with an interval of [0.0027, 0.0105].
The FPR point estimate meets 1%, but the interval extends slightly above 1%; this
limits certainty about the true rate. E2 precision is 0.9627 at measured spam
prevalence 0.1340.

## Curation and NLP

E3 baseline M1 trained on UCI-only catches 0.5686 of Super test spam, versus
0.9680 from curated training at the same raw threshold 0.5. The train-only E4
ablation supports the final M2 normalizer: CV macro-F1 0.9868 versus raw lowercase
0.9856, stopword removal 0.9847, and lemmatization 0.9856. M1 raw lowercase is
slightly higher than normalization (0.9715 versus 0.9702), but normalization is
mandatory to remove the mask shortcut and maintain a shared train/inference path.

## Robustness limitations

Lowercasing produces a small prediction change despite uncased downstream models.
The preserved reference normalizer interprets uppercase `X` masks before lowercasing;
a lowercase perturbation changes that interpretation. This is disclosed rather
than changing a frozen preprocessing rule using test results.

For M3, leetspeak raises ham FPR from 0.0063 to 0.0306; stripping punctuation raises
it to 0.0181. The deployment success criteria on unperturbed SMS do not guarantee
robustness under these perturbations. URL-domain swapping leaves predictions
unchanged for all three models, consistent with URL normalization.

## Human-reviewed errors

The team reviewed all 24 M3 test errors (9 FP, 15 FN) and 5 train-only
confident-disagreement flags. T10 and reports/error_analysis.md contain their
categories and anonymized examples. The suspect flags are not confirmed label
mistakes; the original labels and frozen models remain unchanged.

CPU throughput and long-explanation evidence are recorded separately in
`reports/metrics/cpu_latency.json`; inspect the measured limit flags in
`reports/metrics/runtime_validation.json` before claiming every NFR is met.
The final local benchmark uses up to 12 CPU threads: M3 single-message p95 is
33.2 ms, and the median 60-word stress-case explanation is 1.54 s across three
warm runs. These are local hardware measurements; cloud limits require separate
verification after deployment.
