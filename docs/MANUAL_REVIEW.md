# T14: team review required before application work

The guide requires human interpretation of every deployed-model test FP/FN and
train-only confident-disagreement flag. Do not change labels, thresholds, models,
or raw text during this review. The CSVs are private and excluded from Git.

## Review steps

1. Open `reports/review/errors_private.csv` in Excel or a UTF-8-capable CSV editor.
   Read `text`, `label` (0 ham, 1 spam), `predicted_label`, and `error_type`.
2. Fill `category` using one exact code from the table below. Add optional
   `review_notes` explaining your judgement. A category identifies the apparent
   cause; it does not assert the published label is correct.
3. For two rows per represented category (one if only one exists), fill
   `anonymized_example`. Remove personal names with `[NAME]`, phones with `[PHONE]`,
   links with `[URL]`, amounts with `[MONEY]`, and codes with `[NUM]`.
4. Review `reports/review/suspects_private.csv` the same way and fill `category`
   and optional notes. A confident model disagreement can still be a model error.
5. Save both CSVs as UTF-8. Run
   `.\.venv\Scripts\python.exe scripts\10_finalize_error_analysis.py`.
   The script requires complete categories before writing T10 and final analysis.

| Category code | Interpretation |
|---|---|
| transactional_otp | Transactional / OTP text resembles promotion |
| short_ambiguous | Too short or unclear to interpret reliably |
| conversational_spam | Romance, job scam, or other conversational spam |
| obfuscated | Evasive spelling, symbols, or formatting |
| code_mixed | Non-English or mixed-language wording |
| probable_label_noise | Published label appears doubtful after human inspection |

Use the best-fitting category; explain uncertainty in notes. Never treat a model
score as evidence that a message is safe. Tell the coding agent when the CSVs are
saved; it can run the finalizer and continue T15.
