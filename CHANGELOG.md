# Change Summary

## Architecture

- Split the old monolithic `app.py` into focused modules for config, loading, aggregations, forecasting, patient logic, and utilities.
- Kept the Streamlit dashboard experience and the three original tabs.

## Data And Logic

- Switched the patient predictor to trust + specialty latest-month estimates where available.
- Added explicit trust-wide fallback logic when specialty-level trust data is missing.
- Renamed the misleading wait metric to `estimated_wait_weeks_proxy`.
- Made alternative trust suggestions specialty-aware where possible.
- Reworked specialty suggestion into a reusable multi-keyword scoring function.

## Forecasting And Reliability

- Added minimum-history checks before using Prophet.
- Added forecast fallback methods for short or failed series.
- Prevented negative forecast outputs.
- Added lightweight holdout MAE/MAPE reporting where practical.
- Moved Kaggle download logic out of the UI flow and made it optional.
- Added clearer data validation and load-time error messages.

## Quality And Documentation

- Added unit tests for validation, aggregation, patient logic, and forecast fallback behavior.
- Improved dependency management in `requirements.txt`.
- Rewrote the README with setup, architecture, limitations, and disclaimer guidance.
