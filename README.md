# NHS RTT Predictor

Streamlit dashboard for exploring NHS England Referral to Treatment (RTT) waiting-list trends, specialty performance, and a transparent patient-facing wait proxy tool.

The core product idea is unchanged:

1. National Trend
2. By Specialty
3. Patient Predictor

This refactor focuses on honesty, maintainability, and safer operational behavior rather than changing the product direction.

## What The App Does

- Shows national incomplete-pathway RTT trends from the loaded dataset.
- Lets users inspect specialty-level trends and trust-level latest-month performance.
- Uses Prophet when enough monthly history is available, with safer baseline fallback forecasting when it is not.
- Suggests a likely specialty from symptom text using a simple rule-based keyword scorer.
- Builds patient wait outputs from trust + specialty latest-month aggregates where available, with explicit trust-wide fallback when specialty rows are missing.

## Important Honesty Notes

- The patient-facing wait output is an `estimated_wait_weeks_proxy`, not a true median wait, not a patient-level prediction, and not a guaranteed appointment date.
- The proxy is derived from RTT performance metrics and simple heuristic adjustments, especially for urgency.
- Alternative trust suggestions are specialty-aware where the latest trust-level specialty data exists. Where it does not exist, the app falls back to trust-wide latest-month data and says so in the UI.
- Forecasts are indicative only. With limited history, the app intentionally falls back to simple baselines instead of overclaiming confidence.

## Architecture

The app is now split into small modules:

- `app.py`: Streamlit UI, styling, layout, and cached orchestration.
- `src/config.py`: Colors, targets, dataset paths, wording, and shared constants.
- `src/data_loader.py`: Local-first CSV loading, dataset validation, and optional Kaggle download logic.
- `src/aggregations.py`: National, specialty, trust-wide, and trust + specialty aggregations.
- `src/forecasting.py`: Prophet forecasting with minimum-history checks, holdout evaluation, and fallback baselines.
- `src/patient_logic.py`: Specialty suggestion, urgency adjustment, specialty-aware wait resolution, and patient recommendation text.
- `src/utils.py`: Formatting, percentage, and wait-proxy helpers.
- `tests/`: Unit tests for validation, aggregation, patient logic, and forecast fallback behavior.

## Data Expectations

The dashboard expects a CSV containing at least these columns:

- `period`
- `provider_org_code`
- `provider_org_name`
- `commissioner_org_name`
- `rtt_part_type`
- `treatment_function_code`
- `treatment_function_name`
- `total_waiting`
- `within_18_weeks`
- `over_52_weeks`

The app prefers a local CSV first. By default it looks for:

- `nhs_rtt_waiting_times_2021_2025.csv` in the project root
- `/tmp/nhs_rtt_waiting_times_2021_2025.csv`

If the local CSV has incomplete history, the app can also use the configured Google Drive full-history file. The file must be shared as "Anyone with the link" for server deployments to download it. Kaggle remains available as an optional fallback when Streamlit secrets are configured.

The date range shown in the dashboard is derived from the loaded file, not hard-coded. If the loaded CSV only contains a subset of months, the UI and forecast behavior will reflect that smaller range.

## Setup

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the dashboard:

```bash
streamlit run app.py
```

Run tests:

```bash
pytest -q
```

## Local-First Operation

The recommended way to run the app locally is to place the full RTT CSV in the project folder and run Streamlit locally.

Why this is preferred:

- Fewer runtime failure points
- Clearer reproducibility
- Faster startup
- No dependency on external credentials for normal use

If the local file is incomplete, the app will try the configured Google Drive source and use whichever source has broader monthly coverage.

## Optional Kaggle Secrets

If no local CSV is present, the app can optionally download the dataset from Kaggle when Streamlit secrets are configured.

Example `.streamlit/secrets.toml`:

```toml
[kaggle]
username = "your_kaggle_username"
key = "your_kaggle_api_key"
```

Kaggle is a fallback only. The app does not depend on it during normal local use.

## Forecasting Approach

- Prophet is used only when there are at least 12 monthly points.
- If there is not enough history, the app uses a safer baseline:
  - moving average for short series
  - last value for very short series
- Negative forecasts are clipped to zero.
- A lightweight holdout MAE and MAPE summary is shown where practical.

This makes the dashboard more conservative and less brittle when data history is limited.

## Patient Predictor Approach

The patient tab now follows this order:

1. Try latest trust + specialty RTT aggregate for the selected trust and specialty.
2. If unavailable, fall back to latest trust-wide RTT aggregate.
3. Build alternative trust suggestions within the selected region using specialty-aware rows where available.
4. Apply a simple urgency multiplier to the proxy estimate for display.

The recommendation text is split into:

- data-based findings
- heuristic assumptions
- general NHS guidance

## Limitations

- RTT data is aggregated and administrative, not patient-level.
- `commissioner_org_name` is used as the region / ICB-style grouping shown in the UI because that is what the dataset exposes here.
- The wait proxy is only a heuristic derived mainly from RTT 18-week performance.
- Urgency adjustments are simple rule-based multipliers, not pathway-specific operational data.
- Forecast quality depends heavily on how many months are available in the CSV.
- The specialty suggestion system is intentionally rule-based and lightweight; it is not a clinical classifier.
- This project does not provide medical advice.

## Disclaimer

Use the dashboard for planning and exploratory insight only. Confirm any clinical or referral decisions with a GP, consultant, or the relevant NHS service.
