"""Forecasting helpers with Prophet guardrails and safe fallbacks."""

from __future__ import annotations

from dataclasses import dataclass
import os

import numpy as np
import pandas as pd

from src.config import (
    Columns,
    FORECAST_BASELINE_WINDOW,
    FORECAST_HOLDOUT_PERIODS,
    FORECAST_MIN_HISTORY_POINTS,
    FORECAST_PERIODS,
)

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

try:
    from prophet import Prophet
except Exception:  # pragma: no cover - exercised indirectly via fallback tests
    Prophet = None


@dataclass(frozen=True)
class ForecastResult:
    """Forecast output and related metadata."""

    history: pd.DataFrame
    future: pd.DataFrame
    method: str
    warning: str | None = None
    mae: float | None = None
    mape: float | None = None


def _prepare_history(series: pd.DataFrame) -> pd.DataFrame:
    history = (
        series[[Columns.PERIOD_DT, Columns.TOTAL_WAITING]]
        .rename(columns={Columns.PERIOD_DT: "ds", Columns.TOTAL_WAITING: "y"})
        .dropna()
        .sort_values("ds")
    )

    if history.empty:
        raise ValueError("No valid historical data points were available for forecasting.")

    history["y"] = pd.to_numeric(history["y"], errors="coerce")
    history = history.dropna(subset=["y"])

    if history.empty:
        raise ValueError("Forecast series did not contain any numeric values.")

    return history


def _clip_future_values(future: pd.DataFrame) -> pd.DataFrame:
    clipped = future.copy()
    for column in ("yhat", "yhat_lower", "yhat_upper"):
        clipped[column] = clipped[column].clip(lower=0)
    return clipped


def _build_baseline_forecast(
    history: pd.DataFrame,
    periods: int,
) -> tuple[pd.DataFrame, str]:
    if len(history) >= FORECAST_BASELINE_WINDOW:
        method = "moving_average_fallback"
        baseline_value = float(history["y"].tail(FORECAST_BASELINE_WINDOW).mean())
    else:
        method = "last_value_fallback"
        baseline_value = float(history["y"].iloc[-1])

    baseline_value = max(0.0, baseline_value)
    residual_scale = float(history["y"].std(ddof=0)) if len(history) > 1 else baseline_value * 0.1
    residual_scale = max(0.0, residual_scale)

    start = history["ds"].max() + pd.offsets.MonthBegin(1)
    future_dates = pd.date_range(start=start, periods=periods, freq="MS")
    future = pd.DataFrame(
        {
            "ds": future_dates,
            "yhat": np.repeat(baseline_value, periods),
            "yhat_lower": np.repeat(max(0.0, baseline_value - residual_scale), periods),
            "yhat_upper": np.repeat(baseline_value + residual_scale, periods),
        }
    )

    return _clip_future_values(future), method


def _run_prophet_forecast(history: pd.DataFrame, periods: int) -> pd.DataFrame:
    if Prophet is None:
        raise RuntimeError("Prophet is not available in this environment.")

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=0.05,
        interval_width=0.95,
    )
    model.fit(history[["ds", "y"]])
    future_dates = model.make_future_dataframe(periods=periods, freq="MS")
    forecast = model.predict(future_dates)[["ds", "yhat", "yhat_lower", "yhat_upper"]]
    return _clip_future_values(forecast[forecast["ds"] > history["ds"].max()].reset_index(drop=True))


def _evaluate_with_holdout(
    history: pd.DataFrame,
    periods: int,
    min_history_points: int,
) -> tuple[float | None, float | None]:
    if len(history) <= periods:
        return None, None

    train = history.iloc[:-periods].rename(columns={"ds": Columns.PERIOD_DT, "y": Columns.TOTAL_WAITING})
    actual = history.iloc[-periods:]["y"].reset_index(drop=True)

    if len(train) < 2:
        return None, None

    forecast = forecast_waiting_list(
        train,
        periods=periods,
        min_history_points=min_history_points,
        holdout_periods=periods,
        enable_evaluation=False,
    )

    predicted = forecast.future["yhat"].head(periods).reset_index(drop=True)
    if len(predicted) != len(actual):
        return None, None

    errors = (actual - predicted).abs()
    mae = float(errors.mean())
    mape = float((errors / actual.clip(lower=1)).mean() * 100)
    return round(mae, 2), round(mape, 2)


def forecast_waiting_list(
    series: pd.DataFrame,
    periods: int = FORECAST_PERIODS,
    min_history_points: int = FORECAST_MIN_HISTORY_POINTS,
    holdout_periods: int = FORECAST_HOLDOUT_PERIODS,
    enable_evaluation: bool = True,
) -> ForecastResult:
    """Forecast RTT waiting-list values with Prophet when safe, else a baseline fallback."""

    history = _prepare_history(series)
    warning: str | None = None

    if len(history) < min_history_points:
        future, method = _build_baseline_forecast(history, periods=periods)
        warning = (
            f"Only {len(history)} monthly points are available, so a simple baseline "
            "forecast is shown instead of Prophet."
        )
    else:
        try:
            future = _run_prophet_forecast(history, periods=periods)
            method = "prophet"
        except Exception as exc:
            future, method = _build_baseline_forecast(history, periods=periods)
            warning = (
                f"Prophet could not fit this series ({exc.__class__.__name__}), "
                "so a baseline forecast is shown instead."
            )

    mae: float | None = None
    mape: float | None = None
    if enable_evaluation and len(history) > holdout_periods:
        mae, mape = _evaluate_with_holdout(
            history,
            periods=holdout_periods,
            min_history_points=min_history_points,
        )

    return ForecastResult(
        history=history,
        future=future,
        method=method,
        warning=warning,
        mae=mae,
        mape=mape,
    )
