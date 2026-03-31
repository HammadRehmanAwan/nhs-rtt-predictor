from __future__ import annotations

import pandas as pd

from src.config import Columns
from src.forecasting import forecast_waiting_list


def test_forecast_waiting_list_uses_baseline_for_short_history() -> None:
    series = pd.DataFrame(
        {
            Columns.PERIOD_DT: pd.date_range("2024-01-01", periods=3, freq="MS"),
            Columns.TOTAL_WAITING: [100, 120, 140],
        }
    )

    result = forecast_waiting_list(series)

    assert result.method == "moving_average_fallback"
    assert len(result.future) == 12
    assert result.warning is not None
    assert (result.future["yhat"] >= 0).all()


def test_forecast_waiting_list_falls_back_when_prophet_errors(monkeypatch) -> None:
    series = pd.DataFrame(
        {
            Columns.PERIOD_DT: pd.date_range("2023-01-01", periods=12, freq="MS"),
            Columns.TOTAL_WAITING: [100 + step * 5 for step in range(12)],
        }
    )

    class BrokenProphet:
        def __init__(self, *args, **kwargs):
            pass

        def fit(self, dataframe):
            raise RuntimeError("boom")

    monkeypatch.setattr("src.forecasting.Prophet", BrokenProphet)

    result = forecast_waiting_list(series)

    assert result.method == "moving_average_fallback"
    assert len(result.future) == 12
    assert result.warning is not None
    assert (result.future["yhat_lower"] >= 0).all()
