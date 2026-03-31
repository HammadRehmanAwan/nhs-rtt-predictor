"""Aggregations for national, specialty, and trust-level RTT views."""

from __future__ import annotations

import pandas as pd

from src.config import (
    ALL_SPECIALTIES_CODE,
    Columns,
    ESTIMATE_SOURCE_SPECIALTY,
    ESTIMATE_SOURCE_TRUST_WIDE,
    RECENT_TREND_THRESHOLD_PCT,
    RECENT_TREND_WINDOW,
    RTT_PART_INCOMPLETE,
)
from src.utils import calculate_percentage, estimate_source_label, estimate_wait_weeks_proxy


def _incomplete_pathways(dataframe: pd.DataFrame) -> pd.DataFrame:
    return dataframe[dataframe[Columns.RTT_PART_TYPE] == RTT_PART_INCOMPLETE].copy()


def add_performance_columns(
    dataframe: pd.DataFrame,
    estimate_source: str,
) -> pd.DataFrame:
    """Add standard performance and wait-proxy columns."""

    df = dataframe.copy()
    df[Columns.PCT_WITHIN_18_WEEKS] = calculate_percentage(
        df[Columns.WITHIN_18_WEEKS],
        df[Columns.TOTAL_WAITING],
    )
    if Columns.OVER_52_WEEKS in df.columns:
        df[Columns.PCT_OVER_52_WEEKS] = calculate_percentage(
            df[Columns.OVER_52_WEEKS],
            df[Columns.TOTAL_WAITING],
        )
    df[Columns.ESTIMATED_WAIT_WEEKS_PROXY] = estimate_wait_weeks_proxy(
        df[Columns.PCT_WITHIN_18_WEEKS]
    )
    df[Columns.ESTIMATE_SOURCE] = estimate_source
    df[Columns.ESTIMATE_SOURCE_LABEL] = estimate_source_label(estimate_source)
    return df


def build_national_trend(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Build the national incomplete-pathway time series."""

    filtered = _incomplete_pathways(dataframe)
    filtered = filtered[
        filtered[Columns.TREATMENT_FUNCTION_CODE] == ALL_SPECIALTIES_CODE
    ]

    national = (
        filtered.groupby(Columns.PERIOD_DT)[
            [Columns.TOTAL_WAITING, Columns.WITHIN_18_WEEKS, Columns.OVER_52_WEEKS]
        ]
        .sum()
        .reset_index()
        .sort_values(Columns.PERIOD_DT)
    )

    return add_performance_columns(
        national,
        estimate_source=ESTIMATE_SOURCE_TRUST_WIDE,
    )


def build_specialty_trend(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Build specialty time series for incomplete pathways."""

    filtered = _incomplete_pathways(dataframe)
    filtered = filtered[
        filtered[Columns.TREATMENT_FUNCTION_CODE] != ALL_SPECIALTIES_CODE
    ]

    specialty = (
        filtered.groupby([Columns.PERIOD_DT, Columns.TREATMENT_FUNCTION_NAME])[
            [Columns.TOTAL_WAITING, Columns.WITHIN_18_WEEKS, Columns.OVER_52_WEEKS]
        ]
        .sum()
        .reset_index()
        .sort_values([Columns.TREATMENT_FUNCTION_NAME, Columns.PERIOD_DT])
    )

    return add_performance_columns(
        specialty,
        estimate_source=ESTIMATE_SOURCE_SPECIALTY,
    )


def latest_period_value(dataframe: pd.DataFrame) -> str:
    """Return the latest raw period string present in the data."""

    return str(dataframe[Columns.PERIOD].max())


def build_latest_trust_summary(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Build latest-month trust-wide performance rows."""

    filtered = _incomplete_pathways(dataframe)
    filtered = filtered[
        filtered[Columns.TREATMENT_FUNCTION_CODE] == ALL_SPECIALTIES_CODE
    ]
    filtered = filtered[filtered[Columns.PERIOD] == latest_period_value(dataframe)]

    trust = (
        filtered.groupby(
            [
                Columns.PROVIDER_ORG_CODE,
                Columns.PROVIDER_ORG_NAME,
                Columns.COMMISSIONER_ORG_NAME,
            ]
        )[[Columns.TOTAL_WAITING, Columns.WITHIN_18_WEEKS, Columns.OVER_52_WEEKS]]
        .sum()
        .reset_index()
        .sort_values(Columns.PROVIDER_ORG_NAME)
    )

    return add_performance_columns(
        trust,
        estimate_source=ESTIMATE_SOURCE_TRUST_WIDE,
    )


def build_latest_trust_specialty_summary(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Build latest-month trust + specialty performance rows."""

    filtered = _incomplete_pathways(dataframe)
    filtered = filtered[
        filtered[Columns.TREATMENT_FUNCTION_CODE] != ALL_SPECIALTIES_CODE
    ]
    filtered = filtered[filtered[Columns.PERIOD] == latest_period_value(dataframe)]

    trust_specialty = (
        filtered.groupby(
            [
                Columns.PROVIDER_ORG_CODE,
                Columns.PROVIDER_ORG_NAME,
                Columns.COMMISSIONER_ORG_NAME,
                Columns.TREATMENT_FUNCTION_CODE,
                Columns.TREATMENT_FUNCTION_NAME,
            ]
        )[[Columns.TOTAL_WAITING, Columns.WITHIN_18_WEEKS, Columns.OVER_52_WEEKS]]
        .sum()
        .reset_index()
        .sort_values([Columns.TREATMENT_FUNCTION_NAME, Columns.PROVIDER_ORG_NAME])
    )

    return add_performance_columns(
        trust_specialty,
        estimate_source=ESTIMATE_SOURCE_SPECIALTY,
    )


def classify_recent_trend(
    specialty_history: pd.DataFrame,
    value_column: str = Columns.TOTAL_WAITING,
    window: int = RECENT_TREND_WINDOW,
    threshold_pct: float = RECENT_TREND_THRESHOLD_PCT,
) -> str:
    """Classify the recent specialty trend using a simple rolling change."""

    history = specialty_history.sort_values(Columns.PERIOD_DT).tail(window)
    if len(history) < 2:
        return "stable"

    starting_value = max(float(history[value_column].iloc[0]), 1.0)
    ending_value = float(history[value_column].iloc[-1])
    pct_change = ((ending_value - starting_value) / starting_value) * 100

    if pct_change > threshold_pct:
        return "increasing"
    if pct_change < -threshold_pct:
        return "decreasing"
    return "stable"
