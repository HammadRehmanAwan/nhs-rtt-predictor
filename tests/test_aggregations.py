from __future__ import annotations

import pandas as pd

from src.aggregations import (
    build_latest_trust_specialty_summary,
    build_latest_trust_summary,
    build_national_trend,
)
from src.config import Columns


def test_build_national_trend_aggregates_trustwide_rows(sample_rtt_df) -> None:
    national = build_national_trend(sample_rtt_df)

    january = national[national[Columns.PERIOD_DT] == pd.Timestamp("2024-01-01")].iloc[0]
    february = national[national[Columns.PERIOD_DT] == pd.Timestamp("2024-02-01")].iloc[0]

    assert january[Columns.TOTAL_WAITING] == 170
    assert january[Columns.WITHIN_18_WEEKS] == 115
    assert january[Columns.PCT_WITHIN_18_WEEKS] == 67.6
    assert february[Columns.TOTAL_WAITING] == 320
    assert february[Columns.OVER_52_WEEKS] == 27


def test_build_latest_trust_summary_uses_latest_period_only(sample_rtt_df) -> None:
    trust_summary = build_latest_trust_summary(sample_rtt_df)
    alpha = trust_summary[trust_summary[Columns.PROVIDER_ORG_NAME] == "Alpha Trust"].iloc[0]

    assert alpha[Columns.TOTAL_WAITING] == 150
    assert alpha[Columns.WITHIN_18_WEEKS] == 110
    assert alpha[Columns.PCT_WITHIN_18_WEEKS] == 73.3
    assert alpha[Columns.ESTIMATED_WAIT_WEEKS_PROXY] == 18


def test_build_latest_trust_specialty_summary_groups_by_trust_and_specialty(sample_rtt_df) -> None:
    trust_specialty = build_latest_trust_specialty_summary(sample_rtt_df)
    alpha_ortho = trust_specialty[
        (trust_specialty[Columns.PROVIDER_ORG_NAME] == "Alpha Trust")
        & (
            trust_specialty[Columns.TREATMENT_FUNCTION_NAME]
            == "Trauma and Orthopaedic Service"
        )
    ].iloc[0]

    assert alpha_ortho[Columns.TOTAL_WAITING] == 80
    assert alpha_ortho[Columns.WITHIN_18_WEEKS] == 60
    assert alpha_ortho[Columns.PCT_WITHIN_18_WEEKS] == 75.0
    assert alpha_ortho[Columns.ESTIMATED_WAIT_WEEKS_PROXY] == 17
