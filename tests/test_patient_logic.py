from __future__ import annotations

from src.aggregations import build_latest_trust_specialty_summary, build_latest_trust_summary
from src.config import Columns, ESTIMATE_SOURCE_SPECIALTY, ESTIMATE_SOURCE_TRUST_WIDE
from src.patient_logic import (
    apply_urgency_adjustment,
    build_region_alternatives,
    resolve_wait_estimate,
    suggest_specialty_from_symptoms,
)


def test_suggest_specialty_from_symptoms_scores_multiple_matches() -> None:
    suggestion = suggest_specialty_from_symptoms(
        "Severe migraine headaches with occasional seizure symptoms",
        available_specialties=["Neurology Service", "Cardiology Service"],
    )

    assert suggestion.specialty == "Neurology Service"
    assert set(suggestion.matched_keywords) == {"headache", "migraine", "seizure"}
    assert suggestion.score == 3


def test_resolve_wait_estimate_prefers_specialty_and_falls_back(sample_rtt_df) -> None:
    trust_summary = build_latest_trust_summary(sample_rtt_df)
    trust_specialty = build_latest_trust_specialty_summary(sample_rtt_df)

    specialty_estimate = resolve_wait_estimate(
        trust_name="Alpha Trust",
        region="North ICB",
        specialty="Trauma and Orthopaedic Service",
        trust_summary=trust_summary,
        trust_specialty_summary=trust_specialty,
    )
    fallback_estimate = resolve_wait_estimate(
        trust_name="Bravo Trust",
        region="North ICB",
        specialty="Trauma and Orthopaedic Service",
        trust_summary=trust_summary,
        trust_specialty_summary=trust_specialty,
    )

    assert specialty_estimate.estimate_source == ESTIMATE_SOURCE_SPECIALTY
    assert specialty_estimate.estimated_wait_weeks_proxy == 17
    assert fallback_estimate.estimate_source == ESTIMATE_SOURCE_TRUST_WIDE
    assert fallback_estimate.estimated_wait_weeks_proxy == 30


def test_build_region_alternatives_mixes_specialty_and_fallback_data(sample_rtt_df) -> None:
    trust_summary = build_latest_trust_summary(sample_rtt_df)
    trust_specialty = build_latest_trust_specialty_summary(sample_rtt_df)

    alternatives = build_region_alternatives(
        region="North ICB",
        specialty="General Surgery Service",
        current_trust="Bravo Trust",
        trust_summary=trust_summary,
        trust_specialty_summary=trust_specialty,
    )

    alpha = alternatives[alternatives[Columns.PROVIDER_ORG_NAME] == "Alpha Trust"].iloc[0]
    charlie = alternatives[alternatives[Columns.PROVIDER_ORG_NAME] == "Charlie Trust"].iloc[0]

    assert alpha[Columns.ESTIMATE_SOURCE] == ESTIMATE_SOURCE_TRUST_WIDE
    assert alpha[Columns.ESTIMATED_WAIT_WEEKS_PROXY] == 18
    assert charlie[Columns.ESTIMATE_SOURCE] == ESTIMATE_SOURCE_SPECIALTY
    assert charlie[Columns.ESTIMATED_WAIT_WEEKS_PROXY] == 30


def test_apply_urgency_adjustment_scales_proxy() -> None:
    assert apply_urgency_adjustment(20, "Routine") == 20
    assert apply_urgency_adjustment(20, "Urgent") == 9
    assert apply_urgency_adjustment(20, "Two-Week Wait (Cancer Pathway)") == 2
