"""Patient-facing specialty suggestion and wait-estimate logic."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from src.config import (
    Columns,
    ESTIMATE_SOURCE_SPECIALTY,
    ESTIMATE_SOURCE_TRUST_WIDE,
    MIN_PROXY_WAIT_WEEKS,
    PATIENT_GUIDANCE,
    SPECIALTY_KEYWORDS,
    URGENCY_MULTIPLIERS,
)
from src.utils import estimate_source_label


@dataclass(frozen=True)
class SpecialtySuggestion:
    """A rule-based specialty suggestion derived from symptom keywords."""

    specialty: str | None
    matched_keywords: tuple[str, ...]
    score: int


@dataclass(frozen=True)
class WaitEstimate:
    """Resolved trust-level estimate used in the patient tab."""

    provider_org_name: str
    commissioner_org_name: str
    specialty: str
    pct_within_18_weeks: float
    total_waiting: float
    estimated_wait_weeks_proxy: int
    estimate_source: str

    @property
    def estimate_source_label(self) -> str:
        return estimate_source_label(self.estimate_source)


def _available_specialties(available_specialties: Iterable[str] | None) -> set[str] | None:
    if available_specialties is None:
        return None
    return {specialty for specialty in available_specialties if specialty}


def suggest_specialty_from_symptoms(
    symptoms_text: str | None,
    available_specialties: Iterable[str] | None = None,
) -> SpecialtySuggestion:
    """Suggest a specialty by scoring all keyword matches in the symptom text."""

    if not symptoms_text or not symptoms_text.strip():
        return SpecialtySuggestion(specialty=None, matched_keywords=(), score=0)

    normalised = symptoms_text.lower()
    permitted_specialties = _available_specialties(available_specialties)
    specialty_scores: dict[str, list[str]] = {}

    for specialty, keywords in SPECIALTY_KEYWORDS.items():
        if permitted_specialties is not None and specialty not in permitted_specialties:
            continue

        matches = sorted({keyword for keyword in keywords if keyword in normalised})
        if matches:
            specialty_scores[specialty] = matches

    if not specialty_scores:
        return SpecialtySuggestion(specialty=None, matched_keywords=(), score=0)

    best_specialty, matched_keywords = max(
        specialty_scores.items(),
        key=lambda item: (len(item[1]), sum(len(keyword) for keyword in item[1]), item[0]),
    )
    return SpecialtySuggestion(
        specialty=best_specialty,
        matched_keywords=tuple(matched_keywords),
        score=len(matched_keywords),
    )


def apply_urgency_adjustment(
    estimated_wait_weeks_proxy: int,
    urgency: str,
) -> int:
    """Apply a simple rule-based urgency multiplier to the wait proxy."""

    multiplier = URGENCY_MULTIPLIERS.get(urgency, 1.0)
    return max(MIN_PROXY_WAIT_WEEKS, round(estimated_wait_weeks_proxy * multiplier))


def resolve_wait_estimate(
    trust_name: str,
    region: str,
    specialty: str,
    trust_summary: pd.DataFrame,
    trust_specialty_summary: pd.DataFrame,
) -> WaitEstimate:
    """Return the best available estimate for a trust and specialty."""

    specialty_row = trust_specialty_summary[
        (trust_specialty_summary[Columns.PROVIDER_ORG_NAME] == trust_name)
        & (trust_specialty_summary[Columns.COMMISSIONER_ORG_NAME] == region)
        & (trust_specialty_summary[Columns.TREATMENT_FUNCTION_NAME] == specialty)
    ]
    if not specialty_row.empty:
        row = specialty_row.iloc[0]
        source = ESTIMATE_SOURCE_SPECIALTY
    else:
        trust_row = trust_summary[
            (trust_summary[Columns.PROVIDER_ORG_NAME] == trust_name)
            & (trust_summary[Columns.COMMISSIONER_ORG_NAME] == region)
        ]
        if trust_row.empty:
            raise ValueError(
                f"Could not find a trust summary row for '{trust_name}' in '{region}'."
            )
        row = trust_row.iloc[0]
        source = ESTIMATE_SOURCE_TRUST_WIDE

    return WaitEstimate(
        provider_org_name=str(row[Columns.PROVIDER_ORG_NAME]),
        commissioner_org_name=str(row[Columns.COMMISSIONER_ORG_NAME]),
        specialty=specialty,
        pct_within_18_weeks=float(row[Columns.PCT_WITHIN_18_WEEKS]),
        total_waiting=float(row[Columns.TOTAL_WAITING]),
        estimated_wait_weeks_proxy=int(row[Columns.ESTIMATED_WAIT_WEEKS_PROXY]),
        estimate_source=source,
    )


def build_region_alternatives(
    region: str,
    specialty: str,
    current_trust: str,
    trust_summary: pd.DataFrame,
    trust_specialty_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Build specialty-aware alternatives for trusts in the selected region."""

    region_trusts = trust_summary[
        trust_summary[Columns.COMMISSIONER_ORG_NAME] == region
    ].copy()
    region_trusts = region_trusts[
        region_trusts[Columns.PROVIDER_ORG_NAME] != current_trust
    ].copy()

    region_specialty = trust_specialty_summary[
        (trust_specialty_summary[Columns.COMMISSIONER_ORG_NAME] == region)
        & (trust_specialty_summary[Columns.TREATMENT_FUNCTION_NAME] == specialty)
    ][
        [
            Columns.PROVIDER_ORG_CODE,
            Columns.PROVIDER_ORG_NAME,
            Columns.COMMISSIONER_ORG_NAME,
            Columns.TOTAL_WAITING,
            Columns.PCT_WITHIN_18_WEEKS,
            Columns.ESTIMATED_WAIT_WEEKS_PROXY,
        ]
    ].copy()

    region_specialty = region_specialty.rename(
        columns={
            Columns.TOTAL_WAITING: "specialty_total_waiting",
            Columns.PCT_WITHIN_18_WEEKS: "specialty_pct_within_18_weeks",
            Columns.ESTIMATED_WAIT_WEEKS_PROXY: "specialty_estimated_wait_weeks_proxy",
        }
    )

    alternatives = region_trusts.merge(
        region_specialty,
        on=[
            Columns.PROVIDER_ORG_CODE,
            Columns.PROVIDER_ORG_NAME,
            Columns.COMMISSIONER_ORG_NAME,
        ],
        how="left",
    )

    specialty_available = alternatives["specialty_estimated_wait_weeks_proxy"].notna()
    alternatives[Columns.ESTIMATE_SOURCE] = specialty_available.map(
        {
            True: ESTIMATE_SOURCE_SPECIALTY,
            False: ESTIMATE_SOURCE_TRUST_WIDE,
        }
    )
    alternatives[Columns.ESTIMATE_SOURCE_LABEL] = alternatives[Columns.ESTIMATE_SOURCE].map(
        estimate_source_label
    )
    alternatives[Columns.TOTAL_WAITING] = alternatives[
        "specialty_total_waiting"
    ].combine_first(alternatives[Columns.TOTAL_WAITING])
    alternatives[Columns.PCT_WITHIN_18_WEEKS] = alternatives[
        "specialty_pct_within_18_weeks"
    ].combine_first(alternatives[Columns.PCT_WITHIN_18_WEEKS])
    alternatives[Columns.ESTIMATED_WAIT_WEEKS_PROXY] = alternatives[
        "specialty_estimated_wait_weeks_proxy"
    ].combine_first(alternatives[Columns.ESTIMATED_WAIT_WEEKS_PROXY])

    alternatives = alternatives.sort_values(
        [
            Columns.ESTIMATED_WAIT_WEEKS_PROXY,
            Columns.PCT_WITHIN_18_WEEKS,
            Columns.PROVIDER_ORG_NAME,
        ],
        ascending=[True, False, True],
    ).reset_index(drop=True)

    return alternatives


def build_recommendation_sections(
    estimate: WaitEstimate,
    urgency: str,
    specialty_trend: str,
    adjusted_wait_weeks_proxy: int,
    alternatives: pd.DataFrame,
    latest_period: str,
) -> dict[str, list[str]]:
    """Build transparent recommendation text for the patient tab."""

    data_based_findings = [
        (
            f"Latest loaded RTT month: {latest_period}. {estimate.provider_org_name} in "
            f"{estimate.commissioner_org_name} is currently reporting "
            f"{estimate.pct_within_18_weeks:.1f}% of incomplete pathways within 18 weeks."
        ),
    ]

    if estimate.estimate_source == ESTIMATE_SOURCE_SPECIALTY:
        data_based_findings.append(
            f"The wait proxy uses the latest trust + specialty row for {estimate.specialty}."
        )
    else:
        data_based_findings.append(
            f"No latest trust + specialty row was available for {estimate.specialty}, so the "
            "current estimate falls back to trust-wide incomplete-pathway performance."
        )

    if specialty_trend == "increasing":
        data_based_findings.append(
            f"The recent {estimate.specialty} waiting-list trend is increasing."
        )
    elif specialty_trend == "decreasing":
        data_based_findings.append(
            f"The recent {estimate.specialty} waiting-list trend is decreasing."
        )
    else:
        data_based_findings.append(
            f"The recent {estimate.specialty} waiting-list trend is broadly stable."
        )

    better_alternatives = alternatives[
        alternatives[Columns.ESTIMATED_WAIT_WEEKS_PROXY]
        < estimate.estimated_wait_weeks_proxy
    ].head(3)
    if better_alternatives.empty:
        data_based_findings.append(
            "No shorter-wait alternative trust was identified in the selected region from the latest data."
        )
    else:
        top_alternative = better_alternatives.iloc[0]
        data_based_findings.append(
            f"The shortest current regional alternative shown is {top_alternative[Columns.PROVIDER_ORG_NAME]} "
            f"at about {int(top_alternative[Columns.ESTIMATED_WAIT_WEEKS_PROXY])} proxy weeks "
            f"({top_alternative[Columns.ESTIMATE_SOURCE_LABEL]})."
        )

    heuristic_assumptions = [
        (
            f"The displayed wait is a proxy estimate of about {adjusted_wait_weeks_proxy} weeks "
            "after applying a simple urgency adjustment."
        ),
        (
            "This proxy is derived from RTT performance metrics and is not a true median, "
            "not a patient-level forecast, and not a guaranteed appointment date."
        ),
    ]

    if urgency == "Urgent":
        heuristic_assumptions.append(PATIENT_GUIDANCE["urgent"])
    elif urgency == "Two-Week Wait (Cancer Pathway)":
        heuristic_assumptions.append(
            "The two-week-wait pathway is shown using a very strong rule-based reduction for context only."
        )

    general_guidance = [PATIENT_GUIDANCE["right_to_choose"]]
    if urgency == "Two-Week Wait (Cancer Pathway)":
        general_guidance.append(PATIENT_GUIDANCE["two_week_wait"])
    else:
        general_guidance.append(PATIENT_GUIDANCE["routine"])

    general_guidance.append(
        "Use this dashboard as planning support only and confirm any referral decisions with your GP or care team."
    )

    return {
        "Data-based findings": data_based_findings,
        "Heuristic assumptions": heuristic_assumptions,
        "General NHS guidance": general_guidance,
    }
