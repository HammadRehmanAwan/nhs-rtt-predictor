"""Shared utility helpers."""

from __future__ import annotations

from html import escape

import numpy as np
import pandas as pd

from src.config import (
    ESTIMATE_SOURCE_LABELS,
    MAX_PROXY_WAIT_WEEKS,
    MIN_PROXY_WAIT_WEEKS,
    NHS_COLORS,
)


def calculate_percentage(
    numerator: pd.Series,
    denominator: pd.Series,
    digits: int = 1,
) -> pd.Series:
    """Safely calculate a percentage series."""

    percentage = numerator.div(denominator.replace({0: np.nan})).mul(100)
    return percentage.round(digits)


def estimate_wait_weeks_proxy(
    pct_within_18: pd.Series | float | int | None,
) -> pd.Series | int | None:
    """Convert 18-week performance into a clearly-labeled wait proxy heuristic."""

    if pct_within_18 is None:
        return None

    if isinstance(pct_within_18, pd.Series):
        clipped = pct_within_18.clip(lower=0, upper=100)
        proxy = ((1 - clipped / 100.0) * 52 + 4).round()
        proxy = proxy.clip(lower=MIN_PROXY_WAIT_WEEKS, upper=MAX_PROXY_WAIT_WEEKS)
        return proxy.astype("Int64")

    clipped_value = max(0.0, min(float(pct_within_18), 100.0))
    proxy = round((1 - clipped_value / 100.0) * 52 + 4)
    return int(max(MIN_PROXY_WAIT_WEEKS, min(proxy, MAX_PROXY_WAIT_WEEKS)))


def format_compact_number(value: float | int | None) -> str:
    """Format a number for compact dashboard display."""

    if value is None or pd.isna(value):
        return "—"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"{value / 1_000:.0f}K"
    return str(int(round(value)))


def format_period_range(start: pd.Timestamp, end: pd.Timestamp) -> str:
    """Return a readable date range for the dashboard header."""

    if pd.isna(start) or pd.isna(end):
        return "available data"
    return f"{start.strftime('%b %Y')} to {end.strftime('%b %Y')}"


def performance_color(pct_within_18: float | int | None) -> str:
    """Map 18-week performance to the dashboard's traffic-light palette."""

    if pct_within_18 is None or pd.isna(pct_within_18):
        return NHS_COLORS["blue"]
    if pct_within_18 >= 70:
        return NHS_COLORS["green"]
    if pct_within_18 >= 50:
        return NHS_COLORS["yellow"]
    return NHS_COLORS["red"]


def estimate_source_label(source_code: str | None) -> str:
    """Return a user-facing label for a wait estimate source."""

    if not source_code:
        return "Unknown estimate source"
    return ESTIMATE_SOURCE_LABELS.get(source_code, source_code.replace("_", " ").title())


def render_bullet_list(items: list[str]) -> str:
    """Render a list of strings as simple HTML bullets."""

    bullet_items = "".join(f"<li>{escape(item)}</li>" for item in items)
    return f"<ul>{bullet_items}</ul>"
