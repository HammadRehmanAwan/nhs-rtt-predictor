from __future__ import annotations

import warnings
from html import escape

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.aggregations import (
    build_latest_trust_specialty_summary,
    build_latest_trust_summary,
    build_national_trend,
    build_specialty_trend,
    classify_recent_trend,
    latest_period_value,
)
from src.config import (
    APP_PAGE_CONFIG,
    APP_SUBTITLE_TEMPLATE,
    APP_TITLE,
    Columns,
    DISCLAIMER_TEXT,
    ESTIMATE_SOURCE_SPECIALTY,
    FORECAST_MIN_HISTORY_POINTS,
    FULL_HISTORY_TARGET_MONTHS,
    GLOBAL_STYLES,
    NATIONAL_TARGETS,
    NHS_COLORS,
    PATIENT_PROXY_EXPLANATION,
)
from src.data_loader import (
    choose_fullest_dataset,
    DataLoadError,
    DataValidationError,
    DataLoadResult,
    load_google_drive_dataset,
    load_rtt_data,
)
from src.forecasting import ForecastResult, forecast_waiting_list
from src.patient_logic import (
    apply_urgency_adjustment,
    build_recommendation_sections,
    build_region_alternatives,
    resolve_wait_estimate,
    suggest_specialty_from_symptoms,
)
from src.utils import (
    estimate_source_label,
    format_compact_number,
    format_period_range,
    performance_color,
    render_bullet_list,
)

warnings.filterwarnings("ignore")

st.set_page_config(**APP_PAGE_CONFIG)
st.markdown(GLOBAL_STYLES, unsafe_allow_html=True)

FORECAST_METHOD_LABELS = {
    "prophet": "Prophet",
    "moving_average_fallback": "Moving-average baseline",
    "last_value_fallback": "Last-value baseline",
}


def get_kaggle_credentials_from_streamlit() -> dict[str, str] | None:
    """Read optional Kaggle credentials from Streamlit secrets."""

    try:
        kaggle_config = st.secrets.get("kaggle", {})
    except Exception:
        return None

    username = kaggle_config.get("username")
    key = kaggle_config.get("key")
    if username and key:
        return {"username": username, "key": key}
    return None


@st.cache_data(show_spinner="Loading NHS RTT dataset...")
def load_dashboard_data(
    kaggle_username: str | None,
    kaggle_key: str | None,
) -> DataLoadResult:
    local_result: DataLoadResult | None = None
    drive_result: DataLoadResult | None = None
    kaggle_result: DataLoadResult | None = None

    try:
        local_result = load_rtt_data(
            allow_kaggle_download=False,
            kaggle_credentials=None,
        )
    except DataLoadError:
        local_result = None

    if local_result is None or local_result.period_count < FULL_HISTORY_TARGET_MONTHS:
        try:
            drive_result = load_google_drive_dataset()
        except (DataLoadError, DataValidationError):
            drive_result = None

    if kaggle_username and kaggle_key:
        try:
            kaggle_result = load_rtt_data(
                dataset_paths=(),
                allow_kaggle_download=True,
                kaggle_credentials={"username": kaggle_username, "key": kaggle_key},
            )
        except DataLoadError:
            kaggle_result = None

    preferred = choose_fullest_dataset([local_result, drive_result, kaggle_result])
    if preferred is None:
        raise DataLoadError(
            "No RTT dataset could be loaded from the local CSV, Google Drive, or optional Kaggle fallback."
        )
    return preferred


@st.cache_data(show_spinner="Preparing dashboard views...")
def build_dashboard_views(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    national = build_national_trend(dataframe)
    specialty = build_specialty_trend(dataframe)
    trust_summary = build_latest_trust_summary(dataframe)
    trust_specialty = build_latest_trust_specialty_summary(dataframe)
    return national, specialty, trust_summary, trust_specialty


@st.cache_data(show_spinner="Running forecast...")
def build_forecast(series: pd.DataFrame) -> ForecastResult:
    return forecast_waiting_list(series)


def metric_card(label: str, value: str, subtext: str, color: str) -> str:
    return f"""
    <div class='metric-card'>
        <div class='metric-label'>{escape(label)}</div>
        <div class='metric-value' style='color:{color}'>{escape(value)}</div>
        <div class='metric-sub'>{escape(subtext)}</div>
    </div>
    """


def wait_card(
    title: str,
    wait_weeks: int,
    pct_within_18: float,
    source_label: str,
    css_class: str,
    saving_weeks: int | None = None,
) -> str:
    saving_html = ""
    if saving_weeks is not None:
        saving_html = (
            f"<p style='font-size:11px;color:{NHS_COLORS['green']};margin:6px 0 0'>"
            f"Save about {saving_weeks} weeks</p>"
        )

    return f"""
    <div class='mini-card {css_class}'>
        <p style='font-size:11px;color:#6B7280;margin:0'><b>{escape(title[:60])}</b></p>
        <p style='font-size:30px;font-weight:700;color:{NHS_COLORS['blue'] if css_class == 'primary' else NHS_COLORS['green']};margin:6px 0'>
            ~{wait_weeks} wks
        </p>
        <p style='font-size:11px;color:#6B7280;margin:0'>{pct_within_18:.1f}% within 18 weeks</p>
        <p style='font-size:11px;color:#6B7280;margin:6px 0 0'>{escape(source_label)}</p>
        {saving_html}
    </div>
    """


def render_header(period_range: str, source_label: str) -> None:
    subtitle = APP_SUBTITLE_TEMPLATE.format(period_range=period_range)
    st.markdown(
        f"""
        <div class='hero-card'>
            <h2 class='hero-title'>🏥 {escape(APP_TITLE)}</h2>
            <p class='hero-subtitle'>{escape(subtitle)} · Source: {escape(source_label)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_forecast_caption(forecast_result: ForecastResult) -> None:
    method_label = FORECAST_METHOD_LABELS.get(forecast_result.method, forecast_result.method)
    caption_parts = [f"Forecast method: {method_label}"]
    if forecast_result.mae is not None:
        caption_parts.append(f"Holdout MAE: {forecast_result.mae:,.0f}")
    if forecast_result.mape is not None:
        caption_parts.append(f"Holdout MAPE: {forecast_result.mape:.1f}%")
    st.caption(" | ".join(caption_parts))
    if forecast_result.warning:
        st.caption(forecast_result.warning)


def render_recommendation_box(sections: dict[str, list[str]]) -> None:
    html_sections = "".join(
        f"<h4 style='margin:12px 0 6px;color:{NHS_COLORS['ink']}'>{escape(title)}</h4>{render_bullet_list(items)}"
        for title, items in sections.items()
    )
    st.markdown(
        f"""
        <div class='advice-box'>
            <div style='display:flex;align-items:center;gap:8px;margin-bottom:6px'>
                <span style='font-size:20px'>📋</span>
                <span style='font-weight:700;font-size:15px;color:{NHS_COLORS['ink']}'>
                    Transparent patient recommendation summary
                </span>
            </div>
            {html_sections}
        </div>
        """,
        unsafe_allow_html=True,
    )


def highlight_performance(value: float) -> str:
    return f"color:{performance_color(value)};font-weight:bold"


credentials = get_kaggle_credentials_from_streamlit()
try:
    data_result = load_dashboard_data(
        credentials["username"] if credentials else None,
        credentials["key"] if credentials else None,
    )
    df = data_result.dataframe
    national, specialty_data, trust_data, trust_specialty_data = build_dashboard_views(df)
except (DataLoadError, DataValidationError) as exc:
    st.error(
        f"Could not load the NHS RTT dataset.\n\n{exc}\n\n"
        "Place the CSV next to `app.py` for the most reliable setup. "
        "Optional Kaggle credentials can be supplied in Streamlit secrets under `kaggle.username` and `kaggle.key`."
    )
    st.stop()

latest_period = latest_period_value(df)
period_range = format_period_range(
    national[Columns.PERIOD_DT].min(),
    national[Columns.PERIOD_DT].max(),
)
latest_national = national.sort_values(Columns.PERIOD_DT).iloc[-1]
national_forecast = build_forecast(national[[Columns.PERIOD_DT, Columns.TOTAL_WAITING]])
national_future = national_forecast.future
all_specialties = sorted(specialty_data[Columns.TREATMENT_FUNCTION_NAME].dropna().unique())

render_header(period_range=period_range, source_label=data_result.source_label)
st.caption(f"Loaded from `{data_result.source_path}`")

forecast_12_month = national_future["yhat"].iloc[-1] if not national_future.empty else None

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(
        metric_card(
            "Total Waiting (Latest)",
            format_compact_number(latest_national[Columns.TOTAL_WAITING]),
            "Incomplete pathways",
            NHS_COLORS["blue"],
        ),
        unsafe_allow_html=True,
    )
with c2:
    pct_value = float(latest_national[Columns.PCT_WITHIN_18_WEEKS])
    st.markdown(
        metric_card(
            "% Within 18 Weeks",
            f"{pct_value:.1f}%",
            f"Target: {NATIONAL_TARGETS['constitutional_pct_within_18']:.0f}%",
            performance_color(pct_value),
        ),
        unsafe_allow_html=True,
    )
with c3:
    st.markdown(
        metric_card(
            "Over 52 Weeks",
            format_compact_number(latest_national[Columns.OVER_52_WEEKS]),
            "Long waits",
            NHS_COLORS["red"],
        ),
        unsafe_allow_html=True,
    )
with c4:
    outlook_subtext = FORECAST_METHOD_LABELS.get(national_forecast.method, national_forecast.method)
    st.markdown(
        metric_card(
            "12-Month Outlook",
            format_compact_number(forecast_12_month),
            outlook_subtext,
            NHS_COLORS["yellow"],
        ),
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📊 National Trend", "🔬 By Specialty", "👤 Patient Predictor"])

with tab1:
    metric = st.radio(
        "View",
        ["Waiting List Size", "% Within 18 Weeks"],
        horizontal=True,
        label_visibility="collapsed",
    )

    figure = go.Figure()

    if metric == "Waiting List Size":
        figure.add_trace(
            go.Scatter(
                x=national[Columns.PERIOD_DT],
                y=national[Columns.TOTAL_WAITING] / 1_000_000,
                name="Historical",
                line=dict(color=NHS_COLORS["blue"], width=2.5),
                fill="tozeroy",
                fillcolor="rgba(0,48,135,0.08)",
            )
        )
        figure.add_trace(
            go.Scatter(
                x=national_future["ds"],
                y=national_future["yhat"] / 1_000_000,
                name="Forecast",
                line=dict(color=NHS_COLORS["green"], width=2.5, dash="dash"),
            )
        )
        figure.add_traces(
            [
                go.Scatter(
                    x=national_future["ds"],
                    y=national_future["yhat_upper"] / 1_000_000,
                    fill=None,
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                ),
                go.Scatter(
                    x=national_future["ds"],
                    y=national_future["yhat_lower"] / 1_000_000,
                    fill="tonexty",
                    mode="lines",
                    line=dict(width=0),
                    fillcolor="rgba(0,178,148,0.15)",
                    name="Forecast range",
                ),
            ]
        )
        figure.update_yaxes(title="Patients (Millions)")
    else:
        figure.add_trace(
            go.Scatter(
                x=national[Columns.PERIOD_DT],
                y=national[Columns.PCT_WITHIN_18_WEEKS],
                name="% Within 18 Weeks",
                line=dict(color=NHS_COLORS["blue"], width=2.5),
            )
        )
        figure.add_hline(
            y=NATIONAL_TARGETS["constitutional_pct_within_18"],
            line_dash="dash",
            line_color=NHS_COLORS["red"],
            annotation_text="92% constitutional target",
        )
        figure.add_hline(
            y=NATIONAL_TARGETS["interim_pct_within_18"],
            line_dash="dot",
            line_color=NHS_COLORS["yellow"],
            annotation_text="65% interim target",
        )
        figure.update_yaxes(title="Percentage (%)", range=[40, 100])

    figure.update_layout(
        height=360,
        template="plotly_white",
        title="National Waiting List — Historical and 12-Month Outlook",
        legend=dict(orientation="h", y=-0.15),
    )
    st.plotly_chart(figure, width="stretch")
    render_forecast_caption(national_forecast)

    figure_over_52 = go.Figure(
        go.Bar(
            x=national[Columns.PERIOD_DT],
            y=national[Columns.OVER_52_WEEKS] / 1_000,
            marker_color=NHS_COLORS["red"],
            opacity=0.8,
        )
    )
    figure_over_52.update_layout(
        height=250,
        template="plotly_white",
        title="Patients Waiting Over 52 Weeks (Thousands)",
        yaxis_title="Patients (K)",
    )
    st.plotly_chart(figure_over_52, width="stretch")

with tab2:
    selected_specialty = st.selectbox("Select specialty", all_specialties)

    specialty_history = specialty_data[
        specialty_data[Columns.TREATMENT_FUNCTION_NAME] == selected_specialty
    ].copy()
    specialty_forecast = build_forecast(
        specialty_history[[Columns.PERIOD_DT, Columns.TOTAL_WAITING]]
    )

    specialty_figure = go.Figure()
    specialty_figure.add_trace(
        go.Scatter(
            x=specialty_history[Columns.PERIOD_DT],
            y=specialty_history[Columns.TOTAL_WAITING],
            name="Historical",
            line=dict(color=NHS_COLORS["blue"], width=2.5),
            fill="tozeroy",
            fillcolor="rgba(0,48,135,0.08)",
        )
    )
    specialty_figure.add_trace(
        go.Scatter(
            x=specialty_forecast.future["ds"],
            y=specialty_forecast.future["yhat"],
            name="Forecast",
            line=dict(color=NHS_COLORS["green"], width=2.5, dash="dash"),
        )
    )
    specialty_figure.add_traces(
        [
            go.Scatter(
                x=specialty_forecast.future["ds"],
                y=specialty_forecast.future["yhat_upper"],
                fill=None,
                mode="lines",
                line=dict(width=0),
                showlegend=False,
            ),
            go.Scatter(
                x=specialty_forecast.future["ds"],
                y=specialty_forecast.future["yhat_lower"],
                fill="tonexty",
                mode="lines",
                line=dict(width=0),
                fillcolor="rgba(0,178,148,0.15)",
                name="Forecast range",
            ),
        ]
    )
    specialty_figure.update_layout(
        height=340,
        template="plotly_white",
        title=f"{selected_specialty} — Waiting List and 12-Month Outlook",
        yaxis_title="Patients",
        legend=dict(orientation="h", y=-0.15),
    )
    st.plotly_chart(specialty_figure, width="stretch")
    render_forecast_caption(specialty_forecast)

    st.subheader("Trust Performance — Latest Month")
    specialty_trust_table = trust_specialty_data[
        trust_specialty_data[Columns.TREATMENT_FUNCTION_NAME] == selected_specialty
    ].copy()

    if specialty_trust_table.empty:
        st.caption(
            "No latest-month trust + specialty rows were available for this specialty. "
            "Showing trust-wide latest-month performance instead."
        )
        display_trusts = trust_data[
            [
                Columns.PROVIDER_ORG_NAME,
                Columns.COMMISSIONER_ORG_NAME,
                Columns.TOTAL_WAITING,
                Columns.PCT_WITHIN_18_WEEKS,
                Columns.ESTIMATED_WAIT_WEEKS_PROXY,
                Columns.ESTIMATE_SOURCE_LABEL,
            ]
        ].rename(
            columns={
                Columns.PROVIDER_ORG_NAME: "Trust",
                Columns.COMMISSIONER_ORG_NAME: "Region / ICB",
                Columns.TOTAL_WAITING: "Total Waiting",
                Columns.PCT_WITHIN_18_WEEKS: "% Within 18 Wks",
                Columns.ESTIMATED_WAIT_WEEKS_PROXY: "Wait Proxy (wks)",
                Columns.ESTIMATE_SOURCE_LABEL: "Estimate Basis",
            }
        )
    else:
        st.caption(
            "Wait estimate is based on the latest trust + specialty RTT performance data."
        )
        display_trusts = specialty_trust_table[
            [
                Columns.PROVIDER_ORG_NAME,
                Columns.COMMISSIONER_ORG_NAME,
                Columns.TOTAL_WAITING,
                Columns.PCT_WITHIN_18_WEEKS,
                Columns.ESTIMATED_WAIT_WEEKS_PROXY,
            ]
        ].rename(
            columns={
                Columns.PROVIDER_ORG_NAME: "Trust",
                Columns.COMMISSIONER_ORG_NAME: "Region / ICB",
                Columns.TOTAL_WAITING: "Total Waiting",
                Columns.PCT_WITHIN_18_WEEKS: "% Within 18 Wks",
                Columns.ESTIMATED_WAIT_WEEKS_PROXY: "Wait Proxy (wks)",
            }
        )

    display_trusts = display_trusts.sort_values("Wait Proxy (wks)")
    st.dataframe(
        display_trusts.style.map(highlight_performance, subset=["% Within 18 Wks"]),
        width="stretch",
        height=400,
    )

with tab3:
    st.markdown("### 👤 Patient Wait Time Predictor")
    st.caption(
        "Use the latest RTT data to estimate likely wait pressures and compare alternative trusts in your region."
    )

    symptoms = st.text_area(
        "Describe your symptoms (optional — used only for a rule-based specialty suggestion)",
        placeholder="e.g. knee pain when walking, referred by GP for orthopaedic assessment...",
    )

    specialty_suggestion = suggest_specialty_from_symptoms(symptoms, all_specialties)
    if specialty_suggestion.specialty:
        keyword_text = ", ".join(specialty_suggestion.matched_keywords)
        st.caption(
            f"Suggested specialty from symptom keywords: {specialty_suggestion.specialty}"
            f" ({keyword_text})"
        )

    left_col, right_col = st.columns(2)
    with left_col:
        specialty_options = [""] + all_specialties
        default_index = (
            specialty_options.index(specialty_suggestion.specialty)
            if specialty_suggestion.specialty in specialty_options
            else 0
        )
        selected_patient_specialty = st.selectbox(
            "Referred Specialty *",
            specialty_options,
            index=default_index,
        )

        all_regions = sorted(
            trust_data[Columns.COMMISSIONER_ORG_NAME].dropna().unique().tolist()
        )
        selected_region = st.selectbox("Your Region / ICB *", [""] + all_regions)

    with right_col:
        urgency = st.selectbox(
            "Referral Urgency *",
            ["", "Routine", "Urgent", "Two-Week Wait (Cancer Pathway)"],
        )

        if selected_region:
            region_trust_list = sorted(
                trust_data[
                    trust_data[Columns.COMMISSIONER_ORG_NAME] == selected_region
                ][Columns.PROVIDER_ORG_NAME]
                .dropna()
                .unique()
                .tolist()
            )
        else:
            region_trust_list = []

        selected_trust = st.selectbox(
            "Your NHS Trust *",
            [""] + region_trust_list,
            disabled=not selected_region,
        )

    submitted = st.button(
        "🔮 Predict My Wait & Find Alternatives",
        width="stretch",
        type="primary",
    )

    if submitted:
        if not selected_patient_specialty or not urgency or not selected_region or not selected_trust:
            st.warning(
                "Please fill in Specialty, Urgency, Region and Trust before predicting."
            )
        else:
            try:
                current_estimate = resolve_wait_estimate(
                    trust_name=selected_trust,
                    region=selected_region,
                    specialty=selected_patient_specialty,
                    trust_summary=trust_data,
                    trust_specialty_summary=trust_specialty_data,
                )
            except ValueError as exc:
                st.error(str(exc))
                st.stop()

            alternatives = build_region_alternatives(
                region=selected_region,
                specialty=selected_patient_specialty,
                current_trust=selected_trust,
                trust_summary=trust_data,
                trust_specialty_summary=trust_specialty_data,
            )

            specialty_history_filtered = specialty_data[
                specialty_data[Columns.TREATMENT_FUNCTION_NAME]
                == selected_patient_specialty
            ].copy()
            specialty_trend = classify_recent_trend(specialty_history_filtered)
            current_adjusted_wait = apply_urgency_adjustment(
                current_estimate.estimated_wait_weeks_proxy,
                urgency,
            )

            if current_estimate.estimate_source != ESTIMATE_SOURCE_SPECIALTY:
                st.caption(
                    f"No latest trust + specialty row was available for {selected_trust} / "
                    f"{selected_patient_specialty}. The current estimate uses trust-wide performance instead."
                )

            specialty_based_count = int(
                (alternatives[Columns.ESTIMATE_SOURCE] == ESTIMATE_SOURCE_SPECIALTY).sum()
            )
            fallback_count = int(len(alternatives) - specialty_based_count)
            if len(alternatives) > 0:
                if specialty_based_count == 0:
                    st.caption(
                        "Regional alternatives use trust-wide fallback estimates because "
                        "no latest trust + specialty rows were available in this region."
                    )
                elif fallback_count > 0:
                    st.caption(
                        f"{specialty_based_count} regional alternatives use trust + specialty data and "
                        f"{fallback_count} use trust-wide fallback estimates where specialty rows were missing."
                    )

            display_alternatives = alternatives.copy()
            display_alternatives[Columns.ADJUSTED_WAIT_WEEKS_PROXY] = display_alternatives[
                Columns.ESTIMATED_WAIT_WEEKS_PROXY
            ].apply(lambda value: apply_urgency_adjustment(int(value), urgency))

            better_alternatives = display_alternatives[
                display_alternatives[Columns.ADJUSTED_WAIT_WEEKS_PROXY]
                < current_adjusted_wait
            ].head(3)

            st.markdown("#### 📅 Estimated Wait Proxy")
            cards = st.columns([1.2] + [1] * min(3, len(better_alternatives)))
            with cards[0]:
                st.markdown(
                    wait_card(
                        title=current_estimate.provider_org_name,
                        wait_weeks=current_adjusted_wait,
                        pct_within_18=current_estimate.pct_within_18_weeks,
                        source_label=current_estimate.estimate_source_label,
                        css_class="primary",
                    ),
                    unsafe_allow_html=True,
                )

            for index, (_, alternative) in enumerate(better_alternatives.iterrows(), start=1):
                alternative_wait = int(alternative[Columns.ADJUSTED_WAIT_WEEKS_PROXY])
                saving = max(0, current_adjusted_wait - alternative_wait)
                with cards[index]:
                    st.markdown(
                        wait_card(
                            title=str(alternative[Columns.PROVIDER_ORG_NAME]),
                            wait_weeks=alternative_wait,
                            pct_within_18=float(alternative[Columns.PCT_WITHIN_18_WEEKS]),
                            source_label=estimate_source_label(
                                str(alternative[Columns.ESTIMATE_SOURCE])
                            ),
                            css_class="positive",
                            saving_weeks=saving,
                        ),
                        unsafe_allow_html=True,
                    )

            st.caption(
                PATIENT_PROXY_EXPLANATION
            )

            recommendation_sections = build_recommendation_sections(
                estimate=current_estimate,
                urgency=urgency,
                specialty_trend=specialty_trend,
                adjusted_wait_weeks_proxy=current_adjusted_wait,
                alternatives=alternatives,
                latest_period=latest_period,
            )
            render_recommendation_box(recommendation_sections)

            st.caption(f"⚠️ {DISCLAIMER_TEXT}")

    st.markdown("---")
    info_1, info_2, info_3 = st.columns(3)
    with info_1:
        st.info(
            f"**📊 Real NHS Data**\n\n{national[Columns.PERIOD_DT].nunique()} monthly RTT snapshots loaded "
            f"({period_range})."
        )
    with info_2:
        st.info(
            "**📈 Safer Forecasting**\n\nProphet is used only when enough history is available, "
            "with baseline fallback and simple holdout evaluation."
        )
    with info_3:
        st.info(
            "**🏥 Right to Choose**\n\nAlternative trust suggestions are specialty-aware where the latest data allows."
        )

st.markdown("---")
if data_result.period_count < FULL_HISTORY_TARGET_MONTHS:
    st.caption(
        f"Current loaded history covers {data_result.period_count} monthly releases. "
        "If a fuller RTT file or Kaggle-backed refresh is available, the app will use it for stronger trends and forecasts."
    )
st.caption(
    f"Data: NHS England RTT Open Data · Source mode: {data_result.source_label} · {DISCLAIMER_TEXT}"
)
