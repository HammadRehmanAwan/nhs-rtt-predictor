# ============================================================
# NHS RTT Waiting List Predictor — Streamlit App
# ============================================================
# SETUP (run these once in terminal):
#   pip install streamlit pandas prophet plotly anthropic
#
# RUN:
#   streamlit run app.py
#
# PUT YOUR FILES IN THE SAME FOLDER:
#   app.py                            ← this file
#   nhs_rtt_waiting_times_2021_2025.csv  ← your Kaggle CSV
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from prophet import Prophet
import warnings
warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="NHS RTT Waiting List Predictor",
    page_icon="🏥",
    layout="wide",
)

# ── NHS colour palette ────────────────────────────────────────
NHS_BLUE   = "#003087"
NHS_GREEN  = "#00B294"
NHS_RED    = "#DA291C"
NHS_YELLOW = "#FAB900"

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #F5F7FA; }
    .block-container { padding-top: 1.5rem; }
    .metric-card {
        background: white;
        border-radius: 14px;
        padding: 18px;
        text-align: center;
        border: 1px solid #E5E7EB;
        box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    }
    .metric-label { font-size: 12px; color: #6B7280; margin-bottom: 4px; }
    .metric-value { font-size: 26px; font-weight: 700; }
    .metric-sub   { font-size: 11px; color: #9CA3AF; margin-top: 2px; }
    .ai-box {
        background: #EFF6FF;
        border-left: 4px solid #003087;
        border-radius: 12px;
        padding: 16px;
        margin-top: 12px;
    }
    .trust-card {
        background: white;
        border-radius: 12px;
        padding: 12px 16px;
        border: 1px solid #D1FAE5;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LOAD & CACHE DATA
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@st.cache_data(show_spinner="Loading NHS RTT dataset...")
def load_data():
    import os, json

    LOCAL_PATH = "nhs_rtt_waiting_times_2021_2025.csv"
    TMP_PATH   = "/tmp/nhs_rtt_waiting_times_2021_2025.csv"

    if os.path.exists(LOCAL_PATH):
        # ── Running locally ───────────────────────────────────────
        df = pd.read_csv(LOCAL_PATH, low_memory=False)

    elif os.path.exists(TMP_PATH):
        # ── Already downloaded in this session ────────────────────
        df = pd.read_csv(TMP_PATH, low_memory=False)

    else:
        # ── Running on Streamlit Cloud — download from Kaggle ─────
        try:
            # Write kaggle.json from Streamlit secrets
            kaggle_dir = os.path.expanduser("~/.kaggle")
            os.makedirs(kaggle_dir, exist_ok=True)
            kaggle_creds = {
                "username": st.secrets["kaggle"]["username"],
                "key":      st.secrets["kaggle"]["key"],
            }
            with open(f"{kaggle_dir}/kaggle.json", "w") as f:
                json.dump(kaggle_creds, f)
            os.chmod(f"{kaggle_dir}/kaggle.json", 0o600)

            # Download dataset
            import subprocess
            st.info("⏳ Downloading NHS dataset from Kaggle (first run only, ~30 seconds)...")
            subprocess.run([
                "kaggle", "datasets", "download",
                "-d", "hammad9191/nhs-consultant-led-rtt-waiting-times20212025",
                "--unzip", "-p", "/tmp"
            ], check=True, capture_output=True)

            # Find the CSV in /tmp
            csv_candidates = [f for f in os.listdir("/tmp") if f.endswith(".csv")]
            if not csv_candidates:
                st.error("Download succeeded but no CSV found in /tmp.")
                st.stop()

            downloaded = f"/tmp/{csv_candidates[0]}"
            df = pd.read_csv(downloaded, low_memory=False)

        except Exception as e:
            st.error(f"""
            ❌ Could not load dataset: {e}

            Make sure your Streamlit secrets are set correctly:
            ```
            [kaggle]
            username = "hammad9191"
            key = "your-kaggle-api-key"
            ```
            Go to: app settings → Secrets → paste the above.
            """)
            st.stop()
    df["period_dt"] = pd.to_datetime(df["period"], format="%Y-%m", errors="coerce")
    df = df.dropna(subset=["period_dt"])
    return df

@st.cache_data(show_spinner="Training forecast models...")
def build_national(df):
    nat = (
        df[(df["rtt_part_type"]=="Part_2") & (df["treatment_function_code"]=="C_999")]
        .groupby("period_dt")[["total_waiting","within_18_weeks","over_52_weeks"]]
        .sum().reset_index().sort_values("period_dt")
    )
    nat["pct_within_18"] = (nat["within_18_weeks"] / nat["total_waiting"] * 100).round(1)
    nat["pct_over_52"]   = (nat["over_52_weeks"]   / nat["total_waiting"] * 100).round(1)
    return nat

@st.cache_data(show_spinner="Training Prophet model...")
def run_prophet(series, periods=12):
    prophet_df = series.rename(columns={"period_dt":"ds","total_waiting":"y"})
    m = Prophet(yearly_seasonality=True, weekly_seasonality=False,
                daily_seasonality=False, changepoint_prior_scale=0.05,
                interval_width=0.95)
    m.fit(prophet_df[["ds","y"]].dropna())
    future   = m.make_future_dataframe(periods=periods, freq="MS")
    forecast = m.predict(future)
    return forecast[["ds","yhat","yhat_lower","yhat_upper"]]

@st.cache_data(show_spinner="Processing specialties...")
def build_specialty_data(df):
    spec = (
        df[(df["rtt_part_type"]=="Part_2") & (df["treatment_function_code"]!="C_999")]
        .groupby(["period_dt","treatment_function_name"])[
            ["total_waiting","within_18_weeks","over_52_weeks"]
        ].sum().reset_index()
    )
    spec["pct_within_18"] = (spec["within_18_weeks"] / spec["total_waiting"] * 100).round(1)
    return spec

@st.cache_data(show_spinner="Loading trust data...")
def build_trust_data(df):
    latest_period = df["period"].max()
    trust = (
        df[(df["period"]==latest_period) &
           (df["rtt_part_type"]=="Part_2") &
           (df["treatment_function_code"]=="C_999")]
        .groupby(["provider_org_code","provider_org_name","commissioner_org_name"])[
            ["total_waiting","within_18_weeks","over_52_weeks"]
        ].sum().reset_index()
    )
    trust["pct_within_18"]      = (trust["within_18_weeks"] / trust["total_waiting"] * 100).round(1)
    trust["median_wait_weeks"]  = ((1 - trust["pct_within_18"]/100) * 52 + 4).round(0).astype(int)
    return trust


# ── Load everything ───────────────────────────────────────────
try:
    df           = load_data()
    national     = build_national(df)
    nat_forecast = run_prophet(national[["period_dt","total_waiting"]])
    spec_data    = build_specialty_data(df)
    trust_data   = build_trust_data(df)

    latest       = national.iloc[-1]
    hist_end     = national["period_dt"].max()
    fc_future    = nat_forecast[nat_forecast["ds"] > hist_end]

    DATA_LOADED  = True
except FileNotFoundError:
    DATA_LOADED  = False


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HEADER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

st.markdown(f"""
<div style='background:{NHS_BLUE};padding:20px 24px;border-radius:16px;margin-bottom:20px'>
  <h2 style='color:white;margin:0'>🏥 NHS RTT Waiting List Predictor</h2>
  <p style='color:#93C5FD;margin:4px 0 0;font-size:13px'>
    AI-powered forecasting · England 2021–2025 · NHS England Open Data
  </p>
</div>
""", unsafe_allow_html=True)

if not DATA_LOADED:
    st.error("""
    ⚠️ **CSV file not found.**

    Make sure `nhs_rtt_waiting_times_2021_2025.csv` is in the **same folder** as `app.py`.

    Your folder should look like:
    ```
    📁 my_project/
       app.py
       nhs_rtt_waiting_times_2021_2025.csv
    ```
    Then run: `streamlit run app.py`
    """)
    st.stop()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# KPI CARDS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def kpi(label, value, sub, color):
    return f"""
    <div class='metric-card'>
        <div class='metric-label'>{label}</div>
        <div class='metric-value' style='color:{color}'>{value}</div>
        <div class='metric-sub'>{sub}</div>
    </div>"""

def fmt(n):
    if not n or np.isnan(n): return "—"
    return f"{n/1e6:.2f}M" if n>=1e6 else f"{n/1e3:.0f}K" if n>=1e3 else str(int(n))

pct_color = NHS_GREEN if latest.pct_within_18 >= 70 else NHS_YELLOW if latest.pct_within_18 >= 50 else NHS_RED
fc12 = fc_future["yhat"].iloc[-1] if len(fc_future) >= 12 else fc_future["yhat"].iloc[-1]

c1,c2,c3,c4 = st.columns(4)
with c1: st.markdown(kpi("Total Waiting (Latest)", fmt(latest.total_waiting), "Incomplete pathways", NHS_BLUE), unsafe_allow_html=True)
with c2: st.markdown(kpi("% Within 18 Weeks", f"{latest.pct_within_18}%", "Target: 92%", pct_color), unsafe_allow_html=True)
with c3: st.markdown(kpi("Over 52 Weeks", fmt(latest.over_52_weeks), "Long waits", NHS_RED), unsafe_allow_html=True)
with c4: st.markdown(kpi("12-Month Forecast", fmt(fc12), "Predicted total", NHS_YELLOW), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TABS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

tab1, tab2, tab3 = st.tabs(["📊 National Trend", "🔬 By Specialty", "👤 Patient Predictor"])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 1 — NATIONAL TREND
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

with tab1:
    metric = st.radio("View", ["Waiting List Size", "% Within 18 Weeks"],
                      horizontal=True, label_visibility="collapsed")

    fig = go.Figure()

    if metric == "Waiting List Size":
        fig.add_trace(go.Scatter(
            x=national["period_dt"], y=national["total_waiting"]/1e6,
            name="Historical", line=dict(color=NHS_BLUE, width=2.5),
            fill="tozeroy", fillcolor="rgba(0,48,135,0.08)"
        ))
        fig.add_trace(go.Scatter(
            x=fc_future["ds"], y=fc_future["yhat"]/1e6,
            name="Forecast", line=dict(color=NHS_GREEN, width=2.5, dash="dash")
        ))
        fig.add_traces([
            go.Scatter(x=fc_future["ds"], y=fc_future["yhat_upper"]/1e6,
                       fill=None, mode="lines", line=dict(width=0), showlegend=False),
            go.Scatter(x=fc_future["ds"], y=fc_future["yhat_lower"]/1e6,
                       fill="tonexty", mode="lines", line=dict(width=0),
                       fillcolor="rgba(0,178,148,0.15)", name="95% CI"),
        ])
        fig.update_yaxes(title="Patients (Millions)")
    else:
        fig.add_trace(go.Scatter(
            x=national["period_dt"], y=national["pct_within_18"],
            name="% Within 18 Wks", line=dict(color=NHS_BLUE, width=2.5)
        ))
        fig.add_hline(y=92, line_dash="dash", line_color=NHS_RED,
                      annotation_text="92% constitutional target")
        fig.add_hline(y=65, line_dash="dot", line_color=NHS_YELLOW,
                      annotation_text="65% March 2026 interim target")
        fig.update_yaxes(title="Percentage (%)", range=[40, 100])

    fig.update_layout(height=360, template="plotly_white",
                      title="National Waiting List — Historical & 12-Month Forecast",
                      legend=dict(orientation="h", y=-0.15))
    st.plotly_chart(fig, use_container_width=True)

    # Over 52 weeks bar
    fig2 = go.Figure(go.Bar(
        x=national["period_dt"], y=national["over_52_weeks"]/1e3,
        marker_color=NHS_RED, opacity=0.8
    ))
    fig2.update_layout(height=250, template="plotly_white",
                       title="Patients Waiting Over 52 Weeks (Thousands)",
                       yaxis_title="Patients (K)")
    st.plotly_chart(fig2, use_container_width=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 2 — BY SPECIALTY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

with tab2:
    all_specs = sorted(spec_data["treatment_function_name"].dropna().unique())
    sel_spec  = st.selectbox("Select specialty", all_specs)

    spec_hist = spec_data[spec_data["treatment_function_name"]==sel_spec].copy()

    if len(spec_hist) >= 12:
        spec_fc = run_prophet(spec_hist[["period_dt","total_waiting"]], periods=12)
        spec_future = spec_fc[spec_fc["ds"] > hist_end]

        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=spec_hist["period_dt"], y=spec_hist["total_waiting"],
            name="Historical", line=dict(color=NHS_BLUE, width=2.5),
            fill="tozeroy", fillcolor="rgba(0,48,135,0.08)"
        ))
        fig3.add_trace(go.Scatter(
            x=spec_future["ds"], y=spec_future["yhat"],
            name="Forecast", line=dict(color=NHS_GREEN, width=2.5, dash="dash")
        ))
        fig3.add_traces([
            go.Scatter(x=spec_future["ds"], y=spec_future["yhat_upper"],
                       fill=None, mode="lines", line=dict(width=0), showlegend=False),
            go.Scatter(x=spec_future["ds"], y=spec_future["yhat_lower"],
                       fill="tonexty", mode="lines", line=dict(width=0),
                       fillcolor="rgba(0,178,148,0.15)", name="95% CI"),
        ])
        fig3.update_layout(height=340, template="plotly_white",
                           title=f"{sel_spec} — Waiting List & 12-Month Forecast",
                           yaxis_title="Patients", legend=dict(orientation="h",y=-0.15))
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("Not enough data points for this specialty to generate a forecast.")

    # Trust table
    st.subheader("Trust Performance — Latest Month")
    display_trusts = trust_data.sort_values("median_wait_weeks")[
        ["provider_org_name","commissioner_org_name",
         "total_waiting","pct_within_18","median_wait_weeks"]
    ].rename(columns={
        "provider_org_name":    "Trust",
        "commissioner_org_name":"Region / ICB",
        "total_waiting":        "Total Waiting",
        "pct_within_18":        "% Within 18 Wks",
        "median_wait_weeks":    "Est. Wait (wks)",
    })

    def highlight(val):
        if isinstance(val, (int,float)):
            if val >= 70: return f"color:{NHS_GREEN};font-weight:bold"
            if val >= 50: return f"color:{NHS_YELLOW};font-weight:bold"
            return f"color:{NHS_RED};font-weight:bold"
        return ""

    st.dataframe(
        display_trusts.style.applymap(highlight, subset=["% Within 18 Wks"]),
        use_container_width=True, height=400
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 3 — PATIENT PREDICTOR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

with tab3:
    st.markdown("### 👤 Patient Wait Time Predictor")
    st.caption("Fill in your details below to get an AI-powered wait estimate and alternative Trust suggestions.")

    # ── Symptom keyword → specialty mapper ───────────────────────────────
    SYMPTOM_MAP = {
        "knee":       "Trauma and Orthopaedic Service",
        "hip":        "Trauma and Orthopaedic Service",
        "back":       "Trauma and Orthopaedic Service",
        "joint":      "Trauma and Orthopaedic Service",
        "fracture":   "Trauma and Orthopaedic Service",
        "bone":       "Trauma and Orthopaedic Service",
        "hernia":     "General Surgery Service",
        "gallstone":  "General Surgery Service",
        "appendix":   "General Surgery Service",
        "stomach":    "General Surgery Service",
        "bowel":      "General Surgery Service",
        "kidney":     "Urology Service",
        "bladder":    "Urology Service",
        "prostate":   "Urology Service",
        "urine":      "Urology Service",
        "ear":        "Ear Nose and Throat Service",
        "nose":       "Ear Nose and Throat Service",
        "throat":     "Ear Nose and Throat Service",
        "hearing":    "Ear Nose and Throat Service",
        "tonsil":     "Ear Nose and Throat Service",
        "sinus":      "Ear Nose and Throat Service",
        "eye":        "Ophthalmology Service",
        "vision":     "Ophthalmology Service",
        "cataract":   "Ophthalmology Service",
        "heart":      "Cardiology Service",
        "chest":      "Cardiology Service",
        "cardiac":    "Cardiology Service",
        "skin":       "Dermatology Service",
        "rash":       "Dermatology Service",
        "eczema":     "Dermatology Service",
        "psoriasis":  "Dermatology Service",
        "period":     "Gynaecology Service",
        "gynaecol":   "Gynaecology Service",
        "ovarian":    "Gynaecology Service",
        "endometrio": "Gynaecology Service",
        "neuro":      "Neurology Service",
        "headache":   "Neurology Service",
        "migraine":   "Neurology Service",
        "seizure":    "Neurology Service",
        "digestion":  "Gastroenterology Service",
        "crohn":      "Gastroenterology Service",
        "ibs":        "Gastroenterology Service",
        "colitis":    "Gastroenterology Service",
    }

    def suggest_specialty(symptoms_text):
        """Suggest specialty from symptom keywords."""
        if not symptoms_text:
            return None
        low = symptoms_text.lower()
        for kw, spec in SYMPTOM_MAP.items():
            if kw in low:
                return spec
        return None

    def build_advice(specialty, urgency, symptoms, trust,
                     region, curr_wait, curr_pct,
                     region_alts, spec_trend,
                     spec_hist, nat_hist):
        """
        Generate structured, data-driven advice entirely from
        NHS RTT data — no external API needed.
        """
        lines = []

        # ── 1. Wait estimate ─────────────────────────────────────
        urgency_factor = {
            "Two-Week Wait (Cancer Pathway)": 0.08,
            "Urgent":  0.45,
            "Routine": 1.0,
        }.get(urgency, 1.0)

        adj_wait = max(2, round(curr_wait * urgency_factor))

        lines.append("📅  ESTIMATED WAIT TIME")
        lines.append("─" * 42)

        if urgency == "Two-Week Wait (Cancer Pathway)":
            lines.append(
                f"⚠️  You are on the Two-Week Wait (cancer) pathway. "
                f"The NHS target is to see you within 14 days of referral. "
                f"If you have not received an appointment within 14 days, "
                f"contact your GP immediately."
            )
        elif urgency == "Urgent":
            lines.append(
                f"Your referral is marked URGENT. You are prioritised above "
                f"routine patients. Based on current NHS RTT data for "
                f"{trust}, your estimated wait is approximately "
                f"~{adj_wait} weeks — significantly shorter than the "
                f"routine wait of ~{curr_wait} weeks."
            )
        else:
            lines.append(
                f"Based on current NHS England RTT data, patients referred "
                f"to {trust} for {specialty} are waiting approximately "
                f"~{curr_wait} weeks. Only {curr_pct}% of patients at this "
                f"trust are being seen within the 18-week NHS target "
                f"(constitutional standard: 92%)."
            )

        # ── 2. Trend context ──────────────────────────────────────
        lines.append("")
        lines.append("📈  WAITING LIST TREND")
        lines.append("─" * 42)

        if spec_trend == "increasing":
            lines.append(
                f"⚠️  The waiting list for {specialty} has been "
                f"INCREASING over recent months. If you delay action, "
                f"your wait could be longer. It is advisable to confirm "
                f"your referral with your GP as soon as possible."
            )
        elif spec_trend == "decreasing":
            lines.append(
                f"✅  The waiting list for {specialty} has been "
                f"DECREASING recently — a positive sign. The NHS Elective "
                f"Reform Plan is having some effect in this specialty."
            )
        else:
            lines.append(
                f"The waiting list for {specialty} has been broadly "
                f"stable over recent months."
            )

        # ── 3. Alternatives ───────────────────────────────────────
        lines.append("")
        lines.append("🏥  ALTERNATIVE TRUSTS IN YOUR REGION")
        lines.append("─" * 42)

        better_alts = region_alts[
            region_alts["median_wait_weeks"] < curr_wait
        ].head(3)

        if better_alts.empty:
            lines.append(
                f"Your current trust ({trust}) already has one of the "
                f"shorter wait times in {region}. No significantly better "
                f"alternatives were found nearby."
            )
        else:
            lines.append(
                f"The following trusts in {region} have shorter wait times "
                f"and may be worth considering:"
            )
            lines.append("")
            for i, (_, alt) in enumerate(better_alts.iterrows(), 1):
                saving = max(0, curr_wait - int(alt.median_wait_weeks))
                status = "✅ On track" if alt.pct_within_18 >= 65 else "⚠️  Below target"
                lines.append(
                    f"  {i}. {alt.provider_org_name}\n"
                    f"     Wait: ~{int(alt.median_wait_weeks)} weeks  "
                    f"| Save: ~{saving} weeks  "
                    f"| 18-wk performance: {alt.pct_within_18}%  "
                    f"| {status}"
                )

        # ── 4. Right to Choose ────────────────────────────────────
        lines.append("")
        lines.append("💡  YOUR NHS RIGHT TO CHOOSE")
        lines.append("─" * 42)
        lines.append(
            "Under the NHS Constitution, you have a legal right to choose "
            "where you receive your NHS treatment. You can ask your GP to "
            "refer you to any of the trusts listed above via the NHS "
            "e-Referral Service (formerly Choose and Book). This is free "
            "and you do not need to give a reason for choosing a different "
            "trust."
        )

        # ── 5. GP follow-up ───────────────────────────────────────
        lines.append("")
        lines.append("🩺  SHOULD YOU CONTACT YOUR GP?")
        lines.append("─" * 42)

        if urgency == "Two-Week Wait (Cancer Pathway)":
            lines.append(
                "YES — contact your GP today if you have not received a "
                "hospital appointment within 14 days of referral."
            )
        elif curr_pct < 55:
            lines.append(
                "YES — your trust's 18-week performance is significantly "
                "below the NHS target. We recommend contacting your GP to "
                "discuss switching your referral to a better-performing trust."
            )
        elif spec_trend == "increasing":
            lines.append(
                "CONSIDER IT — given the increasing waiting list trend for "
                "this specialty, speak to your GP about whether your referral "
                "can be expedited or redirected."
            )
        else:
            lines.append(
                "Continue to monitor. If you have not received an appointment "
                "within 18 weeks of your referral date, contact your GP to "
                "chase the referral — this is your right under the NHS "
                "Constitution."
            )

        # ── 6. Data source note ───────────────────────────────────
        lines.append("")
        lines.append("─" * 42)
        lines.append(
            "📊 Analysis based on NHS England Consultant-Led RTT Waiting "
            "Times data (2021–2025). All figures are from the most recent "
            "available monthly release."
        )

        return "\n".join(lines)

    # ── Form ──────────────────────────────────────────────────────────────
    # ── All dropdowns OUTSIDE the form so region → trust updates live ────
    symptoms = st.text_area(
        "Describe your symptoms (optional — helps suggest specialty)",
        placeholder="e.g. knee pain when walking, referred by GP for orthopaedic assessment..."
    )

    col1, col2 = st.columns(2)
    with col1:
        suggested    = suggest_specialty(symptoms)
        spec_default = all_specs.index(suggested) + 1 if suggested and suggested in all_specs else 0
        specialty    = st.selectbox("Referred Specialty *", [""] + all_specs, index=spec_default)

        all_regions       = sorted(trust_data["commissioner_org_name"].dropna().unique().tolist())
        region            = st.selectbox("Your Region / ICB *", [""] + all_regions)

    with col2:
        urgency = st.selectbox(
            "Referral Urgency *",
            ["", "Routine", "Urgent", "Two-Week Wait (Cancer Pathway)"]
        )

        # Filter trusts live based on selected region
        if region:
            region_trust_list = sorted(
                trust_data[
                    trust_data["commissioner_org_name"] == region
                ]["provider_org_name"].dropna().unique().tolist()
            )
        else:
            region_trust_list = []

        trust = st.selectbox(
            "Your NHS Trust *",
            ["Select your region first..." if not region else ""] + region_trust_list,
            disabled=(not region),
        )

    submitted = st.button(
        "🔮 Predict My Wait & Find Alternatives",
        use_container_width=True, type="primary"
    )

    if submitted:
        if not specialty or not urgency or not trust:
            st.warning("Please fill in Specialty, Urgency and Trust before predicting.")
        else:
            trust_row   = trust_data[trust_data["provider_org_name"] == trust]
            region_alts = trust_data[
                (trust_data["commissioner_org_name"] == region) &
                (trust_data["provider_org_name"] != trust)
            ].sort_values("median_wait_weeks")

            curr_wait = int(trust_row["median_wait_weeks"].values[0]) if not trust_row.empty else 24
            curr_pct  = float(trust_row["pct_within_18"].values[0])   if not trust_row.empty else 60.0

            # Trend detection from real spec data
            spec_hist_filt = spec_data[spec_data["treatment_function_name"] == specialty]
            spec_trend = "stable"
            if len(spec_hist_filt) > 3:
                recent = spec_hist_filt.sort_values("period_dt").tail(4)["total_waiting"].values
                change = (recent[-1] - recent[0]) / max(recent[0], 1) * 100
                spec_trend = "increasing" if change > 3 else "decreasing" if change < -3 else "stable"

            # ── Wait time cards ───────────────────────────────────
            st.markdown("#### 📅 Estimated Wait Times")
            urgency_factor = {"Two-Week Wait (Cancer Pathway)":0.08,"Urgent":0.45,"Routine":1.0}.get(urgency,1.0)
            adj_wait = max(2, round(curr_wait * urgency_factor))

            cols = st.columns([1.2] + [1]*min(3, len(region_alts)))
            with cols[0]:
                st.markdown(f"""
                <div style='background:#EFF6FF;border-radius:14px;padding:16px;
                            border:1px solid #BFDBFE;text-align:center'>
                    <p style='font-size:11px;color:#6B7280;margin:0'>Your trust<br>
                    <b>{trust[:28]}</b></p>
                    <p style='font-size:32px;font-weight:700;color:{NHS_BLUE};margin:6px 0'>
                        ~{adj_wait} wks</p>
                    <p style='font-size:11px;color:#6B7280;margin:0'>
                        {curr_pct}% within 18 weeks</p>
                </div>""", unsafe_allow_html=True)

            better = region_alts[region_alts["median_wait_weeks"] < curr_wait].head(3)
            for i, (_, alt) in enumerate(better.iterrows()):
                saving = max(0, curr_wait - int(alt.median_wait_weeks))
                with cols[i+1]:
                    st.markdown(f"""
                    <div style='background:#F0FDF4;border-radius:14px;padding:16px;
                                border:1px solid #BBF7D0;text-align:center'>
                        <p style='font-size:11px;color:#6B7280;margin:0'>
                            {alt.provider_org_name[:28]}</p>
                        <p style='font-size:28px;font-weight:700;color:{NHS_GREEN};margin:6px 0'>
                            ~{int(alt.median_wait_weeks)} wks</p>
                        <p style='font-size:11px;color:{NHS_GREEN};margin:0'>
                            Save ~{saving} weeks ✓</p>
                    </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Data-driven advice ────────────────────────────────
            with st.spinner("Analysing NHS data..."):
                advice_text = build_advice(
                    specialty, urgency, symptoms, trust, region,
                    curr_wait, curr_pct, region_alts, spec_trend,
                    spec_hist_filt, national
                )

            st.markdown(f"""
            <div class='ai-box'>
                <div style='display:flex;align-items:center;gap:8px;margin-bottom:10px'>
                    <span style='font-size:20px'>📋</span>
                    <span style='font-weight:700;font-size:15px;color:#1E3A5F'>
                        NHS Data-Driven Recommendation
                    </span>
                </div>
                <pre style='font-size:13px;color:#374151;white-space:pre-wrap;
                            line-height:1.7;font-family:sans-serif;margin:0'>
{advice_text}</pre>
            </div>
            """, unsafe_allow_html=True)

            st.caption("⚠️ For informational purposes only. Not medical advice. Always consult your GP.")

    # How it works
    st.markdown("---")
    c1,c2,c3 = st.columns(3)
    with c1:
        st.info("**📊 Real NHS Data**\n\n5 years of NHS England RTT data (2021–2025), 60 monthly snapshots.")
    with c2:
        st.info("**🤖 Prophet + Claude AI**\n\nProphet forecasts trends. Claude generates personalised advice.")
    with c3:
        st.info("**🏥 Right to Choose**\n\nNHS Constitution gives you the right to choose where you're treated.")


# ── Footer ────────────────────────────────────────────────────
st.markdown("---")
st.caption("Data: NHS England RTT Open Data · Licence: OGL v3.0 · Not medical advice")