"""Central configuration for the NHS RTT dashboard."""

from __future__ import annotations


class Columns:
    """Common dataset column names."""

    PERIOD = "period"
    PERIOD_DT = "period_dt"
    YEAR = "year"
    MONTH = "month"
    PROVIDER_ORG_CODE = "provider_org_code"
    PROVIDER_ORG_NAME = "provider_org_name"
    COMMISSIONER_ORG_NAME = "commissioner_org_name"
    RTT_PART_TYPE = "rtt_part_type"
    TREATMENT_FUNCTION_CODE = "treatment_function_code"
    TREATMENT_FUNCTION_NAME = "treatment_function_name"
    TOTAL_WAITING = "total_waiting"
    WITHIN_18_WEEKS = "within_18_weeks"
    OVER_18_WEEKS = "over_18_weeks"
    OVER_52_WEEKS = "over_52_weeks"
    PCT_WITHIN_18_WEEKS = "pct_within_18_weeks"
    PCT_OVER_52_WEEKS = "pct_over_52_weeks"
    ESTIMATED_WAIT_WEEKS_PROXY = "estimated_wait_weeks_proxy"
    ESTIMATE_SOURCE = "estimate_source"
    ESTIMATE_SOURCE_LABEL = "estimate_source_label"
    ADJUSTED_WAIT_WEEKS_PROXY = "adjusted_wait_weeks_proxy"
    MATCHED_KEYWORDS = "matched_keywords"


APP_PAGE_CONFIG = {
    "page_title": "NHS RTT Waiting List Predictor",
    "page_icon": "🏥",
    "layout": "wide",
}

APP_TITLE = "NHS RTT Waiting List Predictor"
APP_SUBTITLE_TEMPLATE = (
    "NHS RTT analytics, forecasting, and patient wait guidance "
    "for {period_range}"
)

GLOBAL_STYLES = """
<style>
    .main { background-color: #F5F7FA; }
    .block-container { padding-top: 1.5rem; }
    .metric-card {
        background: white;
        border-radius: 14px;
        padding: 18px;
        text-align: center;
        border: 1px solid #E5E7EB;
        box-shadow: 0 1px 4px rgba(0, 0, 0, 0.05);
    }
    .metric-label { font-size: 12px; color: #6B7280; margin-bottom: 4px; }
    .metric-value { font-size: 26px; font-weight: 700; }
    .metric-sub { font-size: 11px; color: #9CA3AF; margin-top: 2px; }
    .hero-card {
        background: #003087;
        padding: 20px 24px;
        border-radius: 16px;
        margin-bottom: 20px;
    }
    .hero-title { color: white; margin: 0; }
    .hero-subtitle { color: #BFDBFE; margin: 4px 0 0; font-size: 13px; }
    .advice-box {
        background: #EFF6FF;
        border-left: 4px solid #003087;
        border-radius: 12px;
        padding: 16px;
        margin-top: 12px;
    }
    .mini-card {
        border-radius: 14px;
        padding: 16px;
        text-align: center;
        border: 1px solid #D1D5DB;
        background: white;
    }
    .mini-card.primary {
        background: #EFF6FF;
        border-color: #BFDBFE;
    }
    .mini-card.positive {
        background: #F0FDF4;
        border-color: #BBF7D0;
    }
</style>
"""

NHS_COLORS = {
    "blue": "#003087",
    "green": "#00B294",
    "red": "#DA291C",
    "yellow": "#FAB900",
    "ink": "#1E3A5F",
}

LOCAL_DATASET_FILENAME = "nhs_rtt_waiting_times_2021_2025.csv"
TMP_DATASET_PATH = "/tmp/nhs_rtt_waiting_times_2021_2025.csv"
DATASET_PATHS = (
    LOCAL_DATASET_FILENAME,
    TMP_DATASET_PATH,
)
KAGGLE_DATASET_SLUG = "hammad9191/nhs-consultant-led-rtt-waiting-times20212025"
KAGGLE_JSON_FILENAME = "kaggle.json"

REQUIRED_COLUMNS = [
    Columns.PERIOD,
    Columns.PROVIDER_ORG_CODE,
    Columns.PROVIDER_ORG_NAME,
    Columns.COMMISSIONER_ORG_NAME,
    Columns.RTT_PART_TYPE,
    Columns.TREATMENT_FUNCTION_CODE,
    Columns.TREATMENT_FUNCTION_NAME,
    Columns.TOTAL_WAITING,
    Columns.WITHIN_18_WEEKS,
    Columns.OVER_52_WEEKS,
]

NUMERIC_COLUMNS = [
    Columns.TOTAL_WAITING,
    Columns.WITHIN_18_WEEKS,
    Columns.OVER_18_WEEKS,
    Columns.OVER_52_WEEKS,
    Columns.PCT_WITHIN_18_WEEKS,
    Columns.PCT_OVER_52_WEEKS,
]

RTT_PART_INCOMPLETE = "Part_2"
ALL_SPECIALTIES_CODE = "C_999"

NATIONAL_TARGETS = {
    "constitutional_pct_within_18": 92.0,
    "interim_pct_within_18": 65.0,
}

URGENCY_MULTIPLIERS = {
    "Routine": 1.0,
    "Urgent": 0.45,
    "Two-Week Wait (Cancer Pathway)": 0.08,
}

MIN_PROXY_WAIT_WEEKS = 2
MAX_PROXY_WAIT_WEEKS = 56

FORECAST_PERIODS = 12
FORECAST_MIN_HISTORY_POINTS = 12
FORECAST_HOLDOUT_PERIODS = 3
FORECAST_BASELINE_WINDOW = 3
FULL_HISTORY_TARGET_MONTHS = 48

RECENT_TREND_WINDOW = 4
RECENT_TREND_THRESHOLD_PCT = 3.0

ESTIMATE_SOURCE_SPECIALTY = "trust_specialty_latest"
ESTIMATE_SOURCE_TRUST_WIDE = "trust_wide_latest"

ESTIMATE_SOURCE_LABELS = {
    ESTIMATE_SOURCE_SPECIALTY: "Trust + specialty latest-month estimate",
    ESTIMATE_SOURCE_TRUST_WIDE: "Trust-wide latest-month fallback estimate",
}

DISCLAIMER_TEXT = (
    "For informational purposes only. This dashboard does not provide medical advice "
    "and does not guarantee an individual waiting time."
)

PATIENT_PROXY_EXPLANATION = (
    "Estimated wait is derived from RTT performance data and should be treated as a planning guide, "
    "not a confirmed appointment time."
)

SPECIALTY_KEYWORDS = {
    "Trauma and Orthopaedic Service": (
        "knee",
        "hip",
        "back",
        "joint",
        "fracture",
        "bone",
        "orthopaedic",
        "arthritis",
        "shoulder",
        "ankle",
    ),
    "General Surgery Service": (
        "hernia",
        "gallstone",
        "appendix",
        "appendicitis",
        "stomach",
        "bowel",
        "abdominal",
        "surgery",
        "colon",
    ),
    "Urology Service": (
        "kidney",
        "bladder",
        "prostate",
        "urine",
        "urinary",
        "urology",
    ),
    "Ear Nose and Throat Service": (
        "ear",
        "nose",
        "throat",
        "hearing",
        "tonsil",
        "sinus",
        "ent",
        "swallowing",
    ),
    "Ophthalmology Service": (
        "eye",
        "vision",
        "cataract",
        "ophthalm",
        "retina",
        "glaucoma",
    ),
    "Cardiology Service": (
        "heart",
        "chest",
        "cardiac",
        "palpitation",
        "angina",
        "cardiology",
    ),
    "Dermatology Service": (
        "skin",
        "rash",
        "eczema",
        "psoriasis",
        "mole",
        "dermat",
        "itch",
    ),
    "Gynaecology Service": (
        "period",
        "gynaecol",
        "ovarian",
        "endometrio",
        "pelvic",
        "fibroid",
    ),
    "Neurology Service": (
        "neuro",
        "headache",
        "migraine",
        "seizure",
        "numbness",
        "neurology",
        "tingling",
    ),
    "Gastroenterology Service": (
        "digestion",
        "crohn",
        "ibs",
        "colitis",
        "reflux",
        "stomach",
        "gastro",
        "diarrhoea",
        "constipation",
    ),
    "Respiratory Medicine Service": (
        "breath",
        "breathing",
        "asthma",
        "copd",
        "lung",
        "respiratory",
        "cough",
    ),
}

PATIENT_GUIDANCE = {
    "right_to_choose": (
        "Under the NHS Constitution, patients usually have the right to choose where "
        "they receive their first outpatient appointment. A GP or referrer can discuss "
        "whether another trust available through the NHS e-Referral Service is suitable."
    ),
    "routine": (
        "If you have not heard about an appointment or your symptoms are changing, contact "
        "your GP practice or the booking team for an update."
    ),
    "urgent": (
        "Urgent referrals are prioritised ahead of routine pathways, but the adjustment shown "
        "here is still a rule-based estimate rather than pathway-level waiting-time data."
    ),
    "two_week_wait": (
        "If you are on a suspected cancer two-week-wait pathway and have not received an "
        "appointment within 14 days of referral, contact your GP practice promptly."
    ),
}
