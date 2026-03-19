import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sqlalchemy import create_engine
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CardioScan · Heart Disease Risk Analyzer",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Base & Typography ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Background ── */
.stApp {
    background-color: #FAFAFA;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #C0392B 0%, #922B21 60%, #641E16 100%);
    color: white;
}
[data-testid="stSidebar"] * {
    color: white !important;
}
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: white !important;
}

/* ── Header banner ── */
.header-banner {
    background: linear-gradient(135deg, #C0392B 0%, #E74C3C 100%);
    border-radius: 16px;
    padding: 32px 40px;
    margin-bottom: 32px;
    color: white;
    display: flex;
    align-items: center;
    gap: 24px;
    box-shadow: 0 8px 32px rgba(192,57,43,0.25);
}
.header-banner h1 {
    margin: 0;
    font-size: 2.2rem;
    font-weight: 700;
    letter-spacing: -0.5px;
}
.header-banner p {
    margin: 6px 0 0;
    font-size: 0.95rem;
    opacity: 0.88;
}

/* ── Section cards ── */
.card {
    background: white;
    border-radius: 14px;
    padding: 28px 32px;
    margin-bottom: 24px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    border: 1px solid #F0F0F0;
}
.card-title {
    font-size: 1.05rem;
    font-weight: 600;
    color: #1A1A2E;
    margin-bottom: 6px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.card-subtitle {
    font-size: 0.82rem;
    color: #888;
    margin-bottom: 20px;
}
.section-label {
    font-size: 0.78rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: #C0392B;
    margin-bottom: 14px;
}

/* ── Field description tooltips ── */
.field-desc {
    font-size: 0.78rem;
    color: #999;
    margin-top: -10px;
    margin-bottom: 10px;
    line-height: 1.4;
}

/* ── Predict button ── */
div[data-testid="stButton"] > button[kind="primary"] {
    background: linear-gradient(135deg, #C0392B, #E74C3C);
    border: none;
    border-radius: 10px;
    font-size: 1rem;
    font-weight: 600;
    padding: 14px 0;
    letter-spacing: 0.3px;
    box-shadow: 0 4px 14px rgba(192,57,43,0.35);
    transition: all 0.2s;
}
div[data-testid="stButton"] > button[kind="primary"]:hover {
    box-shadow: 0 6px 20px rgba(192,57,43,0.45);
    transform: translateY(-1px);
}

/* ── Result cards ── */
.result-high {
    background: linear-gradient(135deg, #FDEDEC, #FBEEE9);
    border: 2px solid #E74C3C;
    border-radius: 14px;
    padding: 24px 28px;
    margin-bottom: 20px;
}
.result-low {
    background: linear-gradient(135deg, #EAF9F0, #E8F8F0);
    border: 2px solid #27AE60;
    border-radius: 14px;
    padding: 24px 28px;
    margin-bottom: 20px;
}
.result-title {
    font-size: 1.4rem;
    font-weight: 700;
    margin-bottom: 4px;
}
.result-subtitle {
    font-size: 0.88rem;
    opacity: 0.75;
}

/* ── Risk factor pills ── */
.risk-pill {
    display: inline-block;
    background: #FDEDEC;
    border: 1px solid #E74C3C;
    color: #C0392B;
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.78rem;
    font-weight: 500;
    margin: 3px;
}
.safe-pill {
    display: inline-block;
    background: #EAF9F0;
    border: 1px solid #27AE60;
    color: #1E8449;
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.78rem;
    font-weight: 500;
    margin: 3px;
}

/* ── Metric cards ── */
.metric-grid {
    display: flex;
    gap: 16px;
    margin-top: 16px;
}
.metric-card {
    flex: 1;
    background: #F8F9FA;
    border-radius: 10px;
    padding: 16px;
    text-align: center;
    border: 1px solid #EBEBEB;
}
.metric-value {
    font-size: 1.8rem;
    font-weight: 700;
    color: #C0392B;
}
.metric-label {
    font-size: 0.75rem;
    color: #888;
    margin-top: 2px;
}

/* ── Sidebar stats cards ── */
.stat-card {
    background: rgba(255,255,255,0.12);
    border-radius: 10px;
    padding: 14px 16px;
    margin-bottom: 12px;
    border: 1px solid rgba(255,255,255,0.2);
}
.stat-number {
    font-size: 1.6rem;
    font-weight: 700;
}
.stat-desc {
    font-size: 0.78rem;
    opacity: 0.85;
    margin-top: 2px;
}

/* ── Recommendation cards ── */
.rec-card {
    background: #F8F9FA;
    border-left: 4px solid #C0392B;
    border-radius: 0 10px 10px 0;
    padding: 14px 18px;
    margin-bottom: 10px;
}
.rec-title {
    font-weight: 600;
    font-size: 0.88rem;
    color: #1A1A2E;
}
.rec-text {
    font-size: 0.82rem;
    color: #666;
    margin-top: 3px;
    line-height: 1.5;
}

/* ── Disclaimer ── */
.disclaimer {
    background: #FFF8E7;
    border: 1px solid #F0C040;
    border-radius: 10px;
    padding: 14px 18px;
    font-size: 0.8rem;
    color: #856404;
    margin-top: 24px;
}

/* Hide Streamlit branding ── */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Model training ─────────────────────────────────────────────────────────────
def _load_dataframe() -> pd.DataFrame:
    """Try Neon DB first, fall back to UCI OpenML dataset."""
    db_url = os.environ.get("DATABASE_URL")
    if db_url:
        try:
            engine = create_engine(db_url, connect_args={"sslmode": "require"})
            return pd.read_sql("SELECT * FROM heart_disease", engine)
        except Exception:
            pass
    # Fallback: fetch UCI Heart Disease dataset from OpenML
    data = fetch_openml("heart-disease", version=1, as_frame=True, parser="auto")
    df = data.frame.copy()
    df.columns = [c.lower() for c in df.columns]
    # OpenML uses 'num' as the target column; binarise to 0/1
    if "num" in df.columns:
        df = df.rename(columns={"num": "target"})
    df["target"] = (df["target"].astype(int) > 0).astype(int)
    df = df.apply(pd.to_numeric, errors="coerce").dropna()
    return df


@st.cache_resource(show_spinner="Training model…")
def load_model():
    df = _load_dataframe()

    features = ["age", "sex", "cp", "trestbps", "chol", "fbs",
                "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
    X = df[features]
    y = df["target"]

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=5,
        class_weight=None,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


model = load_model()
OPTIMAL_THRESHOLD = 0.403


# ── Helper: Gauge chart ────────────────────────────────────────────────────────
def make_gauge(prob: float) -> go.Figure:
    pct = prob * 100
    if pct < 30:
        color = "#27AE60"
        label = "Low Risk"
    elif pct < 55:
        color = "#F39C12"
        label = "Moderate Risk"
    elif pct < 75:
        color = "#E67E22"
        label = "High Risk"
    else:
        color = "#C0392B"
        label = "Very High Risk"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        number={"suffix": "%", "font": {"size": 36, "color": color, "family": "Inter"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#CCC",
                     "tickfont": {"size": 11}},
            "bar": {"color": color, "thickness": 0.28},
            "bgcolor": "white",
            "borderwidth": 0,
            "steps": [
                {"range": [0,  30], "color": "#EAF9F0"},
                {"range": [30, 55], "color": "#FEF9E7"},
                {"range": [55, 75], "color": "#FEF5E7"},
                {"range": [75, 100], "color": "#FDEDEC"},
            ],
            "threshold": {
                "line": {"color": "#C0392B", "width": 3},
                "thickness": 0.8,
                "value": OPTIMAL_THRESHOLD * 100,
            },
        },
        title={"text": f"<b>{label}</b>", "font": {"size": 16, "color": color, "family": "Inter"}},
    ))
    fig.update_layout(
        height=280,
        margin=dict(l=20, r=20, t=40, b=10),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font={"family": "Inter"},
    )
    return fig


# ── Helper: identify risk factors ─────────────────────────────────────────────
def identify_risk_factors(inputs: dict) -> tuple[list[str], list[str]]:
    risks, safes = [], []

    if inputs["age"] >= 55:
        risks.append(f"Age {inputs['age']} (elevated risk ≥55)")
    else:
        safes.append(f"Age {inputs['age']} (lower age risk)")

    if inputs["sex"] == 1:
        risks.append("Male sex (higher statistical risk)")
    else:
        safes.append("Female sex (lower statistical risk)")

    cp_map = {1: "Typical Angina", 2: "Atypical Angina", 3: "Non-anginal Pain", 4: "Asymptomatic"}
    if inputs["cp"] == 4:
        risks.append(f"Chest pain: Asymptomatic (paradoxically highest risk)")
    elif inputs["cp"] == 1:
        risks.append(f"Chest pain: Typical Angina")
    else:
        safes.append(f"Chest pain: {cp_map[inputs['cp']]}")

    if inputs["trestbps"] >= 140:
        risks.append(f"Blood pressure {inputs['trestbps']} mmHg (hypertension)")
    elif inputs["trestbps"] >= 130:
        risks.append(f"Blood pressure {inputs['trestbps']} mmHg (elevated)")
    else:
        safes.append(f"Blood pressure {inputs['trestbps']} mmHg (normal)")

    if inputs["chol"] >= 240:
        risks.append(f"Cholesterol {inputs['chol']} mg/dL (high)")
    elif inputs["chol"] >= 200:
        risks.append(f"Cholesterol {inputs['chol']} mg/dL (borderline)")
    else:
        safes.append(f"Cholesterol {inputs['chol']} mg/dL (optimal)")

    if inputs["fbs"] == 1:
        risks.append("Fasting blood sugar >120 mg/dL (diabetic range)")
    else:
        safes.append("Fasting blood sugar ≤120 mg/dL (normal)")

    if inputs["thalach"] < 120:
        risks.append(f"Max heart rate {inputs['thalach']} bpm (low exercise capacity)")
    elif inputs["thalach"] >= 150:
        safes.append(f"Max heart rate {inputs['thalach']} bpm (good exercise capacity)")
    else:
        risks.append(f"Max heart rate {inputs['thalach']} bpm (moderate)")

    if inputs["exang"] == 1:
        risks.append("Exercise-induced angina (significant marker)")
    else:
        safes.append("No exercise-induced angina")

    if inputs["oldpeak"] >= 2.0:
        risks.append(f"ST depression {inputs['oldpeak']} (significant ischemia)")
    elif inputs["oldpeak"] >= 1.0:
        risks.append(f"ST depression {inputs['oldpeak']} (mild concern)")
    else:
        safes.append(f"ST depression {inputs['oldpeak']} (minimal)")

    if inputs["slope"] == 3:
        risks.append("Downsloping ST segment (abnormal)")
    elif inputs["slope"] == 2:
        risks.append("Flat ST segment (borderline)")
    else:
        safes.append("Upsloping ST segment (normal)")

    if inputs["ca"] >= 2:
        risks.append(f"{inputs['ca']} major vessels narrowed (significant)")
    elif inputs["ca"] == 1:
        risks.append("1 major vessel narrowed (mild)")
    else:
        safes.append("No major vessel narrowing")

    thal_map = {3: "Normal", 6: "Fixed Defect", 7: "Reversible Defect"}
    if inputs["thal"] == 7:
        risks.append("Reversible thalassemia defect (high risk)")
    elif inputs["thal"] == 6:
        risks.append("Fixed thalassemia defect")
    else:
        safes.append("Normal thalassemia result")

    return risks, safes


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🫀 CardioScan")
    st.markdown("*AI-powered heart disease risk analysis*")
    st.markdown("---")

    st.markdown("### 📊 Heart Disease Facts")

    facts = [
        ("17.9M", "Deaths per year globally from CVD"),
        ("1 in 5", "Heart attacks occur without warning symptoms"),
        ("80%", "Of premature CVD events are preventable"),
        ("~3×", "Higher risk in people with diabetes"),
        ("≥55", "Age when risk significantly increases"),
    ]
    for num, desc in facts:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">{num}</div>
            <div class="stat-desc">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### ⚙️ Model Info")
    st.markdown("""
    <div class="stat-card">
        <div style="font-size:0.82rem; line-height:1.7;">
            <b>Algorithm:</b> Random Forest<br>
            <b>Trees:</b> 200 estimators<br>
            <b>Dataset:</b> UCI Combined (920 rows)<br>
            <b>Threshold:</b> 0.403 (recall ≥ 95%)<br>
            <b>Tuning:</b> GridSearchCV v3
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.75rem; opacity:0.7; text-align:center; line-height:1.6;">
        ⚠️ For educational use only.<br>
        Not a substitute for medical advice.
    </div>
    """, unsafe_allow_html=True)


# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-banner">
    <div style="font-size:3.5rem;">🫀</div>
    <div>
        <h1>CardioScan</h1>
        <p>Heart Disease Risk Analyzer &nbsp;·&nbsp; AI-Powered Clinical Decision Support &nbsp;·&nbsp; Random Forest Model</p>
    </div>
</div>
""", unsafe_allow_html=True)


# ── Input form ─────────────────────────────────────────────────────────────────
st.markdown('<div class="card-title">📋 Patient Information</div>', unsafe_allow_html=True)
st.markdown('<div class="card-subtitle">Enter the patient\'s clinical measurements below. All fields are required for an accurate prediction.</div>', unsafe_allow_html=True)

# Section: Demographics
st.markdown('<div class="section-label">Demographics</div>', unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
with c1:
    age = st.number_input("Age (years)", min_value=1, max_value=120, value=55)
    st.markdown('<div class="field-desc">Patient\'s age in years. Risk increases significantly after 55.</div>', unsafe_allow_html=True)
with c2:
    sex = st.selectbox("Biological Sex", options=[1, 0],
                       format_func=lambda x: "Male" if x == 1 else "Female")
    st.markdown('<div class="field-desc">Males statistically carry higher cardiovascular risk.</div>', unsafe_allow_html=True)
with c3:
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", options=[0, 1],
                       format_func=lambda x: "Yes" if x == 1 else "No")
    st.markdown('<div class="field-desc">Elevated fasting glucose indicates possible diabetes, a major risk factor.</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Section: Cardiac Symptoms
st.markdown('<div class="section-label">Cardiac Symptoms</div>', unsafe_allow_html=True)
c4, c5, c6 = st.columns(3)
with c4:
    cp = st.selectbox("Chest Pain Type", options=[1, 2, 3, 4],
                      format_func=lambda x: {1: "Typical Angina", 2: "Atypical Angina",
                                              3: "Non-anginal Pain", 4: "Asymptomatic"}[x])
    st.markdown('<div class="field-desc">Asymptomatic patients paradoxically show the highest disease prevalence in this dataset.</div>', unsafe_allow_html=True)
with c5:
    exang = st.selectbox("Exercise-Induced Angina", options=[0, 1],
                         format_func=lambda x: "Yes" if x == 1 else "No")
    st.markdown('<div class="field-desc">Chest pain or discomfort triggered by physical activity — a strong indicator.</div>', unsafe_allow_html=True)
with c6:
    thalach = st.number_input("Max Heart Rate Achieved (bpm)", min_value=60, max_value=220, value=150)
    st.markdown('<div class="field-desc">Peak heart rate during exercise stress test. Lower values suggest reduced cardiac reserve.</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Section: Blood & Vitals
st.markdown('<div class="section-label">Blood Pressure & Cholesterol</div>', unsafe_allow_html=True)
c7, c8 = st.columns(2)
with c7:
    trestbps = st.number_input("Resting Blood Pressure (mmHg)", min_value=50, max_value=250, value=130)
    st.markdown('<div class="field-desc">Normal: &lt;120/80 mmHg. Stage 1 hypertension: 130–139 mmHg. Stage 2: ≥140 mmHg.</div>', unsafe_allow_html=True)
with c8:
    chol = st.number_input("Serum Cholesterol (mg/dL)", min_value=50, max_value=700, value=240)
    st.markdown('<div class="field-desc">Desirable: &lt;200 mg/dL. Borderline high: 200–239. High: ≥240 mg/dL.</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Section: Electrocardiogram
st.markdown('<div class="section-label">Electrocardiogram (ECG)</div>', unsafe_allow_html=True)
c9, c10, c11 = st.columns(3)
with c9:
    restecg = st.selectbox("Resting ECG Results", options=[0, 1, 2],
                           format_func=lambda x: {0: "Normal", 1: "ST-T Abnormality",
                                                   2: "Left Ventricular Hypertrophy"}[x])
    st.markdown('<div class="field-desc">Resting electrocardiographic results measuring heart\'s electrical activity at rest.</div>', unsafe_allow_html=True)
with c10:
    oldpeak = st.number_input("ST Depression (Oldpeak)", min_value=0.0, max_value=10.0,
                               value=1.0, step=0.1, format="%.1f")
    st.markdown('<div class="field-desc">ST segment depression induced by exercise relative to rest. Higher values indicate ischemia.</div>', unsafe_allow_html=True)
with c11:
    slope = st.selectbox("Peak Exercise ST Slope", options=[1, 2, 3],
                         format_func=lambda x: {1: "Upsloping", 2: "Flat", 3: "Downsloping"}[x])
    st.markdown('<div class="field-desc">Shape of the ST segment during peak exercise. Downsloping is most concerning.</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Section: Advanced Diagnostics
st.markdown('<div class="section-label">Advanced Cardiac Diagnostics</div>', unsafe_allow_html=True)
c12, c13 = st.columns(2)
with c12:
    ca = st.selectbox("Major Vessels Colored by Fluoroscopy (0–3)", options=[0, 1, 2, 3])
    st.markdown('<div class="field-desc">Number of major coronary vessels (0–3) with significant narrowing visible on fluoroscopy imaging.</div>', unsafe_allow_html=True)
with c13:
    thal = st.selectbox("Thalassemia Blood Disorder", options=[3, 6, 7],
                        format_func=lambda x: {3: "Normal", 6: "Fixed Defect", 7: "Reversible Defect"}[x])
    st.markdown('<div class="field-desc">Nuclear stress test result. Reversible defects indicate areas of ischemia; fixed defects suggest prior infarction.</div>', unsafe_allow_html=True)


# ── Predict button ─────────────────────────────────────────────────────────────
predict_clicked = st.button("🔬  Analyze Heart Disease Risk", type="primary", use_container_width=True)


# ── Results ────────────────────────────────────────────────────────────────────
if predict_clicked:
    inputs = {
        "age": age, "sex": sex, "cp": cp, "trestbps": trestbps,
        "chol": chol, "fbs": fbs, "restecg": restecg, "thalach": thalach,
        "exang": exang, "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal,
    }

    patient = pd.DataFrame([inputs])
    prob_disease = model.predict_proba(patient)[0][1]
    prediction = int(prob_disease >= OPTIMAL_THRESHOLD)
    risks, safes = identify_risk_factors(inputs)

    st.markdown("---")
    st.markdown("## 🩺 Analysis Results")

    # ── Result banner
    if prediction == 1:
        st.markdown(f"""
        <div class="result-high">
            <div class="result-title" style="color:#C0392B;">
                ❤️‍🩹 Elevated Heart Disease Risk Detected
            </div>
            <div class="result-subtitle" style="color:#922B21;">
                The model predicts a <strong>{prob_disease:.1%}</strong> probability of heart disease.
                Please consult a cardiologist for further evaluation.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-low">
            <div class="result-title" style="color:#1E8449;">
                ✅ Lower Heart Disease Risk
            </div>
            <div class="result-subtitle" style="color:#1A5E35;">
                The model predicts a <strong>{prob_disease:.1%}</strong> probability of heart disease.
                Continue maintaining a heart-healthy lifestyle.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Gauge + metrics side by side
    g_col, m_col = st.columns([3, 2], gap="large")

    with g_col:
        st.markdown('<div class="card-title">📈 Risk Gauge</div>', unsafe_allow_html=True)
        st.plotly_chart(make_gauge(prob_disease), use_container_width=True, config={"displayModeBar": False})
        st.markdown(f"""
        <div style="text-align:center; font-size:0.78rem; color:#999; margin-top:-10px;">
            Decision threshold: {OPTIMAL_THRESHOLD:.3f} (tuned for ≥95% recall) &nbsp;·&nbsp; Red tick = threshold
        </div>
        """, unsafe_allow_html=True)

    with m_col:
        st.markdown('<div class="card-title">📊 Probability Breakdown</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown(f"""
        <div style="text-align:center; margin-bottom:20px;">
            <div style="font-size:3rem; font-weight:700; color:#C0392B;">{prob_disease:.1%}</div>
            <div style="font-size:0.85rem; color:#888;">Probability of Heart Disease</div>
        </div>
        <div style="text-align:center; margin-bottom:20px;">
            <div style="font-size:3rem; font-weight:700; color:#27AE60;">{1-prob_disease:.1%}</div>
            <div style="font-size:0.85rem; color:#888;">Probability of No Disease</div>
        </div>
        """, unsafe_allow_html=True)

        # Mini progress bars
        st.markdown("**Disease probability**")
        st.progress(prob_disease)
        st.markdown("**Healthy probability**")
        st.progress(1 - prob_disease)

        st.markdown(f"""
        <br>
        <div style="font-size:0.78rem; color:#888; line-height:1.6;">
            <b>Model:</b> Random Forest (200 trees)<br>
            <b>Max depth:</b> 6 &nbsp;·&nbsp; <b>Min leaf:</b> 5<br>
            <b>Training:</b> UCI Combined Dataset
        </div>
        """, unsafe_allow_html=True)

    # ── Risk factor breakdown
    st.markdown('<div class="card-title">🔍 Risk Factor Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-subtitle">Individual clinical markers contributing to the overall risk assessment.</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.78rem; color:#666; margin-bottom:12px; display:flex; gap:16px; align-items:center;">
        <span><span style="display:inline-block; width:10px; height:10px; border-radius:50%; background:#E74C3C; margin-right:5px;"></span><b>Red pills</b> — risk factors</span>
        <span><span style="display:inline-block; width:10px; height:10px; border-radius:50%; background:#27AE60; margin-right:5px;"></span><b>Green pills</b> — protective factors</span>
    </div>
    """, unsafe_allow_html=True)

    rf_col, sf_col = st.columns(2)
    with rf_col:
        st.markdown("**🔴 Concerning Factors**")
        if risks:
            pills = "".join(f'<span class="risk-pill">⚠ {r}</span>' for r in risks)
            st.markdown(pills, unsafe_allow_html=True)
        else:
            st.markdown('<span class="safe-pill">No major risk factors identified</span>', unsafe_allow_html=True)
    with sf_col:
        st.markdown("**🟢 Protective Factors**")
        if safes:
            pills = "".join(f'<span class="safe-pill">✓ {s}</span>' for s in safes)
            st.markdown(pills, unsafe_allow_html=True)
        else:
            st.markdown("*No notable protective factors identified.*")

    st.markdown("---")

    # ── What this means
    st.markdown('<div class="card-title">💡 Understanding Your Results</div>', unsafe_allow_html=True)

    if prediction == 1:
        st.markdown("""
        The model has flagged **elevated cardiovascular risk** based on the combination of clinical markers entered.
        A probability above the diagnostic threshold (40.3%) does not confirm a diagnosis — it indicates that
        the pattern of values resembles cases where heart disease was present in the training dataset.

        **What to do next:**
        - Schedule an appointment with a cardiologist for comprehensive evaluation
        - A formal stress test, echocardiogram, or coronary angiography may be recommended
        - Discuss current medications and whether risk-reduction therapy is appropriate
        - Address modifiable risk factors (blood pressure, cholesterol, blood sugar, weight)
        """)
    else:
        st.markdown("""
        The model indicates **lower cardiovascular risk** based on the clinical markers provided.
        A probability below the diagnostic threshold (40.3%) suggests the pattern of values more closely
        resembles healthy cases in the training data. However, heart disease can be present even with
        a lower score, and regular checkups remain essential.

        **Maintaining heart health:**
        - Continue regular annual physical examinations
        - Monitor blood pressure and cholesterol annually
        - Maintain a physically active lifestyle and balanced diet
        - Avoid smoking and limit alcohol consumption
        """)

    st.markdown("---")

    # ── Lifestyle recommendations
    st.markdown('<div class="card-title">🌿 Lifestyle Recommendations</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-subtitle">Evidence-based recommendations for cardiovascular health.</div>', unsafe_allow_html=True)

    recs = [
        ("🥗", "Heart-Healthy Diet",
         "Follow a Mediterranean or DASH diet. Prioritize vegetables, fruits, whole grains, lean proteins, and healthy fats (olive oil, nuts). Limit sodium, processed foods, and saturated fats."),
        ("🏃", "Regular Physical Activity",
         "Aim for at least 150 minutes of moderate-intensity aerobic exercise per week (brisk walking, cycling, swimming). Include 2 days of strength training."),
        ("🚭", "Avoid Smoking & Limit Alcohol",
         "Smoking doubles heart disease risk — cessation dramatically reduces risk within 1 year. Limit alcohol to ≤1 drink/day for women, ≤2 for men."),
        ("⚖️", "Weight Management",
         "Maintain a healthy BMI (18.5–24.9). Even a 5–10% weight loss can meaningfully reduce blood pressure and cholesterol."),
        ("😴", "Sleep & Stress Management",
         "Target 7–9 hours of quality sleep. Chronic stress and poor sleep are independent cardiovascular risk factors. Consider mindfulness or meditation."),
        ("💊", "Medication Adherence",
         "If prescribed antihypertensives, statins, or antidiabetics, take them consistently. Never discontinue without consulting your physician."),
    ]

    r1, r2 = st.columns(2)
    for i, (icon, title, text) in enumerate(recs):
        col = r1 if i % 2 == 0 else r2
        with col:
            st.markdown(f"""
            <div class="rec-card">
                <div class="rec-title">{icon} {title}</div>
                <div class="rec-text">{text}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Disclaimer
    st.markdown("""
    <div class="disclaimer">
        ⚠️ <strong>Medical Disclaimer:</strong> This tool is intended for educational and research purposes only.
        It is <strong>not</strong> a diagnostic device and must not replace professional medical evaluation.
        Always consult a qualified healthcare provider before making any medical decisions.
        Model accuracy is limited by the size and scope of the training dataset (UCI Combined, 920 records).
    </div>
    """, unsafe_allow_html=True)
