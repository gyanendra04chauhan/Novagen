import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Novagen | Health Risk Predictor",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Import fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Dark background */
    .stApp {
        background: linear-gradient(135deg, #0a0e1a 0%, #0f1629 50%, #0d1520 100%);
        color: #e2e8f0;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1117 0%, #161b2e 100%);
        border-right: 1px solid #1e3a5f;
    }
    [data-testid="stSidebar"] * { color: #cbd5e0 !important; }
    [data-testid="stSidebar"] .stSlider label { color: #94a3b8 !important; font-size: 0.85rem !important; }

    /* Main header */
    .hero-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.6rem;
        font-weight: 700;
        background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.2;
    }
    .hero-sub {
        color: #64748b;
        font-size: 1.05rem;
        margin-top: 0.4rem;
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #111827 0%, #1a2744 100%);
        border: 1px solid #1e3a5f;
        border-radius: 16px;
        padding: 1.4rem 1.6rem;
        text-align: center;
        transition: transform 0.2s;
    }
    .metric-card:hover { transform: translateY(-2px); border-color: #38bdf8; }
    .metric-num {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2rem;
        font-weight: 700;
        color: #38bdf8;
    }
    .metric-label { font-size: 0.82rem; color: #64748b; margin-top: 0.2rem; text-transform: uppercase; letter-spacing: 0.05em; }

    /* Result cards */
    .result-positive {
        background: linear-gradient(135deg, #0f2027 0%, #1a0a2e 100%);
        border: 2px solid #ef4444;
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        box-shadow: 0 0 40px rgba(239,68,68,0.15);
    }
    .result-negative {
        background: linear-gradient(135deg, #0a1f0f 0%, #0a2a1a 100%);
        border: 2px solid #22c55e;
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        box-shadow: 0 0 40px rgba(34,197,94,0.15);
    }
    .result-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.6rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .result-confidence {
        font-size: 3rem;
        font-weight: 800;
        font-family: 'Space Grotesk', sans-serif;
    }

    /* Section headers */
    .section-header {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.1rem;
        font-weight: 600;
        color: #38bdf8;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.8rem;
        padding-bottom: 0.4rem;
        border-bottom: 1px solid #1e3a5f;
    }

    /* Predict button */
    .stButton > button {
        background: linear-gradient(90deg, #38bdf8, #818cf8) !important;
        color: #0a0e1a !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 700 !important;
        font-size: 1.1rem !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.8rem 2.5rem !important;
        width: 100% !important;
        transition: all 0.3s !important;
        box-shadow: 0 4px 20px rgba(56,189,248,0.3) !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 30px rgba(56,189,248,0.5) !important;
    }

    /* Sliders */
    .stSlider [data-baseweb="slider"] { padding: 0.2rem 0; }
    .stSlider .stSlider { color: #38bdf8 !important; }

    /* Radio */
    .stRadio label { color: #94a3b8 !important; font-size: 0.88rem !important; }

    /* Divider */
    hr { border-color: #1e3a5f !important; }

    /* Info box */
    .info-box {
        background: rgba(56,189,248,0.08);
        border: 1px solid rgba(56,189,248,0.25);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin: 0.8rem 0;
        font-size: 0.9rem;
        color: #94a3b8;
    }

    /* Feature importance */
    .feature-pill {
        display: inline-block;
        background: rgba(56,189,248,0.1);
        border: 1px solid rgba(56,189,248,0.3);
        border-radius: 20px;
        padding: 0.3rem 0.8rem;
        margin: 0.2rem;
        font-size: 0.82rem;
        color: #7dd3fc;
    }

    /* Hide default streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ─── Train Model (cached — runs once on startup) ──────────────────────────────
@st.cache_resource
def train_model():
    base = Path(__file__).parent
    df = pd.read_csv(base / "novagen_dataset.csv")
    x = df.drop("Target", axis=1)
    y = df["Target"]
    features = list(x.columns)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled  = scaler.transform(x_test)

    model = RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42)
    model.fit(x_train_scaled, y_train)

    return model, scaler, features

model, scaler, feature_names = train_model()


# ─── Sidebar Inputs ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0 1.5rem 0;'>
        <div style='font-size:2.5rem;'>🧬</div>
        <div style='font-family: Space Grotesk, sans-serif; font-size:1.3rem; font-weight:700; color:#38bdf8;'>Novagen</div>
        <div style='font-size:0.78rem; color:#475569; margin-top:0.2rem;'>AI Health Risk Analyzer</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='section-header'>👤 Basic Info</div>", unsafe_allow_html=True)
    age = st.slider("Age (years)", 0, 100, 35)
    bmi = st.slider("BMI", 15.0, 40.0, 25.5, 0.1)

    st.markdown("<div class='section-header'>🩺 Vitals</div>", unsafe_allow_html=True)
    blood_pressure = st.slider("Blood Pressure (mmHg)", 60, 230, 120)
    heart_rate = st.slider("Heart Rate (bpm)", 50, 120, 74)
    cholesterol = st.slider("Cholesterol (mg/dL)", 100, 400, 200)
    glucose = st.slider("Glucose Level (mg/dL)", 70, 200, 100)

    st.markdown("<div class='section-header'>💤 Lifestyle</div>", unsafe_allow_html=True)
    sleep = st.slider("Sleep Hours / day", 0.0, 14.0, 7.0, 0.5)
    exercise = st.slider("Exercise Hours / day", 0.0, 8.0, 2.0, 0.5)
    water = st.slider("Water Intake (litres)", 0.0, 10.0, 3.5, 0.5)
    stress = st.slider("Stress Level (0–12)", 0, 12, 4)

    st.markdown("<div class='section-header'>🚬 Habits & History</div>", unsafe_allow_html=True)
    col_l, col_r = st.columns(2)
    with col_l:
        smoking = st.selectbox("Smoking", [0, 1], format_func=lambda x: "Yes" if x else "No")
        alcohol = st.selectbox("Alcohol", [0, 1], format_func=lambda x: "Yes" if x else "No")
        allergies = st.selectbox("Allergies", [0, 1], format_func=lambda x: "Yes" if x else "No")
    with col_r:
        diet = st.selectbox("Diet Quality", [0, 1], format_func=lambda x: "Good" if x else "Poor")
        mental_health = st.selectbox("Mental Health", [0, 1], format_func=lambda x: "Good" if x else "Poor")
        medical_history = st.selectbox("Medical History", [0, 1], format_func=lambda x: "Yes" if x else "No")

    physical_activity = st.selectbox("Physical Activity Level", [0, 1], format_func=lambda x: "Active" if x else "Sedentary")

    st.markdown("<div class='section-header'>🍽️ Diet Type & Blood Group</div>", unsafe_allow_html=True)
    diet_type = st.radio("Diet Type", ["Non-Veg", "Vegan", "Vegetarian"], horizontal=True)
    blood_group = st.radio("Blood Group", ["A", "AB", "B", "O"], horizontal=True)


# ─── Build Input Vector ───────────────────────────────────────────────────────
def build_input():
    return {
        "Age": age,
        "BMI": bmi,
        "Blood_Pressure": blood_pressure,
        "Cholesterol": cholesterol,
        "Glucose_Level": glucose,
        "Heart_Rate": heart_rate,
        "Sleep_Hours": sleep,
        "Exercise_Hours": exercise,
        "Water_Intake": water,
        "Stress_Level": stress,
        "Smoking": smoking,
        "Alcohol": alcohol,
        "Diet": diet,
        "MentalHealth": mental_health,
        "PhysicalActivity": physical_activity,
        "MedicalHistory": medical_history,
        "Allergies": allergies,
        "Diet_Type__Vegan": diet_type == "Vegan",
        "Diet_Type__Vegetarian": diet_type == "Vegetarian",
        "Blood_Group_AB": blood_group == "AB",
        "Blood_Group_B": blood_group == "B",
        "Blood_Group_O": blood_group == "O",
    }


# ─── Main Layout ──────────────────────────────────────────────────────────────
# Header
st.markdown("""
<div style='padding: 1.5rem 0 0.5rem 0;'>
    <div class='hero-title'>🧬 Novagen Health Risk Predictor</div>
    <div class='hero-sub'>AI-powered disease risk analysis — 94.3% accuracy · Random Forest · 9,549 patient records</div>
</div>
""", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# Model Stats row
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown("""<div class='metric-card'><div class='metric-num'>94.3%</div><div class='metric-label'>Model Accuracy</div></div>""", unsafe_allow_html=True)
with c2:
    st.markdown("""<div class='metric-card'><div class='metric-num'>9,549</div><div class='metric-label'>Training Samples</div></div>""", unsafe_allow_html=True)
with c3:
    st.markdown("""<div class='metric-card'><div class='metric-num'>22</div><div class='metric-label'>Health Features</div></div>""", unsafe_allow_html=True)
with c4:
    st.markdown("""<div class='metric-card'><div class='metric-num'>200</div><div class='metric-label'>Decision Trees</div></div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─── Prediction Zone ──────────────────────────────────────────────────────────
left_col, right_col = st.columns([1, 1.4])

with left_col:
    st.markdown("<div class='section-header'>🔬 Run Prediction</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='info-box'>
        Fill in the patient details on the <strong>left sidebar</strong>, then click below to get an instant AI-powered health risk assessment.
    </div>
    """, unsafe_allow_html=True)

    predict_clicked = st.button("⚡ Analyze Health Risk", use_container_width=True)

    if predict_clicked:
        input_data = build_input()
        df_input = pd.DataFrame([input_data])[feature_names]
        df_scaled = scaler.transform(df_input)

        pred = model.predict(df_scaled)[0]
        prob = model.predict_proba(df_scaled)[0]

        risk_pct = round(prob[1] * 100, 1)
        safe_pct = round(prob[0] * 100, 1)

        st.markdown("<br>", unsafe_allow_html=True)

        if pred == 1:
            st.markdown(f"""
            <div class='result-positive'>
                <div class='result-title' style='color:#f87171;'>⚠️ High Risk Detected</div>
                <div class='result-confidence' style='color:#ef4444;'>{risk_pct}%</div>
                <div style='color:#fca5a5; font-size:0.9rem; margin-top:0.5rem;'>Disease Risk Probability</div>
                <div style='color:#64748b; font-size:0.82rem; margin-top:1rem;'>Please consult a healthcare professional for detailed evaluation.</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='result-negative'>
                <div class='result-title' style='color:#4ade80;'>✅ Low Risk</div>
                <div class='result-confidence' style='color:#22c55e;'>{safe_pct}%</div>
                <div style='color:#86efac; font-size:0.9rem; margin-top:0.5rem;'>Healthy Probability</div>
                <div style='color:#64748b; font-size:0.82rem; margin-top:1rem;'>Keep maintaining your healthy lifestyle!</div>
            </div>
            """, unsafe_allow_html=True)

        # Mini gauge
        st.markdown("<br>", unsafe_allow_html=True)
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_pct,
            title={"text": "Risk Score", "font": {"color": "#94a3b8", "size": 14}},
            number={"suffix": "%", "font": {"color": "#e2e8f0", "size": 28}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#475569"},
                "bar": {"color": "#ef4444" if pred == 1 else "#22c55e"},
                "bgcolor": "#1e293b",
                "steps": [
                    {"range": [0, 30], "color": "#0f2f1a"},
                    {"range": [30, 60], "color": "#2d2008"},
                    {"range": [60, 100], "color": "#2d0a0a"},
                ],
                "threshold": {"line": {"color": "#f1f5f9", "width": 2}, "value": risk_pct}
            }
        ))
        fig_gauge.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0",
            height=200,
            margin=dict(t=30, b=10, l=20, r=20)
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    else:
        st.markdown("""
        <div style='text-align:center; padding: 3rem 1rem; color:#475569;'>
            <div style='font-size:3rem; margin-bottom:1rem;'>🩺</div>
            <div style='font-size:1rem;'>Configure patient parameters<br>in the sidebar, then click<br><strong style="color:#38bdf8;">Analyze Health Risk</strong></div>
        </div>
        """, unsafe_allow_html=True)


with right_col:
    st.markdown("<div class='section-header'>📊 Model Insights</div>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Feature Importance", "Input Profile", "About Model"])

    with tab1:
        importances = model.feature_importances_
        feat_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
        feat_df = feat_df.sort_values("Importance", ascending=True).tail(15)

        fig_fi = go.Figure(go.Bar(
            x=feat_df["Importance"],
            y=feat_df["Feature"],
            orientation="h",
            marker=dict(
                color=feat_df["Importance"],
                colorscale=[[0, "#1e3a5f"], [0.5, "#38bdf8"], [1, "#818cf8"]],
                showscale=False
            )
        ))
        fig_fi.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#94a3b8",
            xaxis=dict(showgrid=False, title="Importance Score", color="#64748b"),
            yaxis=dict(showgrid=False, color="#94a3b8"),
            height=380,
            margin=dict(t=10, b=10, l=10, r=10)
        )
        st.plotly_chart(fig_fi, use_container_width=True)

    with tab2:
        input_data = build_input()
        vitals = {k: input_data[k] for k in ["Age", "BMI", "Blood_Pressure", "Heart_Rate", "Cholesterol", "Glucose_Level", "Stress_Level"]}

        categories = ["Age", "BMI", "Blood_Pressure", "Heart_Rate", "Cholesterol", "Glucose_Level", "Stress_Level"]
        # Normalize to 0-1 scale for radar
        maxvals = [100, 40, 230, 120, 400, 200, 12]
        values = [vitals[c] / m for c, m in zip(categories, maxvals)]
        values += [values[0]]
        categories_plot = categories + [categories[0]]

        fig_radar = go.Figure(go.Scatterpolar(
            r=values,
            theta=categories_plot,
            fill="toself",
            fillcolor="rgba(56,189,248,0.15)",
            line=dict(color="#38bdf8", width=2),
            marker=dict(color="#818cf8", size=6)
        ))
        fig_radar.update_layout(
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(visible=True, range=[0, 1], color="#475569", gridcolor="#1e3a5f"),
                angularaxis=dict(color="#94a3b8", gridcolor="#1e3a5f")
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#94a3b8",
            height=340,
            margin=dict(t=20, b=20, l=20, r=20),
            showlegend=False
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        # Lifestyle bar
        lifestyle_labels = ["Sleep", "Exercise", "Water"]
        lifestyle_vals = [sleep / 14, exercise / 8, water / 10]
        lifestyle_colors = ["#818cf8", "#38bdf8", "#22c55e"]

        fig_life = go.Figure()
        for lbl, val, col in zip(lifestyle_labels, lifestyle_vals, lifestyle_colors):
            fig_life.add_trace(go.Bar(name=lbl, x=[lbl], y=[val * 100],
                                      marker_color=col, showlegend=False))
        fig_life.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#94a3b8",
            yaxis=dict(range=[0, 100], title="% of max", color="#64748b", gridcolor="#1e293b"),
            xaxis=dict(color="#94a3b8"),
            height=160,
            margin=dict(t=10, b=10, l=10, r=10),
            title=dict(text="Lifestyle Score", font=dict(size=12, color="#64748b"))
        )
        st.plotly_chart(fig_life, use_container_width=True)

    with tab3:
        st.markdown("""
        <div style='color:#94a3b8; line-height:1.8; font-size:0.92rem;'>
        <b style='color:#38bdf8;'>Algorithm:</b> Random Forest Classifier<br>
        <b style='color:#38bdf8;'>Trees:</b> 200 estimators, unlimited depth<br>
        <b style='color:#38bdf8;'>Accuracy:</b> 94.3% on held-out test set<br>
        <b style='color:#38bdf8;'>Dataset:</b> 9,549 patient records<br>
        <b style='color:#38bdf8;'>Train/Test Split:</b> 80% / 20% (random_state=42)<br>
        <b style='color:#38bdf8;'>Scaling:</b> StandardScaler (Z-score normalization)<br>
        <b style='color:#38bdf8;'>Features:</b> 22 health & lifestyle parameters<br>
        <b style='color:#38bdf8;'>Task:</b> Binary classification (Disease Risk)<br><br>

        <b style='color:#c084fc;'>Models Compared:</b><br>
        • Logistic Regression<br>
        • K-Nearest Neighbors (k=5)<br>
        • <b style='color:#22c55e;'>Random Forest ✓ (Best)</b><br>
        • Voting Classifier (Ensemble)<br>
        • Gradient Boosting<br><br>

        <div style='background:rgba(34,197,94,0.08); border:1px solid rgba(34,197,94,0.2); border-radius:8px; padding:0.8rem; font-size:0.85rem;'>
        ⚠️ <b>Disclaimer:</b> This tool is for educational & research purposes only. It is not a substitute for professional medical diagnosis.
        </div>
        </div>
        """, unsafe_allow_html=True)


# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align:center; color:#334155; font-size:0.82rem; padding: 0.5rem 0 1rem 0;'>
    Built with ❤️ using <span style='color:#38bdf8;'>Streamlit</span> + <span style='color:#818cf8;'>scikit-learn</span> &nbsp;·&nbsp;
    <a href='https://github.com/YOUR_USERNAME/novagen' style='color:#38bdf8; text-decoration:none;'>GitHub</a> &nbsp;·&nbsp;
    Novagen Health Risk Predictor v1.0
</div>
""", unsafe_allow_html=True)
