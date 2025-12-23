import streamlit as st
import pandas as pd
import joblib

# --------------------------------------------------
# LOAD TRAINED MODEL
# --------------------------------------------------
MODEL_PATH = "heart_rf_model.pkl"
model = joblib.load(MODEL_PATH)

FEATURES = [
    "age", "sex", "bp", "chol", "sugar",
    "ecg", "heartrate", "exercise",
    "smoking", "alcohol"
]

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Heart Disease Risk Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# HARD OVERRIDE STREAMLIT DEFAULT HEADER
# --------------------------------------------------
st.markdown("""
<style>
header {visibility: hidden;}
footer {visibility: hidden;}
#MainMenu {visibility: hidden;}

:root {
    --bg-main: #020617;
    --bg-card: #0f172a;
    --text-main: #e5e7eb;
    --text-muted: #94a3b8;
}

.stApp { background-color: var(--bg-main); color: var(--text-main); }
label, span, p, div { color: var(--text-main) !important; }

.card {
    background: var(--bg-card);
    padding: 24px;
    border-radius: 16px;
    box-shadow: 0 24px 48px rgba(0,0,0,0.85);
    margin-bottom: 20px;
}

.result-high {
    background: rgba(255, 80, 80, 0.15);
    padding: 18px;
    border-radius: 12px;
    border-left: 6px solid #ff6b6b;
}
.result-low {
    background: rgba(34, 197, 94, 0.15);
    padding: 18px;
    border-radius: 12px;
    border-left: 6px solid #22c55e;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# TITLE
# --------------------------------------------------
st.markdown("""
<div class="card">
    <h1 style="text-align:center;">❤️ Heart Disease Risk Predictor</h1>
    <p style="text-align:center; color:#94a3b8;">
        AI-assisted clinical risk assessment system
    </p>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# MAIN LAYOUT
# --------------------------------------------------
left, right = st.columns([2, 1])

# --------------------------------------------------
# LEFT COLUMN — INPUTS
# --------------------------------------------------
with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🧾 Patient Details")

    c1, c2 = st.columns(2)

    with c1:
        age = st.slider("Age", 20, 80, 45)
        sex = st.selectbox("Sex", ["Male", "Female"])
        bp = st.slider("Resting Blood Pressure (mm Hg)", 80, 200, 120)

    with c2:
        chol = st.slider("Serum Cholesterol (mg/dl)", 100, 400, 200)
        sugar = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])

    with st.expander("➕ Optional Clinical Parameters"):
        ecg = st.selectbox("Resting ECG", ["Normal", "Abnormal"])
        heartrate = st.slider("Maximum Heart Rate", 70, 210, 150)
        exercise = st.selectbox("Exercise Induced Angina", ["No", "Yes"])

    # ----------------------------
    # ENCODING (MATCH BACKEND)
    # ----------------------------
    sex = 1 if sex == "Male" else 0
    sugar = 1 if sugar == "Yes" else 0
    exercise = 1 if exercise == "Yes" else 0
    ecg = 0 if ecg == "Normal" else 1

    smoking = 0
    alcohol = 0

    # ----------------------------
    # PREDICT
    # ----------------------------
    if st.button("🔍 Predict Risk"):
        input_df = pd.DataFrame([[
            age, sex, bp, chol, sugar,
            ecg, heartrate, exercise,
            smoking, alcohol
        ]], columns=FEATURES)

        probability = model.predict_proba(input_df)[0][1]
        percentage = round(probability * 100, 2)

        if probability > 0.5:
            st.markdown(
                f"<div class='result-high'><b>⚠️ High Risk</b><br>Estimated Risk: {percentage}%</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<div class='result-low'><b>✅ Low Risk</b><br>Estimated Risk: {percentage}%</div>",
                unsafe_allow_html=True
            )

    st.markdown('</div>', unsafe_allow_html=True)

# --------------------------------------------------
# RIGHT COLUMN — EXPLANATION
# --------------------------------------------------
with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("ℹ️ What does this mean?")
    st.write("""
• **Low Risk** → Lower likelihood of heart disease  
• **High Risk** → Higher likelihood, consult a doctor  
• **Risk %** → Model confidence, not diagnosis  
    """)
    st.write("⚠️ Educational use only.")
    st.markdown('</div>', unsafe_allow_html=True)

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown(
    "<p style='text-align:center;opacity:0.6;'>Built for Kaggle Royale</p>",
    unsafe_allow_html=True
)
