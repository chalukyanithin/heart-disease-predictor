import streamlit as st
import pandas as pd
import numpy as np

# ---------------------------------------------------
# Page Configuration
# ---------------------------------------------------
st.set_page_config(
    page_title="Heart Disease Risk Predictor",
    page_icon="❤️",
    layout="wide"
)

# ---------------------------------------------------
# Custom CSS for Styling
# ---------------------------------------------------
st.markdown("""
<style>
.main-title {
    font-size: 46px;
    font-weight: 800;
    text-align: center;
    color: #111;
    letter-spacing: 1px;
}
.main-title span {
    background: linear-gradient(90deg, #e63946, #ff6b6b);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.subtitle {
    text-align: center;
    font-size: 18px;
    color: white;
    margin-top: -5px;
}
.title-container {
    background: Hunter green;
    padding: 30px;
    border-radius: 14px;
    box-shadow: 0px 8px 25px rgba(0,0,0,0.12);
    margin-bottom: 25px;
}
</style>

<div class="title-container">
    <div class="main-title">
        ❤️ <span>Heart Disease Risk Predictor</span>
    </div>
    <div class="subtitle">
        AI-assisted clinical decision support system
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# Header
# ---------------------------------------------------
st.markdown("""
<div class="card">
    <div class="title">❤️ Heart Disease Risk Prediction</div>
    <div class="subtitle">
        An interactive system to estimate heart disease risk using clinical parameters.
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# Sidebar - Input Mode
# ---------------------------------------------------
st.sidebar.header("📥 Input Method")

input_mode = st.sidebar.radio(
    "Choose input type:",
    ["Manual Input", "Upload CSV"]
)

# ---------------------------------------------------
# CSV Upload Mode
# ---------------------------------------------------
if input_mode == "Upload CSV":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📁 Upload Patient Data (CSV)")

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Preview of uploaded data:")
        st.dataframe(df.head())

        st.info("⚙️ Prediction logic will be applied here when model is integrated.")

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------
# Manual Input Mode
# ---------------------------------------------------
else:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🧾 Mandatory Patient Details")

    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age *", 20, 80, 45)
        sex = st.selectbox("Sex *", ["Male", "Female"])
        trestbps = st.slider("Resting Blood Pressure (mm Hg) *", 80, 200, 120)

    with col2:
        chol = st.slider("Serum Cholesterol (mg/dl) *", 100, 400, 200)
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl *", ["No", "Yes"])

    st.markdown("</div>", unsafe_allow_html=True)

    # ---------------------------------------------------
    # Optional Inputs
    # ---------------------------------------------------
    with st.expander("➕ Optional Clinical Parameters"):
        cp = st.selectbox("Chest Pain Type (0–3)", [0, 1, 2, 3])
        restecg = st.selectbox("Resting ECG Result (0,1,2)", [0, 1, 2])
        thalach = st.slider("Maximum Heart Rate Achieved", 70, 210, 150)
        exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
        oldpeak = st.slider("ST Depression (Oldpeak)", 0.0, 6.0, 1.0)
        slope = st.selectbox("Slope of Peak Exercise ST", [0, 1, 2])
        ca = st.selectbox("Number of Major Vessels (0–3)", [0, 1, 2, 3])
        thal = st.selectbox("Thalassemia", ["Normal", "Fixed Defect", "Reversible Defect"])

    # ---------------------------------------------------
    # Encode Inputs (Dummy Encoding)
    # ---------------------------------------------------
    sex = 1 if sex == "Male" else 0
    fbs = 1 if fbs == "Yes" else 0
    exang = 1 if exang == "Yes" else 0

    # ---------------------------------------------------
    # Predict Button
    # ---------------------------------------------------
    if st.button("🔍 Predict Risk"):
        # -----------------------------
        # DUMMY PREDICTION LOGIC
        # (Replace later with real ML)
        # -----------------------------
        risk_score = (
            (age > 55) +
            (chol > 240) +
            (trestbps > 140) +
            fbs +
            exang
        )

        probability = min(risk_score * 0.15, 0.95)

        if probability > 0.5:
            st.markdown(f"""
            <div class="result-high">
                <h3>⚠️ High Risk Detected</h3>
                <p>Estimated Risk Probability: <b>{probability:.2f}</b></p>
                <p>Please consult a healthcare professional.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-low">
                <h3>✅ Low Risk</h3>
                <p>Estimated Risk Probability: <b>{probability:.2f}</b></p>
                <p>Maintain a healthy lifestyle.</p>
            </div>
            """, unsafe_allow_html=True)

# ---------------------------------------------------
# Footer
# ---------------------------------------------------
st.markdown("""
<hr>
<p style="text-align:center; color:gray;">
Built for Kaggle Royale • Open-source • Educational Use Only
</p>
""", unsafe_allow_html=True)
