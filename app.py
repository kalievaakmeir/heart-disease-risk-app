import streamlit as st
import pandas as pd
import joblib

# =========================
# Load model bundle
# =========================
bundle = joblib.load("cvd_model.pkl")
model = bundle["model"]
FEATURES = bundle["features"]

st.set_page_config(page_title="CVD Risk Prediction", layout="centered")

st.title("CVD Risk Prediction")
st.write("Enter patient information to estimate cardiovascular disease risk.")

# =========================
# User input form
# =========================
values = {}

st.subheader("Demographics")
values["Age"] = st.slider("Age", 18, 100, 40)

values["hypertension"] = st.selectbox("Hypertension", [0, 1])
values["Diabetes"] = st.selectbox("Diabetes", [0, 1])
values["Obesity"] = st.selectbox("Obesity", [0, 1])
values["Smoking"] = st.selectbox("Smoking", [0, 1])
values["Alcohol_drinking"] = st.selectbox("Alcohol drinking", [0, 1])
values["Physical_inactivity"] = st.selectbox("Physical inactivity", [0, 1])
values["High_salt_intake"] = st.selectbox("High salt intake", [0, 1])

st.subheader("Clinical measurements")
values["Waist_circumference"] = st.slider("Waist circumference (cm)", 50, 150, 80)
values["CHOL"] = st.slider("Total cholesterol (mmol/L)", 2.0, 10.0, 4.5)

# =========================
# Prediction
# =========================
if st.button("Predict CVD risk"):
    try:
        input_df = pd.DataFrame([[values[f] for f in FEATURES]], columns=FEATURES)

        prob = model.predict_proba(input_df)[0][1]
        risk_percent = prob * 100

        st.subheader("Result")

        if risk_percent >= 50:
            st.error(f"High risk: {risk_percent:.1f}%")
        else:
            st.success(f"Low / moderate risk: {risk_percent:.1f}%")

        st.progress(int(risk_percent))

        st.caption("This tool is for research purposes only and does not replace clinical diagnosis.")

    except Exception as e:
        st.error("Model prediction failed")
        st.exception(e)

