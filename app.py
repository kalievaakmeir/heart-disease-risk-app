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

st.title("❤️ CVD Risk Prediction")
st.write("Enter patient information to estimate cardiovascular disease risk.")

# =========================
# Helper function
# =========================
def yes_no(label):
    choice = st.radio(label, ["No", "Yes"], horizontal=True)
    return 1 if choice == "Yes" else 0

# =========================
# User input form
# =========================
values = {}

st.subheader("🧍 Demographics")
values["Age"] = st.slider("Age (years)", 18, 100, 40)

st.subheader("🩺 Medical history")
values["hypertension"] = yes_no("Hypertension")
values["Diabetes"] = yes_no("Diabetes")
values["Obesity"] = yes_no("Obesity")
values["Smoking"] = yes_no("Smoking")
values["Alcohol_drinking"] = yes_no("Alcohol drinking")
values["Physical_inactivity"] = yes_no("Physical inactivity")
values["High_salt_intake"] = yes_no("High salt intake")

st.subheader("🧪 Clinical measurements")
values["Waist_circumference"] = st.slider("Waist circumference (cm)", 50, 150, 80)
values["CHOL"] = st.slider("Total cholesterol (mmol/L)", 2.0, 10.0, 4.5)

# =========================
# Prediction
# =========================
st.markdown("---")

if st.button("🔍 Predict CVD risk"):
    try:
        input_df = pd.DataFrame([[values[f] for f in FEATURES]], columns=FEATURES)

        # ===== FIXED PROBABILITY LOGIC =====
        probs = model.predict_proba(input_df)[0]
        classes = model.classes_
        disease_index = list(classes).index(0)  # class 0 = disease
        risk_percent = probs[disease_index] * 100
        # ==================================

        st.subheader("📊 Result")

        st.progress(int(risk_percent))

        if risk_percent >= 50:
            st.error(f"⚠️ High risk: {risk_percent:.1f}%")
        elif risk_percent >= 20:
            st.warning(f"🟠 Moderate risk: {risk_percent:.1f}%")
        else:
            st.success(f"🟢 Low risk: {risk_percent:.1f}%")

        st.caption("For research purposes only. Not a substitute for medical diagnosis.")

    except Exception as e:
        st.error("Model prediction failed")
        st.exception(e)
