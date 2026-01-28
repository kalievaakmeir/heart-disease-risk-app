import streamlit as st
import pandas as pd
import joblib

# =========================
# Page config
# =========================
st.set_page_config(
    page_title="CVD Risk Prediction",
    page_icon="❤️",
    layout="centered"
)

# =========================
# Load model
# =========================
@st.cache_resource
def load_model():
    bundle = joblib.load("cvd_model.pkl")
    return bundle["model"], bundle["features"]

try:
    model, FEATURES = load_model()
except Exception as e:
    st.error("❌ Failed to load model file (cvd_model.pkl)")
    st.stop()

# =========================
# Helpers
# =========================
def yes_no(label):
    return 1 if st.radio(label, ["No", "Yes"], horizontal=True) == "Yes" else 0

# =========================
# UI
# =========================
st.title("❤️ Cardiovascular Disease Risk Prediction")
st.write("Enter patient information to estimate cardiovascular disease risk.")

st.divider()

values = {}

with st.container():
    st.subheader("👤 Demographics")
    values["Age"] = st.slider("Age (years)", 18, 100, 40)

with st.container():
    st.subheader("🩺 Medical history")
    values["hypertension"] = yes_no("Hypertension")
    values["Diabetes"] = yes_no("Diabetes")
    values["Obesity"] = yes_no("Obesity")
    values["Smoking"] = yes_no("Smoking")
    values["Alcohol_drinking"] = yes_no("Alcohol drinking")
    values["Physical_inactivity"] = yes_no("Physical inactivity")
    values["High_salt_intake"] = yes_no("High salt intake")

with st.container():
    st.subheader("🧪 Clinical measurements")
    values["Waist_circumference"] = st.slider("Waist circumference (cm)", 50, 150, 80)
    values["CHOL"] = st.slider("Total cholesterol (mmol/L)", 2.0, 10.0, 4.5)

st.divider()

# =========================
# Prediction
# =========================
if st.button("🔍 Predict CVD risk", use_container_width=True):
    try:
        input_df = pd.DataFrame([[values[f] for f in FEATURES]], columns=FEATURES)

        prob = model.predict_proba(input_df)[0][1]
        risk_percent = prob * 100

        st.subheader("📊 Result")

        st.progress(int(risk_percent))

        if risk_percent >= 50:
            st.error(f"⚠️ High risk: {risk_percent:.1f}%")
        elif risk_percent >= 20:
            st.warning(f"🟡 Moderate risk: {risk_percent:.1f}%")
        else:
            st.success(f"🟢 Low risk: {risk_percent:.1f}%")

        st.caption("For research purposes only. Not a substitute for medical diagnosis.")

    except Exception as e:
        st.error("❌ Prediction failed")
        st.exception(e)
