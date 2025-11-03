import streamlit as st
import numpy as np
import pandas as pd
import joblib
import shap
import plotly.graph_objects as go
import xgboost as xgb

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="GlycoTrack: Diabetes Risk Prediction", page_icon="🩺", layout="centered")

st.title("🩺 GlycoTrack: Diabetes Risk Prediction")
st.markdown("""
### Predict your likelihood of having diabetes based on key health indicators.
This model uses a Tuned **XGBoost (SMOTE)** algorithm optimized for balanced recall and AUC.
""")

# -------------------- LOAD MODEL --------------------
@st.cache_resource
def load_model():
    return joblib.load("final_glycotrack_model.pkl")

model = load_model()

# -------------------- USER INPUTS --------------------
st.sidebar.header("Enter Your Health Information")

age = st.sidebar.slider("Age", 18, 100, 35)
bmi = st.sidebar.slider("Body Mass Index (BMI)", 10.0, 60.0, 25.0)
high_bp = st.sidebar.selectbox("High Blood Pressure", ["No", "Yes"])
high_chol = st.sidebar.selectbox("High Cholesterol", ["No", "Yes"])
phys_activity = st.sidebar.selectbox("Physically Active", ["No", "Yes"])
general_health = st.sidebar.selectbox("General Health (1=Excellent, 5=Poor)", [1, 2, 3, 4, 5])

# Feature order (same as training)
feature_names = ["HighBP", "HighChol", "PhysActivity", "GenHlth", "Age", "BMI"]
input_data = pd.DataFrame([[
    1 if high_bp == "Yes" else 0,
    1 if high_chol == "Yes" else 0,
    1 if phys_activity == "Yes" else 0,
    general_health,
    age,
    bmi
]], columns=feature_names)

# -------------------- PREDICTION --------------------
if st.button("🔍 Predict Diabetes Risk"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("📊 Prediction Result")
    if prediction == 1:
        st.error(f"⚠️ High Diabetes Risk — ({probability*100:.1f}% probability)")
    else:
        st.success(f"✅ Low Diabetes Risk — ({(1 - probability)*100:.1f}% probability)")

    # -------------------- PROBABILITY GAUGE --------------------
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        title={'text': "Predicted Diabetes Risk (%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "crimson" if prediction == 1 else "green"},
            'steps': [
                {'range': [0, 50], 'color': "lightgreen"},
                {'range': [50, 80], 'color': "orange"},
                {'range': [80, 100], 'color': "red"}
            ]}
    ))
    st.plotly_chart(fig, use_container_width=True)

    # -------------------- SHAP EXPLANATION --------------------
    st.subheader("🔎 Why This Prediction?")
    st.write("This chart shows how each feature influenced your result:")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_data)
    shap.initjs()
    st.pyplot(shap.force_plot(explainer.expected_value, shap_values, input_data, matplotlib=True, show=False))

    # -------------------- SUMMARY METRICS --------------------
    st.markdown("---")
    st.subheader("📈 Model Performance Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", "0.84")
    col2.metric("Recall", "0.85")
    col3.metric("ROC-AUC", "0.91")

st.markdown("---")
st.caption("Final Model: Tuned XGBoost (SMOTE) | Developed by Kaushlendra Pratap Singh")
