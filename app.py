import streamlit as st
import numpy as np
import pandas as pd
import joblib
import shap
import plotly.graph_objects as go
import xgboost as xgb
import matplotlib.pyplot as plt

# -------------------- 🎯 PAGE CONFIG --------------------
st.set_page_config(page_title="GlycoTrack: Diabetes Risk Prediction", page_icon="🩺", layout="centered")

st.title("🩺 GlycoTrack: Diabetes Risk Prediction")
st.markdown("""
### Predict your likelihood of having diabetes based on key health indicators.
This model uses a Tuned **XGBoost (SMOTE)** algorithm optimized for balanced recall and AUC.
""")

# -------------------- 📂 LOAD MODEL --------------------
@st.cache_resource
def load_model():
    return joblib.load("final_glycotrack_model.pkl")

model = load_model()

# -------------------- 📥 USER INPUTS --------------------
st.sidebar.header("Enter Your Health Information")

HighBP = st.sidebar.selectbox("High Blood Pressure", ["No", "Yes"])
HighChol = st.sidebar.selectbox("High Cholesterol", ["No", "Yes"])
CholCheck = st.sidebar.selectbox("Cholesterol Check (Past 5 Years)", ["No", "Yes"])
BMI = st.sidebar.slider("Body Mass Index (BMI)", 10.0, 60.0, 25.0)
Smoker = st.sidebar.selectbox("Smoked 100+ Cigarettes in Lifetime", ["No", "Yes"])
Stroke = st.sidebar.selectbox("Ever Had a Stroke?", ["No", "Yes"])
HeartDiseaseorAttack = st.sidebar.selectbox("Heart Disease or Attack History", ["No", "Yes"])
PhysActivity = st.sidebar.selectbox("Physically Active in Last 30 Days", ["No", "Yes"])
Fruits = st.sidebar.selectbox("Consume Fruits Daily?", ["No", "Yes"])
Veggies = st.sidebar.selectbox("Consume Vegetables Daily?", ["No", "Yes"])
HvyAlcoholConsump = st.sidebar.selectbox("Heavy Alcohol Consumption?", ["No", "Yes"])
AnyHealthcare = st.sidebar.selectbox("Have Any Health Coverage?", ["No", "Yes"])
NoDocbcCost = st.sidebar.selectbox("Couldn’t See Doctor Due to Cost?", ["No", "Yes"])
GenHlth = st.sidebar.selectbox("General Health (1=Excellent, 5=Poor)", [1, 2, 3, 4, 5])
MentHlth = st.sidebar.slider("Days of Poor Mental Health (Last 30 Days)", 0, 30, 5)
PhysHlth = st.sidebar.slider("Days of Poor Physical Health (Last 30 Days)", 0, 30, 5)
DiffWalk = st.sidebar.selectbox("Difficulty Walking or Climbing Stairs?", ["No", "Yes"])
Sex = st.sidebar.selectbox("Sex", ["Female", "Male"])
Age = st.sidebar.slider("Age", 18, 100, 35)
Education = st.sidebar.selectbox("Education Level (1=No school ... 6=College 4+ years)", [1, 2, 3, 4, 5, 6])
Income = st.sidebar.selectbox("Income Level (1=10000 or less ... 8=75000 or more)", [1, 2, 3, 4, 5, 6, 7, 8])

# Additional derived features
BMI_Category = st.sidebar.selectbox("BMI Category (1=Underweight, 2=Normal, 3=Overweight, 4=Obese)", [1, 2, 3, 4])
Age_Category = st.sidebar.selectbox("Age Category (1=Youth, 2=Adult, 3=Middle-aged, 4=Senior)", [1, 2, 3, 4])
Smoke_Alcohol = st.sidebar.slider("Combined Smoke-Alcohol Score (0–5)", 0.0, 5.0, 1.0)
BMIxAge = BMI * Age
Lifestyle_Score = st.sidebar.slider("Lifestyle Score (0–10)", 0.0, 10.0, 5.0)

# -------------------- 📂 FEATURE ORDER --------------------
feature_names = [
    "HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke",
    "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
    "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "GenHlth",
    "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age", "Education",
    "Income", "BMI_Category", "Age_Category", "Smoke_Alcohol",
    "BMIxAge", "Lifestyle_Score"
]

# -------------------- 📥 INPUT DATAFRAME --------------------
input_data = pd.DataFrame([[
    1 if HighBP == "Yes" else 0,
    1 if HighChol == "Yes" else 0,
    1 if CholCheck == "Yes" else 0,
    BMI,
    1 if Smoker == "Yes" else 0,
    1 if Stroke == "Yes" else 0,
    1 if HeartDiseaseorAttack == "Yes" else 0,
    1 if PhysActivity == "Yes" else 0,
    1 if Fruits == "Yes" else 0,
    1 if Veggies == "Yes" else 0,
    1 if HvyAlcoholConsump == "Yes" else 0,
    1 if AnyHealthcare == "Yes" else 0,
    1 if NoDocbcCost == "Yes" else 0,
    GenHlth,
    MentHlth,
    PhysHlth,
    1 if DiffWalk == "Yes" else 0,
    1 if Sex == "Male" else 0,
    Age,
    Education,
    Income,
    BMI_Category,
    Age_Category,
    Smoke_Alcohol,
    BMIxAge,
    Lifestyle_Score
]], columns=feature_names)

# ------------ 👁️ PREVIEW DATA --------------

st.divider()
st.subheader("🔍 Data Preview")
st.dataframe(input_data, use_container_width=True)

# -------------------- 🧠 PREDICTION  --------------------
if st.button("🔍 Predict Diabetes Risk"):
    try:
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]
    except ValueError:
        prediction = model.predict(input_data.to_numpy())[0]
        probability = model.predict_proba(input_data.to_numpy())[0][1]

    st.subheader("📊 Prediction Result")
    if prediction == 1:
        st.error(f"⚠️ High Diabetes Risk — ({probability*100:.1f}% probability)")
    else:
        st.success(f"✅ Low Diabetes Risk — ({(1 - probability)*100:.1f}% probability)")

    # -------------------- 🎯 RISK GAUGE CHART --------------------
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

# -------------------- 🧩 SHAP EXPLANATIONS --------------------
st.divider()
st.subheader("🔎 Why This Prediction?")
st.write("Feature contribution visualization (using SHAP values):")

# Ensure input_data is numeric
input_data = input_data.replace(r'^\[|\]$', '', regex=True)
input_data = input_data.apply(pd.to_numeric, errors='coerce')

try:
    booster = model.get_booster()
    explainer = shap.TreeExplainer(booster)
    shap_values = explainer.shap_values(input_data)

    tab1, tab2 = st.tabs(["🌿 Local Explanation", "📊 Global Feature Importance"])

    with tab1:
        st.write("### Local Explanation (Current Input)")
        fig, ax = plt.subplots()
        shap.force_plot(explainer.expected_value, shap_values, input_data, matplotlib=True, show=False)
        st.pyplot(fig, bbox_inches="tight")

    with tab2:
        st.write("### Global Feature Importance")
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        shap.summary_plot(shap_values, input_data, plot_type="bar", show=False)
        st.pyplot(fig2)

except Exception as e:
    st.warning("⚠️ TreeExplainer failed — using fallback SHAP method.")
    st.caption(str(e))

    try:
        # Use model.predict_proba for proper numeric outputs
        predict_fn = lambda x: model.predict_proba(x)[:, 1]
        explainer = shap.Explainer(predict_fn, input_data.to_numpy())
        shap_values = explainer(input_data.to_numpy())

        st.write("### Local Explanation (Fallback)")
        fig3, ax3 = plt.subplots()
        shap.waterfall_plot(shap_values[0])
        st.pyplot(fig3, bbox_inches="tight")

    except Exception as e2:
        st.error(f"❌ SHAP visualization unavailable. Reason: {e2}")
    
    

    # -------------------- SUMMARY METRICS --------------------
    st.markdown("---")
    st.subheader("📈 Model Performance Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", "0.84")
    col2.metric("Recall", "0.85")
    col3.metric("ROC-AUC", "0.91")

st.markdown("---")
st.caption("Final Model: Tuned XGBoost (SMOTE) | Developed by **Kaushlendra Pratap Singh**")
