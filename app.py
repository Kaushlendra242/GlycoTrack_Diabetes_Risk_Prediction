import streamlit as st
import numpy as np
import pandas as pd
import joblib
import shap
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# -------------------- 🎯 PAGE CONFIG --------------------
st.set_page_config(
    page_title="GlycoTrack: Diabetes Risk Prediction",
    page_icon="🩺",
    layout="centered"
)

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
MentHlth = st.sidebar.slider("Poor Mental Health Days (Last 30)", 0, 30, 5)
PhysHlth = st.sidebar.slider("Poor Physical Health Days (Last 30)", 0, 30, 5)
DiffWalk = st.sidebar.selectbox("Difficulty Walking?", ["No", "Yes"])
Sex = st.sidebar.selectbox("Sex", ["Female", "Male"])
Age = st.sidebar.slider("Age", 18, 100, 35)
Education = st.sidebar.selectbox("Education Level (1–6)", [1, 2, 3, 4, 5, 6])
Income = st.sidebar.selectbox("Income Level (1–8)", [1, 2, 3, 4, 5, 6, 7, 8])

# Derived features
BMI_Category = st.sidebar.selectbox("BMI Category", [1, 2, 3, 4])
Age_Category = st.sidebar.selectbox("Age Category", [1, 2, 3, 4])
Smoke_Alcohol = st.sidebar.slider("Smoke + Alcohol Score", 0.0, 5.0, 1.0)
BMIxAge = BMI * Age
Lifestyle_Score = st.sidebar.slider("Lifestyle Score", 0.0, 10.0, 5.0)

# -------------------- 📂 FEATURE ORDER --------------------
feature_names = [
    "HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke",
    "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
    "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "GenHlth",
    "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age", "Education",
    "Income", "BMI_Category", "Age_Category", "Smoke_Alcohol",
    "BMIxAge", "Lifestyle_Score"
]

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
    GenHlth, MentHlth, PhysHlth,
    1 if DiffWalk == "Yes" else 0,
    1 if Sex == "Male" else 0,
    Age, Education, Income,
    BMI_Category, Age_Category,
    Smoke_Alcohol, BMIxAge, Lifestyle_Score
]], columns=feature_names)

# -------------------- 🧠 PREDICTION --------------------
if st.button("🔍 Predict Diabetes Risk"):
    prediction = model.predict(input_data.to_numpy())[0]
    probability = model.predict_proba(input_data.to_numpy())[0][1]

    st.subheader("📊 Prediction Result")
    if prediction == 1:
        st.error(f"⚠️ High Diabetes Risk — {probability*100:.1f}%")
    else:
        st.success(f"✅ Low Diabetes Risk — {(1-probability)*100:.1f}%")

    # -------------------- 🎯 RISK GAUGE --------------------
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        title={'text': "Predicted Diabetes Risk (%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "crimson" if prediction else "green"},
            'steps': [
                {'range': [0, 50], 'color': "lightgreen"},
                {'range': [50, 80], 'color': "orange"},
                {'range': [80, 100], 'color': "red"}
            ]
        }
    ))
    st.plotly_chart(fig, use_container_width=True)

    # -------------------- 🧩 SHAP EXPLANATION --------------------
    st.divider()
    st.subheader("🔎 Why This Prediction?")

    @st.cache_resource
    def load_explainer(m):
        return shap.TreeExplainer(m)

    explainer = load_explainer(model)
    shap_values = explainer(input_data.to_numpy())

    # ---- LOCAL EXPLANATION ----
    st.markdown("### 🧠 Local Explanation (Current Input)")
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    shap.plots.waterfall(shap_values[0], feature_names=feature_names, show=False)
    st.pyplot(fig1, clear_figure=True)

    # ---- GLOBAL IMPORTANCE ----
    st.markdown("### 🌍 Global Feature Importance")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    shap.summary_plot(
        explainer.shap_values(input_data.to_numpy()),
        input_data,
        plot_type="bar",
        show=False
    )
    st.pyplot(fig2, clear_figure=True)

    # -------------------- 📈 METRICS --------------------
    st.divider()
    st.subheader("📈 Model Performance Summary")
    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", "0.84")
    c2.metric("Recall", "0.85")
    c3.metric("ROC-AUC", "0.91")

st.markdown("---")
st.caption("Final Model: Tuned XGBoost (SMOTE) | Developed by **Kaushlendra Pratap Singh**")
