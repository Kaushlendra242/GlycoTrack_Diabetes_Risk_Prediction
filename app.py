import streamlit as st
import pandas as pd
import numpy as np
import shap
import joblib
import matplotlib.pyplot as plt
import xgboost as xgb
from pathlib import Path

# --------------------------

# 🎯 PAGE CONFIG

# --------------------------

st.set_page_config(
    page_title="GlycoTrack – Diabetes Risk Predictor",
    layout="wide",
    page_icon="🩸",
)

st.title("🩺 GlycoTrack – Predicting Diabetes Risk")
st.write("Upload patient biomarker data to estimate diabetes risk using the trained XGBoost model.")

# --------------------------

# 📂 LOAD MODEL + FEATURES

# --------------------------

@st.cache_resource
def load_model_and_features():
    model = joblib.load("xgboost_final_model.pkl")  # your saved XGBoost model
    df_ref = pd.read_csv("diabetes_prepared.csv")
    features = df_ref.drop(columns=["Outcome"], errors="ignore").columns.tolist()
    return model, features

try:
    model, feature_order = load_model_and_features()
except Exception as e:
    st.error(f"❌ Failed to load model or dataset reference: {e}")
    st.stop()

# --------------------------

# 📥 DATA INPUT

# --------------------------

st.subheader("📤 Upload or Enter Data")

uploaded_file = st.file_uploader("Upload a CSV file (must have the same columns as the training dataset)", type=["csv"])

if uploaded_file:
    input_data = pd.read_csv(uploaded_file)
    st.success(f"✅ Uploaded {uploaded_file.name}")
else:
    # Manual data entry as fallback
    st.info("Or enter patient data manually below:")
    input_data = pd.DataFrame(columns=feature_order)
    for col in feature_order:
        val = st.number_input(f"{col}", value=0.0)
        input_data.loc[0, col] = val

# Ensure feature order consistency

try:
    input_data = input_data[feature_order]
except KeyError as e:
    st.error(f"❌ Uploaded file missing columns: {e}")
    st.stop()

# --------------------------
# 👁️ PREVIEW DATA
# --------------------------
st.divider()
st.subheader("🔍 Data Preview")
st.dataframe(input_data, use_container_width=True)

# --------------------------

# 🧠 PREDICTION

# --------------------------

if st.button("🔮 Predict Diabetes Risk"):
    try:
        pred_prob = model.predict_proba(input_data)[:, 1]
        pred_class = model.predict(input_data)

        st.success(f"**Predicted Probability of Diabetes:** {pred_prob[0]:.3f}")
        st.info(f"**Predicted Class:** {'Positive' if pred_class[0] == 1 else 'Negative'}")

        # --------------------------
        # 🎯 RISK GAUGE CHART
        # --------------------------
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pred_prob[0] * 100,
            title={'text': "Predicted Diabetes Risk (%)"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': 'crimson' if pred_class[0] == 1 else 'green'},
                'steps': [
                    {'range': [0, 50], 'color': 'lightgreen'},
                    {'range': [50, 80], 'color': 'orange'},
                    {'range': [80, 100], 'color': 'red'}
                ]
            }
        ))
        st.plotly_chart(fig_gauge, use_container_width=True)

        # --------------------------
        # 🧩 SHAP EXPLANATIONS
        # --------------------------
        st.divider()
        st.subheader("🔎 Feature Contribution Visualization (SHAP Values)")

        try:
            # Try using native XGBoost booster
            booster = model.get_booster()
            explainer = shap.TreeExplainer(booster)
            shap_values = explainer.shap_values(input_data)

            st.write("### Local Explanation (Current Input)")
            fig, ax = plt.subplots()
            shap.force_plot(explainer.expected_value, shap_values, input_data, matplotlib=True, show=False)
            st.pyplot(fig, bbox_inches="tight")

            st.write("### Global Feature Importance (Sample-Based)")
            fig2, ax2 = plt.subplots(figsize=(8, 6))
            shap.summary_plot(shap_values, input_data, plot_type="bar", show=False)
            st.pyplot(fig2)

        except Exception as e:
            st.warning("⚠️ TreeExplainer failed — using fallback SHAP method.")
            st.caption(str(e))

            try:
                # Use numeric probabilities and NumPy for fallback
                predict_fn = lambda x: model.predict_proba(x)[:, 1]
                explainer = shap.Explainer(predict_fn, input_data.to_numpy())
                shap_values = explainer(input_data.to_numpy())

                st.write("### Local Explanation (Fallback)")
                fig3, ax3 = plt.subplots()
                shap.waterfall_plot(shap_values[0])
                st.pyplot(fig3, bbox_inches="tight")

            except Exception as e2:
                st.error(f"❌ SHAP visualization unavailable. Reason: {e2}")

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")

# --------------------------
# 🧾 FOOTER
# --------------------------
st.divider()
st.caption("Developed with ❤️ by GlycoTrack AI – XGBoost model deployed for web.")
