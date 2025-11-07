### 🩺 GlycoTrack: Diabetes Risk Prediction

Predict your likelihood of having diabetes using a Tuned **XGBoost (SMOTE)** model deployed via **Streamlit**.

---
### 🚀 Live Demo
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://kaushlendra-glycotrack.streamlit.app)

---

# 🎯 Project Overview

**GlycoTrack** is a machine learning project that predicts the **likelihood of diabetes** using demographic, lifestyle, and clinical health factors.
The goal is to enable **early identification of high-risk individuals** for preventive healthcare interventions and policy planning.

This project leverages advanced data preprocessing, exploratory data analysis (EDA), and machine learning (XGBoost + SMOTE) to build an interpretable and high-performing diabetes risk prediction model.

---
# 🧠 Business Context

Diabetes is one of the fastest-growing health concerns globally.
Accurate prediction helps:

* Healthcare providers design **targeted intervention programs**

* Insurance companies **reduce costs** by identifying high-risk members

* Governments and NGOs **allocate resources** more efficiently
  
---
### 🧩 Project Pipeline

**1. Data Collection & Preparation**

* **Dataset:** BRFSS 2021 (Behavioral Risk Factor Surveillance System)
* **Size:** ~253,680 records
* **Features:** Demographics, lifestyle habits, physical & mental health indicators, clinical metrics

* **Preprocessing:**
      * Missing values (~20–25k) dropped
      * Outliers removed (BMI > 60 or < 10)
      * Encoding categorical variables
      * Feature scaling
      * Balanced data using **SMOTE** (Synthetic Minority Oversampling Technique)

---
### 2. Exploratory Data Analysis (EDA)
   * Target distribution: **14% diabetic, 86% non-diabetic**
   * Strong positive correlation with **BMI, age, physical inactivity, poor general health, high BP, high cholesterol**
   * Weaker correlation with **smoking** and **alcohol consumption**
   * Visualized distributions, correlations, and feature interactions

---
### 3. Modeling Approach

  * Tested models:
    **Logistic Regression, Decision Tree, Random Forest, KNN, SVM, and XGBoost**
  * Evaluation metrics:
    **Accuracy, Precision, Recall, F1-Score, ROC-AUC**
  * Addressed class imbalance with **SMOTE**
  * Final model selected: **XGBoost (after SMOTE balancing)**

---
### 4.📈 Model Performance

| Model               | Accuracy | Precision |  Recall  |    F1    |  ROC-AUC |
| :------------------ | :------: | :-------: | :------: | :------: | :------: |
| Logistic Regression |   0.85   |    0.68   |   0.54   |   0.60   |   0.79   |
| Decision Tree       |   0.81   |    0.53   |   0.62   |   0.57   |   0.72   |
| Random Forest       |   0.86   |    0.71   |   0.59   |   0.64   |   0.82   |
| KNN                 |   0.84   |    0.60   |   0.48   |   0.53   |   0.74   |
| SVM                 |   0.85   |    0.67   |   0.56   |   0.61   |   0.80   |
| XGBoost (Baseline)  |   0.87   |    0.72   |   0.61   |   0.66   |   0.84   |
| **XGBoost (SMOTE)** | **0.84** |  **0.82** | **0.85** | **0.83** | **0.91** |


**🩵 Key takeaway:** After applying SMOTE, recall improved from **61% → 85%**, which is critical in healthcare (minimizing false negatives).

---

### 5. Feature Importance & Interpretability

 * Top features influencing diabetes risk:
    * BMI
    * Age group
    * General Health
    * Physical Activity
    * HighBP and HighChol

* Feature importance and SHAP analysis align with clinical expectations.

---

### 🧠 6. Tools & Technologies

| Category              | Tools                                                                       |
| --------------------- | --------------------------------------------------------------------------- |
| Programming           | Python                                                                      |
| Libraries             | Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, XGBoost, Imbalanced-learn |
| Visualization         | Matplotlib                                                                  |
| Deployment (optional) | Streamlit                                                                   |
| Documentation         | Jupyter Notebook, Word, PDF                                                 |

---
### 📈 Results Summary

 - **Best Model:** XGBoost (SMOTE)
 - **ROC-AUC:** 0.91
 - **Recall:** 0.85
 - **F1-score:** 0.83
 - **Interpretation:** High recall ensures minimal false negatives — critical for preventive healthcare screening.

---

### 📂 Project Structure
📦 GlycoTrack_Diabetes_Risk_Prediction
 ┣ 📜 app.py
 ┣ 📜 final_glycotrack_model.pkl
 ┣ 📜 requirements.txt
 ┣ 📜 Executive_Summary_Report.pdf
 ┗ 📜 README.md

| File                                              | Description                      |
| ------------------------------------------------- | -------------------------------- |
| `GlycoTrack_Predicting_Diabetes_Risk.ipynb`       | Baseline model (imbalanced data) |
| `GlycoTrack_Predicting_Diabetes_Risk_SMOTE.ipynb` | Balanced model (SMOTE applied)   |
| `Executive_Summary_Report.pdf`                    | Final stakeholder report         |
| `app.py` *(optional)*                             | Streamlit app for live model     |
| `README.md`                                       | Project documentation            |
| `requirements.txt`                                | Required libraries               |

---

### 💡 Features
- Interactive Streamlit web app  
- Predicts diabetes risk with probability gauge  
- SHAP-based feature explainability  
- Tuned XGBoost model (SMOTE-balanced)  
- Clean, deployable structure  

---

### 🏁 How to Run Locally
1. Clone this repository  
   git clone https://github.com/<your-username>/GlycoTrack_Diabetes_Risk_Prediction.git  
   cd GlycoTrack_Diabetes_Risk_Prediction  

2. Install dependencies  
   pip install -r requirements.txt  

3. Run the Streamlit app  
   streamlit run app.py  

---

### 👨‍💻 Author
**Kaushlendra Pratap Singh**  
Data Analyst | Data Science Enthusiast | Optum SME  
📍 New Delhi, India  
[LinkedIn](https://www.linkedin.com/in/kaushlendra-singh-51397655/) | [GitHub](https://github.com/Kaushlendra242)
