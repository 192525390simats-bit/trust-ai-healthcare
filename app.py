import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# -----------------------------
# Title
# -----------------------------
st.title("Trust-Aware AI Clinical Decision Support System")
st.markdown("### Reducing Diagnostic Errors using Explainable AI")

# -----------------------------
# Simulated realistic dataset
# -----------------------------
np.random.seed(42)

data = pd.DataFrame({
    'Glucose': np.random.normal(120, 25, 300),
    'BMI': np.random.normal(27, 5, 300),
    'Age': np.random.randint(20, 70, 300),
    'BP': np.random.normal(80, 10, 300),
})

# Create meaningful outcome (not random)
data['Outcome'] = (
    (data['Glucose'] > 130).astype(int) +
    (data['BMI'] > 30).astype(int) +
    (data['Age'] > 45).astype(int)
)

data['Outcome'] = (data['Outcome'] > 1).astype(int)

X = data[['Glucose','BMI','Age','BP']]
y = data['Outcome']

# -----------------------------
# Scaling
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# Model Training
# -----------------------------
model = LogisticRegression()
model.fit(X_scaled, y)

# -----------------------------
# Input Section
# -----------------------------
st.header("Enter Patient Details")

col1, col2 = st.columns(2)

with col1:
    glucose = st.number_input("Glucose Level", 50, 200, 110)
    bmi = st.number_input("BMI", 10.0, 50.0, 25.0)

with col2:
    age = st.number_input("Age", 1, 100, 30)
    bp = st.number_input("Blood Pressure", 40, 140, 80)

# -----------------------------
# Validation
# -----------------------------
if glucose == 0 or bmi == 0:
    st.warning("Please enter valid medical values")

# -----------------------------
# Prediction
# -----------------------------
if st.button("Run Diagnosis"):

    input_data = np.array([[glucose, bmi, age, bp]])
    input_scaled = scaler.transform(input_data)

    prob = model.predict_proba(input_scaled)[0][1]
    risk = int(prob * 100)

    confidence = max(prob, 1 - prob) * 100

    # -----------------------------
    # Output
    # -----------------------------
    st.subheader("Diagnostic Output")

    st.metric("Risk Percentage", f"{risk}%")
    st.progress(risk / 100)

    st.metric("Model Confidence", f"{int(confidence)}%")

    # -----------------------------
    # Explainability (Advanced)
    # -----------------------------
    st.subheader("Explainability (Feature Contribution)")

    coefficients = model.coef_[0]
    feature_names = ['Glucose','BMI','Age','BP']

    contributions = input_scaled[0] * coefficients

    explanation_df = pd.DataFrame({
        "Feature": feature_names,
        "Contribution": contributions
    }).sort_values(by="Contribution", ascending=False)

    st.dataframe(explanation_df)

    # -----------------------------
    # Trust Score (Advanced ⭐)
    # -----------------------------
    completeness = 100 if all([glucose, bmi, age, bp]) else 50
    stability = 100 - np.std(contributions)*10  # variation measure

    trust_score = int((confidence * 0.5) + (completeness * 0.2) + (stability * 0.3))

    st.subheader("Trust Evaluation")
    st.metric("Trust Score", f"{trust_score}%")

    # -----------------------------
    # Risk Flagging
    # -----------------------------
    if confidence < 65:
        st.warning("⚠ Low confidence prediction. Additional clinical tests recommended.")

    if trust_score < 60:
        st.error("⚠ Low trust score. Prediction may not be reliable.")

    # -----------------------------
    # Recommendation System
    # -----------------------------
    st.subheader("Clinical Recommendation")

    if risk > 75:
        st.error("High Risk: Immediate medical consultation required.")
    elif risk > 50:
        st.warning("Moderate Risk: Lifestyle changes and monitoring advised.")
    else:
        st.success("Low Risk: Maintain healthy lifestyle.")

    # -----------------------------
    # Insight
    # -----------------------------
    st.subheader("AI Insight")
    st.write("This system integrates explainability and trust evaluation to support safer diagnostic decisions.")
