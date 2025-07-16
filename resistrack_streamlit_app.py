import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import shap
import matplotlib.pyplot as plt

# 1) Prepare & train
data = {
    "Cancer_Type": ["Breast","Lung","Colon"] * 3 + ["Breast"],
    "MDR1_Expression": [0.9,0.2,0.5, 0.8,0.1,0.3, 0.95,0.15,0.4, 0.85],
    "miR21_Level":     [0.85,0.3,0.6, 0.9,0.2,0.4, 0.88,0.25,0.5, 0.87],
    "Prior_Platinum":  [1,0,1, 1,0,0, 1,0,0, 1],
    "WBC_Count":       [3.2,6.5,4.7, 3.0,7.1,5.4, 2.9,6.8,5.1, 3.3],
    "Resistant":       [1,0,0, 1,0,0, 1,0,0, 1]
}
df = pd.DataFrame(data)
df = pd.get_dummies(df, columns=["Cancer_Type"], drop_first=True)
X = df.drop("Resistant", axis=1)
y = df["Resistant"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)
explainer = shap.Explainer(model, X_train)

# 2) Build UI
st.title("🔬 ResisTrack MVP Demo")
st.markdown("Basic proof-of-concept: predict chemo-resistance & show top drivers.")

with st.sidebar:
    st.header("Therapy Scenario")
    cancer    = st.selectbox("Cancer Type",       ["Breast","Lung","Colon"])
    mdr1      = st.slider("MDR1 Expression",     0.0,1.0,0.5,0.01)
    mir21     = st.slider("miR21 Level",         0.0,1.0,0.5,0.01)
    wbc       = st.number_input("WBC (×10⁹ cells/L)", 0.0,20.0,5.0,0.1)
    prior     = st.checkbox("Prior Platinum Therapy")
    drug      = st.selectbox("Treatment / Drug", ["Doxorubicin","Oxaliplatin","5-FU"])
    duration  = st.slider("Duration (weeks)", 0,24,4,1)

if st.button("🔍 Predict"):
    # assemble the input row
    inp = pd.DataFrame({
        "MDR1_Expression":[mdr1],
        "miR21_Level":    [mir21],
        "Prior_Platinum":[1 if prior else 0],
        "WBC_Count":      [wbc],
        "Cancer_Type_Lung":[1 if cancer=="Lung" else 0],
        "Cancer_Type_Colon":[1 if cancer=="Colon" else 0]
    })
    # predict
    p_res = model.predict_proba(inp)[0,1]
    label = "Resistant" if p_res>0.5 else "Sensitive"
    emoji = "🔴" if p_res>0.5 else "🟢"

    st.markdown(f"**Prediction:** {label} to {drug} ({p_res*100:.1f}%) {emoji}")
    st.markdown(f"**Duration:** {duration} weeks")

    # SHAP
    shap_vals = explainer(inp)
    st.subheader("Feature Impact (SHAP)")
    fig, ax = plt.subplots()
    shap.plots.bar(shap_vals, show=False, ax=ax)
    st.pyplot(fig)
