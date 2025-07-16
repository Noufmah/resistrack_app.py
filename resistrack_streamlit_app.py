import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import shap
import matplotlib.pyplot as plt

# --- Data Preparation and Model Training ---
data = {
    "Cancer_Type": ["Breast", "Lung", "Colon", "Breast", "Lung", "Colon", "Breast", "Lung", "Colon", "Breast"],
    "MDR1_Expression": [0.9, 0.2, 0.5, 0.8, 0.1, 0.3, 0.95, 0.15, 0.4, 0.85],
    "miR21_Level":      [0.85, 0.3, 0.6, 0.9, 0.2, 0.4, 0.88, 0.25, 0.5, 0.87],
    "Prior_Platinum_Therapy": [1, 0, 1, 1, 0, 0, 1, 0, 0, 1],
    "WBC_Count":        [3.2, 6.5, 4.7, 3.0, 7.1, 5.4, 2.9, 6.8, 5.1, 3.3],
    "Resistance_Label": [1, 0, 0, 1, 0, 0, 1, 0, 0, 1]
}
df = pd.DataFrame(data)
df = pd.get_dummies(df, columns=["Cancer_Type"], drop_first=True)
X = df.drop("Resistance_Label", axis=1)
y = df["Resistance_Label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)
explainer = shap.Explainer(model, X_train)

# --- Streamlit UI ---
st.title("🔬 ResisTrack MVP Demo")
st.markdown("**Interactive Clinical Interface** – Predict drug resistance and view confidence.")

with st.sidebar:
    st.header("🧪 Therapy Inputs")
    cancer_type = st.selectbox("Cancer Type", ["Breast", "Lung", "Colon"])
    MDR1         = st.slider("MDR1 Expression", 0.0, 1.0, 0.5, 0.01)
    miR21        = st.slider("miR21 Level",    0.0, 1.0, 0.5, 0.01)
    WBC          = st.number_input("WBC Count (×10⁹ cells/L)", 0.0, 20.0, 5.0, 0.1)
    prior        = st.checkbox("Prior Platinum Therapy")

if st.button("Predict"):
    # 1) Build input DataFrame & one-hot encode cancer type
    input_df = pd.DataFrame({
        "MDR1_Expression": [MDR1],
        "miR21_Level":      [miR21],
        "Prior_Platinum_Therapy": [1 if prior else 0],
        "WBC_Count":        [WBC],
        "Cancer_Type":      [cancer_type],
    })
    input_enc = pd.get_dummies(input_df, columns=["Cancer_Type"], drop_first=True)
    for col in X_train.columns:
        if col not in input_enc.columns:
            input_enc[col] = 0
    input_enc = input_enc[X_train.columns]

    # 2) Predict & probability
    pred = model.predict(input_enc)[0]
    prob = model.predict_proba(input_enc)[0,1]
    label = "🔴 Resistant" if pred==1 else "🟢 Sensitive"
    st.markdown(f"**Prediction:** {label} ({prob*100:.1f}% confidence)")

    # 3) SHAP values & top feature
    shap_values = explainer(input_enc)
    top_feature = input_enc.columns[np.abs(shap_values.values[0]).argmax()]
    st.markdown(f"**Top Feature:** {top_feature}")

    # 4) SHAP bar chart
    st.subheader("Feature Impact (SHAP Values)")
    fig, ax = plt.subplots()
    shap.plots.bar(shap_values, show=False, ax=ax)
    st.pyplot(fig)
