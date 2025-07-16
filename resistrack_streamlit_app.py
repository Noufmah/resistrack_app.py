import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import shap
import matplotlib.pyplot as plt
import streamlit_shap as st_shap

# ——————————————————————
# 1. Build & train a tiny demo model
# ——————————————————————
data = {
    "Cancer_Type": ["Breast","Lung","Colon","Breast","Lung","Colon","Breast","Lung","Colon","Breast"],
    "MDR1_Expression": [0.9,0.2,0.5,0.8,0.1,0.3,0.95,0.15,0.4,0.85],
    "miR21_Level":     [0.85,0.3,0.6,0.9,0.2,0.4,0.88,0.25,0.5,0.87],
    "Prior_Platinum_Therapy": [1,0,1,1,0,0,1,0,0,1],
    "WBC_Count":       [3.2,6.5,4.7,3.0,7.1,5.4,2.9,6.8,5.1,3.3],
    "Resistance_Label":[1,0,0,1,0,0,1,0,0,1]
}
df = pd.DataFrame(data)
df = pd.get_dummies(df, columns=["Cancer_Type"], drop_first=True)
X = df.drop("Resistance_Label", axis=1)
y = df["Resistance_Label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
model = LogisticRegression()
model.fit(X_train, y_train)

explainer = shap.Explainer(model, X_train)

# ——————————————————————
# 2. Sidebar: get inputs from user
# ——————————————————————
st.title("🔬 ResisTrack MVP Demo")
st.markdown(
    "Basic proof-of-concept interface — predict resistance and see feature impacts."
)

with st.sidebar:
    st.header("🧪 Therapy Inputs")
    cancer_type = st.selectbox("Cancer Type", ["Breast","Lung","Colon"])
    MDR1    = st.slider("MDR1 Expression",   0.0, 1.0, 0.5, 0.01)
    miR21   = st.slider("miR21 Level",       0.0, 1.0, 0.5, 0.01)
    WBC     = st.number_input("WBC Count (×10⁹ cells/L)", 0.0, 20.0, 5.0, 0.1)
    prior   = st.checkbox("Prior Platinum Therapy")
    drug    = st.selectbox("Treatment / Drug", ["Doxorubicin","Oxaliplatin","5-FU"])
    # If you want duration:
    # duration_weeks = st.slider("Duration (weeks)", 0, 24, 4, 1)

# ——————————————————————
# 3. When “Predict” is clicked: encode, predict, explain
# ——————————————————————
if st.button("Predict"):

    # a) build input dataframe + one‐hot encode cancer type
    input_df = pd.DataFrame({
        "MDR1_Expression":       [MDR1],
        "miR21_Level":           [miR21],
        "Prior_Platinum_Therapy":[1 if prior else 0],
        "WBC_Count":             [WBC]
    })
    input_df = pd.concat(
        [input_df, pd.DataFrame({"Cancer_Type":[cancer_type]})], axis=1
    )
    input_enc = pd.get_dummies(input_df, columns=["Cancer_Type"], drop_first=True)
    # ensure all training columns exist
    for col in X_train.columns:
        if col not in input_enc.columns:
            input_enc[col] = 0
    input_enc = input_enc[X_train.columns]

    # b) predict & probability
    pred = model.predict(input_enc)[0]
    prob = model.predict_proba(input_enc)[0,1]
    label = "Resistant" if pred==1 else "Sensitive"
    emoji = "🔴" if pred==1 else "🟢"

    # c) compute SHAP values and top feature
    shap_values = explainer(input_enc)
    top_feature = input_enc.columns[np.abs(shap_values.values[0]).argmax()]

    # — display summary
    st.markdown(f"### Prediction:")
    st.write(f"**{label} to {drug} ({prob*100:.1f}% risk)** {emoji}")
    st.markdown(f"**Top driver feature:** `{top_feature}`")

    # — bar chart of absolute mean SHAP impact
    st.subheader("Feature Impact (SHAP Values)")
    fig, ax = plt.subplots()
    shap.plots.bar(shap_values, show=False, ax=ax)
    st.pyplot(fig)

    # — detailed force plot inside an expander
    with st.expander("🔍 Show detailed feature contributions"):
        force_plot = shap.force_plot(
            explainer.expected_value,
            shap_values.values[0],
            input_enc.iloc[0],
            matplotlib=False
        )
        st_shap(force_plot, height=300)
