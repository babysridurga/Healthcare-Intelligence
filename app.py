import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Healthcare Intelligence", layout="centered")

# Load data
df = pd.read_csv("data/healthcare_data.csv")

# Load model
model = joblib.load("model.pkl")
le_dept = joblib.load("le_dept.pkl")
le_ins = joblib.load("le_ins.pkl")

st.title("🏥 Healthcare Revenue Intelligence Dashboard")

# KPIs
col1, col2, col3 = st.columns(3)

col1.metric("Total Revenue Leakage", f"${int(df['Revenue_Leakage'].sum())}")
col2.metric("Average Delay (Days)", int(df["Payment_Delay_Days"].mean()))
col3.metric("Denial Rate", f"{round(df['Claim_Denied'].mean()*100, 2)}%")

st.divider()

# Charts
st.subheader("Revenue Leakage by Department")
# st.bar_chart(df.groupby("Department")["Revenue_Leakage"].sum())
fig = px.bar(
    df.groupby("Department")["Revenue_Leakage"].sum().reset_index(),
    x="Department",
    y="Revenue_Leakage",
    title="Revenue Leakage by Department"
)
st.plotly_chart(fig, use_container_width=True)


st.subheader("Payment Delay Distribution")
st.line_chart(df["Payment_Delay_Days"])

st.divider()

# Prediction Section
st.subheader("🔮 Claim Denial Risk Prediction")

dept = st.selectbox("Department", le_dept.classes_)
insurance = st.selectbox("Insurance Type", le_ins.classes_)
claim_amount = st.number_input("Claim Amount", min_value=1000)
delay = st.number_input("Payment Delay Days", min_value=0)

if st.button("Predict Risk"):
    dept_enc = le_dept.transform([dept])[0]
    ins_enc = le_ins.transform([insurance])[0]

    input_data = np.array([[dept_enc, ins_enc, claim_amount, delay]])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f"⚠ High Risk of Claim Denial ({round(probability*100, 2)}%)")
    else:
        st.success(
            f"✅ Low Risk of Claim Denial ({round(probability*100, 2)}%)")
