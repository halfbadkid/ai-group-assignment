# =========================
# AI LOAN APPROVAL SYSTEM
# End-to-End Streamlit App
# =========================

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(
    page_title="💳 AI Loan Approval System",
    layout="wide"
)

# -------------------------
# UNIVERSITY LOGO (Header)
# -------------------------
col1, col2, col3, col4, col5 = st.columns([1, 1, 2, 1, 1])
with col3:
    st.image("umpsa_logo.png", use_container_width=True)

st.markdown("<h1 style='text-align: center;'>💳 AI Loan Approval System</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; color: gray;'>"
    "An accurate and unbiased loan approval platform that empowers users to assess eligibility "
    "through transparent and trustworthy decision-making."
    "</p>",
    unsafe_allow_html=True
)
st.markdown("---")

# -------------------------
# DATA LOADING & CLEANING
# -------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("loan_data.csv")
    df.columns = df.columns.str.strip()

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Current_loan_status"])
    df = df.fillna(df.median(numeric_only=True))
    return df

df = load_data()

# -------------------------
# FEATURES / TARGET
# -------------------------
X = df.drop(["Current_loan_status", "home_ownership"], axis=1)
y = df["Current_loan_status"]

# -------------------------
# MODEL TRAINING
# -------------------------
@st.cache_resource
def train_model(X, y):
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced"
    )
    model.fit(X, y)
    return model

rf_model = train_model(X, y)

# -------------------------
# FEATURE IMPORTANCE
# -------------------------
st.subheader("🔍 Top 5 Feature Importance")

fi = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf_model.feature_importances_
}).sort_values(by="Importance", ascending=False).head(5)

st.bar_chart(fi.set_index("Feature"))

# -------------------------
# FAIRNESS ANALYSIS
# -------------------------
st.subheader("⚖️ Fairness Check (Historical Default)")

fair_df = X.copy()
fair_df["Prediction"] = rf_model.predict(X)
fair_df["Historical Default"] = fair_df["historical_default"].map(
    {0: "NO", 1: "Unknown", 2: "YES"}
)

approval_rate = fair_df.groupby("Historical Default")["Prediction"].mean()
st.table(approval_rate)

# -------------------------
# LOAN ELIGIBILITY FORM
# -------------------------
st.subheader("🧾 Loan Eligibility Checker")

with st.form("loan_form"):
    customer_age = st.slider("Customer Age", 18, 70, 30)
    customer_income = st.number_input("Annual Income", min_value=0.0, value=30000.0)
    employment_duration = st.number_input("Employment Duration (months)", 0, 600, 60)
    cred_hist_length = st.number_input("Credit History Length (years)", 0, 40, 5)

    historical_default = st.selectbox(
        "Historical Default",
        ["NO", "Unknown", "YES"]
    )

    loan_amnt = st.number_input("Loan Amount", min_value=0.0, value=10000.0)
    loan_int_rate = st.slider("Interest Rate (%)", 3.0, 15.0, 7.5)
    term_years = st.selectbox("Loan Term (years)", list(range(1, 11)))

    loan_intent = st.selectbox(
        "Loan Intent",
        ["Debt Consolidation", "Education", "Home Improvement", "Medical", "Personal"]
    )

    submit = st.form_submit_button("Check Eligibility")

# -------------------------
# PREDICTION & OUTPUT
# -------------------------
if submit:

    if customer_income <= 0 or loan_amnt <= 0:
        st.error("⚠️ Annual income and loan amount must be greater than zero.")
    else:
        hist_map = {"NO": 0, "Unknown": 1, "YES": 2}
        intent_map = {
            "Debt Consolidation": 0,
            "Education": 1,
            "Home Improvement": 2,
            "Medical": 3,
            "Personal": 4
        }

        # Ensure feature alignment
        user_df = pd.DataFrame([{
            "customer_age": customer_age,
            "customer_income": customer_income,
            "employment_duration": employment_duration,
            "loan_intent": intent_map[loan_intent],
            "loan_grade": 2,  # neutral placeholder for consistency
            "loan_amnt": loan_amnt,
            "loan_int_rate": loan_int_rate,
            "term_years": term_years,
            "historical_default": hist_map[historical_default],
            "cred_hist_length": cred_hist_length
        }])

        # Predict probability FIRST
        prob = rf_model.predict_proba(user_df)[0][1]

        # Apply realistic financial threshold
        approval_threshold = 0.40
        pred = 1 if prob >= approval_threshold else 0

        # -------------------------
        # LOAN GRADE
        # -------------------------
        if prob >= 0.85:
            grade = "Grade A – Lowest Risk"
        elif prob >= 0.70:
            grade = "Grade B – Low Risk"
        elif prob >= 0.55:
            grade = "Grade C – Moderate Risk"
        elif prob >= 0.40:
            grade = "Grade D – High Risk"
        else:
            grade = "Grade E – Highest Risk"

        st.markdown("---")
        st.subheader("📌 Loan Decision Result")
        st.metric("Approval Probability", f"{prob:.2%}")
        st.write(f"**AI-Assigned Loan Grade:** {grade}")

        if pred == 1:
            st.success("✅ Loan Approved")
            st.balloons()
        else:
            st.error("❌ Loan Rejected")

            st.subheader("💡 Key Risk Factors Identified")
            reasons = []

            if hist_map[historical_default] > 0:
                reasons.append("• Previous default history increases credit risk.")
            if customer_income < (loan_amnt / 4):
                reasons.append("• Loan amount is high relative to annual income.")
            if employment_duration < 12:
                reasons.append("• Short employment duration suggests income instability.")
            if loan_int_rate > 12:
                reasons.append("• Higher interest rate increases repayment risk.")

            if not reasons:
                reasons.append("• Overall financial profile does not meet approval criteria.")

            for r in reasons:
                st.write(r)

# -------------------------
# FOOTER
# -------------------------
st.markdown("---")
st.caption(
    "AI Loan Approval System demonstrating predictive modeling, explainable AI, "
    "fairness evaluation, and real-world deployment using Streamlit."
)