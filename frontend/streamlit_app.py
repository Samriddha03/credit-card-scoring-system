import streamlit as st
import requests

FASTAPI_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Credit Default Scoring System", page_icon="💳", layout="wide")

st.title("💳 Credit Default Scoring System (UCI Taiwan Dataset)")
st.write("Predict whether a credit card client will default next month.")

with st.form("credit_form"):
    st.subheader("Client Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        limit_bal = st.number_input("Limit Balance ($)", min_value=1000, max_value=1000000, value=50000)
        sex = st.selectbox("Sex", options=[1, 2], format_func=lambda x: "Male" if x == 1 else "Female")
        education = st.selectbox("Education", options=[1, 2, 3, 4], format_func=lambda x: {1: "Graduate", 2: "University", 3: "High School", 4: "Others"}[x])
        marriage = st.selectbox("Marriage", options=[1, 2, 3], format_func=lambda x: {1: "Married", 2: "Single", 3: "Others"}[x])
        age = st.number_input("Age", min_value=18, max_value=100, value=30)

    with col2:
        st.write("**Repayment Status, last 6 months (-1=Pay duly, 1+=Months Delayed)**")
        pay_0 = st.slider("Pay Status (Most recent month)", -2, 8, 0)
        pay_2 = st.slider("Pay Status (2 months ago)", -2, 8, 0)
        pay_3 = st.slider("Pay Status (3 months ago)", -2, 8, 0)
        pay_4 = st.slider("Pay Status (4 months ago)", -2, 8, 0)
        pay_5 = st.slider("Pay Status (5 months ago)", -2, 8, 0)
        pay_6 = st.slider("Pay Status (6 months ago)", -2, 8, 0)

    with col3:
        st.write("**Bill Amount, last 6 months ($)**")
        bill_amt1 = st.number_input("Bill Amount (Most recent month)", value=2000.0, step=100.0)
        bill_amt2 = st.number_input("Bill Amount (2 months ago)", value=2000.0, step=100.0)
        bill_amt3 = st.number_input("Bill Amount (3 months ago)", value=2000.0, step=100.0)
        bill_amt4 = st.number_input("Bill Amount (4 months ago)", value=2000.0, step=100.0)
        bill_amt5 = st.number_input("Bill Amount (5 months ago)", value=2000.0, step=100.0)
        bill_amt6 = st.number_input("Bill Amount (6 months ago)", value=2000.0, step=100.0)

    st.divider()
    st.write("**Payment Amount, last 6 months ($)**")
    pcol1, pcol2, pcol3, pcol4, pcol5, pcol6 = st.columns(6)
    with pcol1:
        pay_amt1 = st.number_input("Paid (Most recent)", value=500.0, step=100.0)
    with pcol2:
        pay_amt2 = st.number_input("Paid (2 mo. ago)", value=500.0, step=100.0)
    with pcol3:
        pay_amt3 = st.number_input("Paid (3 mo. ago)", value=500.0, step=100.0)
    with pcol4:
        pay_amt4 = st.number_input("Paid (4 mo. ago)", value=500.0, step=100.0)
    with pcol5:
        pay_amt5 = st.number_input("Paid (5 mo. ago)", value=500.0, step=100.0)
    with pcol6:
        pay_amt6 = st.number_input("Paid (6 mo. ago)", value=500.0, step=100.0)

    submit = st.form_submit_button("Evaluate Default Risk")

if submit:
    payload = {
        "LIMIT_BAL": limit_bal, "SEX": sex, "EDUCATION": education, "MARRIAGE": marriage, "AGE": age,
        "PAY_0": pay_0, "PAY_2": pay_2, "PAY_3": pay_3, "PAY_4": pay_4, "PAY_5": pay_5, "PAY_6": pay_6,
        "BILL_AMT1": bill_amt1, "BILL_AMT2": bill_amt2, "BILL_AMT3": bill_amt3, "BILL_AMT4": bill_amt4, "BILL_AMT5": bill_amt5, "BILL_AMT6": bill_amt6,
        "PAY_AMT1": pay_amt1, "PAY_AMT2": pay_amt2, "PAY_AMT3": pay_amt3, "PAY_AMT4": pay_amt4, "PAY_AMT5": pay_amt5, "PAY_AMT6": pay_amt6
    }

    try:
        response = requests.post(f"{FASTAPI_URL}/predict", json=payload)
        if response.status_code == 200:
            res = response.json()
            label = res["risk_label"]
            prob = res["risk_probability"]

            if res["risk_prediction"] == 0:
                st.success(f"**Result:** {label} (Low Probability of Default)")
            else:
                st.error(f"**Result:** {label} (High Risk)")

            st.metric("Default Probability", f"{prob * 100:.1f}%")
            st.progress(min(max(float(prob), 0.0), 1.0))
        else:
            st.error(f"Error ({response.status_code}): {response.text}")
    except Exception as e:
        st.error(f"Could not connect to FastAPI server: {e}")

# --- Health Check Sidebar Indicator ---
try:
    health_resp = requests.get(f"{FASTAPI_URL}/docs", timeout=2)
    if health_resp.status_code == 200:
        st.sidebar.success("Backend API: Online 🟢")
    else:
        st.sidebar.warning("Backend API: Issues Detected 🟡")
except Exception:
    st.sidebar.error("Backend API: Offline 🔴")