import streamlit as st
import pickle
import pandas as pd
import numpy as np
import time

# --- 1. PAGE CONFIG ---
st.set_page_config(
    page_title="FinSafe AI | Loan Underwriting Portal",
    page_icon="🏦",
    layout="wide"
)

# --- 2. PROFESSIONAL BANKING CSS ---
st.markdown("""
    <style>
    .stApp { background-color: #f4f7f9; }
    
    /* Main Card Container */
    .bank-card {
        background-color: #ffffff;
        padding: 40px;
        border-radius: 15px;
        border-top: 10px solid #1e3a8a;
        box-shadow: 0 10px 25px rgba(0,0,0,0.05);
        color: #1e293b;
    }
    
    /* Headers */
    .main-title { color: #1e3a8a; font-weight: 800; font-size: 32px; }
    .section-header { color: #334155; font-weight: 700; border-bottom: 2px solid #e2e8f0; margin-bottom: 20px; }
    
    /* Result Boxes */
    .status-box { padding: 25px; border-radius: 10px; text-align: center; margin-top: 20px; }
    .approved { background-color: #ecfdf5; border: 2px solid #10b981; color: #065f46; }
    .rejected { background-color: #fef2f2; border: 2px solid #ef4444; color: #991b1b; }
    </style>
    """, unsafe_allow_html=True)

# --- 3. LOAD ASSETS ---
@st.cache_resource
def load_loan_assets():
    try:
        model = pickle.load(open('loan_model.pkl', 'rb'))
        scaler = pickle.load(open('loan_scaler.pkl', 'rb'))
        encoders = pickle.load(open('loan_encoders.pkl', 'rb'))
        return model, scaler, encoders
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

model, scaler, encoders = load_loan_assets()

# --- 4. SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2830/2830284.png", width=100)
    st.markdown("### Decision Engine Settings")
    threshold = st.slider("Risk Tolerance Threshold", 0.0, 1.0, 0.50)
    st.info("System Version: 2.5.0\nNode: Pakistan-South-1")

# --- 5. MAIN UI ---
st.markdown('<p class="main-title">🏦 FINSAFE AI: CORPORATE LOAN UNDERWRITING</p>', unsafe_allow_html=True)
st.markdown("<p style='color: #64748b;'>Automated Credit Risk Assessment for Financial Institutions</p>", unsafe_allow_html=True)

if model is None:
    st.stop()

st.markdown('<div class="bank-card">', unsafe_allow_html=True)

# Using 3 columns for a clean data-entry experience
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<p class="section-header">👤 Applicant Profile</p>', unsafe_allow_html=True)
    no_of_dep = st.number_input("Dependents", 0, 15, 0)
    edu = st.selectbox("Education Level", encoders['education'].classes_)
    self_emp = st.selectbox("Self Employed Status", encoders['self_employed'].classes_)

with col2:
    st.markdown('<p class="section-header">💰 Financials (PKR)</p>', unsafe_allow_html=True)
    income = st.number_input("Annual Income", min_value=0, value=800000)
    loan_amt = st.number_input("Loan Amount Requested", min_value=0, value=500000)
    loan_term = st.number_input("Term (Years)", 1, 40, 5)

with col3:
    st.markdown('<p class="section-header">🛡️ Risk Factors</p>', unsafe_allow_html=True)
    cibil = st.slider("Credit (CIBIL) Score", 300, 900, 700)
    res_asset = st.number_input("Residential Assets", value=1000000)
    com_asset = st.number_input("Commercial Assets", value=0)
    lux_asset = st.number_input("Luxury Assets", value=0)
    bank_asset = st.number_input("Bank Assets", value=200000)

# --- 6. PREDICTION LOGIC ---
if st.button("EXECUTE RISK ANALYSIS"):
    with st.spinner("Analyzing creditworthiness against 4,300 historical benchmarks..."):
        time.sleep(1.5)
        
        # Prepare Input Data (Must match Training Column Order exactly)
        input_data = pd.DataFrame({
            'no_of_dependents': [no_of_dep],
            'education': [edu],
            'self_employed': [self_emp],
            'income_annum': [income],
            'loan_amount': [loan_amt],
            'loan_term': [loan_term],
            'cibil_score': [cibil],
            'residential_assets_value': [res_asset],
            'commercial_assets_value': [com_asset],
            'luxury_assets_value': [lux_asset],
            'bank_asset_value': [bank_asset]
        })

        # Apply Encoders
        for col, le in encoders.items():
            if col in input_data.columns:
                input_data[col] = le.transform(input_data[col].astype(str))
        
        # Scale and Predict
        input_scaled = scaler.transform(input_data)
        prob = model.predict_proba(input_scaled)[0][1]
        
        # Determine Status based on Threshold
        is_approved = prob >= threshold

        st.markdown("---")
        if is_approved:
            st.markdown(f"""
                <div class="status-box approved">
                    <h2 style="margin:0;">✅ LOAN APPLICATION APPROVED</h2>
                    <p style="font-size: 20px;">System Confidence: {prob:.2%}</p>
                    <p>Applicant meets the liquidity and credit history requirements for this tier.</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="status-box rejected">
                    <h2 style="margin:0;">❌ LOAN APPLICATION REJECTED</h2>
                    <p style="font-size: 20px;">System Confidence: {(1-prob):.2%}</p>
                    <p>Reason: High credit risk or insufficient collateral coverage.</p>
                </div>
            """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
st.markdown("<br><p style='text-align: center; color: #94a3b8;'>© 2026 FinSafe AI Systems | Secure Financial Node</p>", unsafe_allow_html=True)