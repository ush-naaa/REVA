import os
import streamlit as st
from dotenv import load_dotenv
from agents.Predictor_agent import PredictorAgent
from agents.Insights_agent import InsightsAgent
from utils.constants import (
    CITY_OPTIONS,
    PURPOSE_OPTIONS,
    PROPERTY_TYPE_OPTIONS,
    CITY_LOCATION_MAP
)

# ---------------------------
# 1. Load Environment & Config
# ---------------------------
load_dotenv()
GROQ_KEY = os.getenv("GROQ_API_KEY")
# Add this line to define the new supported model
MODEL_NAME = "llama-3.3-70b-versatile"

st.set_page_config(
    page_title="Reva AI",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------
# 2. Optimized Agent Initialization
# ---------------------------
@st.cache_resource
def initialize_reva_engine():
    if not GROQ_KEY:
        st.error("⚠️ GROQ_API_KEY not found! Please check your .env file.")
        st.stop()
    
    predictor = PredictorAgent()
    insights = InsightsAgent(groq_api_key=GROQ_KEY, model_name=MODEL_NAME)
    return predictor, insights

predictor, insights_agent = initialize_reva_engine()

# ---------------------------
# 3. Professional Styling (CSS)
# ---------------------------
primary_color = "#1E293B"
secondary_color = "#64748B"
border_color = "#F1F5F9"

st.markdown(f"""
    <style>
    .stApp {{ background-color: #FFFFFF; color: #0F172A; }}
    h1, h2, h3 {{ color: {primary_color} !important; font-weight: 700 !important; }}
    div[data-testid="column"] {{
        background-color: #FFFFFF; padding: 1.5rem; border-radius: 8px;
        border: 1px solid {border_color}; box-shadow: 0 1px 3px rgba(0,0,0,0.02);
    }}
    .stButton>button {{
        background-color: {primary_color}; color: white !important;
        border-radius: 6px; width: 100%; font-weight: 600;
    }}
    </style>
    """, unsafe_allow_html=True)

# ---------------------------
# 4. Header Section
# ---------------------------
col_header, col_status = st.columns([0.8, 0.2])
with col_header:
    st.markdown("""
        <h1 style='margin-bottom: 0px;'>
            <span style='color: #1E293B; font-weight: 800;'>REVA</span>
            <span style='color: #64748B; font-weight: 300;'>AI</span>
        </h1>
        <p style='color: #64748B; font-size: 1.1rem; margin-top: -5px;'>
            Institutional Grade Real Estate Valuation Engine
        </p>
    """, unsafe_allow_html=True)

with col_status:
    st.markdown("<div style='text-align: right; padding-top: 25px;'><span style='color: #1E293B; font-size: 10px; font-weight: 700; border: 1.5px solid #1E293B; padding: 4px 10px; border-radius: 4px;'>STABLE | V1.0</span></div>", unsafe_allow_html=True)

st.markdown("<hr style='margin-top: 0px; margin-bottom: 30px; border: 0; border-top: 1px solid #F1F5F9;'>", unsafe_allow_html=True)

# ---------------------------
# 5. Input Section (Defining variables)
# ---------------------------
st.header("Property Details")
col1, col2 = st.columns([1, 1])

with col1:
    city = st.selectbox("City", CITY_OPTIONS)
    locations = CITY_LOCATION_MAP.get(city, [])
    location = st.selectbox("Location", locations)
    property_type = st.selectbox("Property Type", PROPERTY_TYPE_OPTIONS)
    purpose = st.selectbox("Purpose", PURPOSE_OPTIONS)

with col2:
    bedrooms = st.slider("Bedrooms", 1, 10, 3)
    baths = st.slider("Bathrooms", 1, 10, 2)
    area = st.number_input("Area (in Marla)", min_value=1.0, max_value=1000.0, value=5.0, step=0.5)

# --- CRITICAL STEP: Define the dictionary BEFORE the button click ---
user_input = {
    "city": city,
    "location": location,
    "property_type": property_type,
    "purpose": purpose,
    "bedrooms": bedrooms,
    "baths": baths,
    "Area_in_Marla": area
}

# ---------------------------
# 6. Prediction & Insights Execution
# ---------------------------
if st.button("Estimate Property"):
    with st.spinner("Reva is analyzing market trends..."):
        # Predict using Agent 1
        predicted_class, confidence = predictor.predict(user_input)

        st.markdown("---")
        st.subheader("Market Assessment")

        m_col1, m_col2 = st.columns(2)
        m_col1.metric("Valuation Tier", predicted_class)
        m_col2.metric("Analysis Confidence", f"{confidence}%")

        # Explain using Agent 2 (RAG + LLM)
        with st.expander("Strategic Market Insights", expanded=True):
            report = insights_agent.explain_prediction(
                user_input=user_input,
                predicted_class=predicted_class,
                confidence=confidence
            )
            st.markdown(report)

st.divider()
st.caption("© 2025 Reva AI • Professional Real Estate Analytics")