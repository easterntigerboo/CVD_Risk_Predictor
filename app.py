import streamlit as st
import pandas as pd
import joblib

# Set web page configurations
st.set_page_config(
    page_title="Cardiovascular Disease (CVD) Risk Predictor",
    page_icon="❤️",
    layout="centered"
)

# Application Heading Layout
st.title("❤️ Cardiovascular Disease Risk Predictor")
st.markdown("Enter patient clinical attributes below to estimate CVD development probability.")
st.write("---")

# Safe function to load the saved joblib classifier
@st.cache_resource
def load_prediction_model():
    try:
        return joblib.load("cvd_random_forest.joblib")
    except FileNotFoundError:
        st.error("Error: 'cvd_random_forest.joblib' file not found in the directory. Please check file placement.")
        return None

model = load_prediction_model()

if model is not None:
    # --- Input Form Fields Layout ---
    st.header("Patient Demographics & Vitals")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age (Years)", min_value=1, max_value=120, value=50, step=1)
        gender = st.selectbox("Gender", options=[1, 2], format_func=lambda x: "Female" if x == 1 else "Male")
        height = st.number_input("Height (cm)", min_value=50, max_value=250, value=174, step=1)
        weight = st.number_input("Weight (kg)", min_value=10, max_value=300, value=83, step=1)
        
    with col2:
        sbp = st.number_input("Systolic Blood Pressure (mmHg)", min_value=50, max_value=250, value=125, step=1)
        dbp = st.number_input("Diastolic Blood Pressure (mmHg)", min_value=30, max_value=150, value=90, step=1)
        chol = st.selectbox("Cholesterol Level", options=[1, 2, 3], format_func=lambda x: {1: "Normal", 2: "Above Normal", 3: "Well Above Normal"}[x])
        gluc = st.selectbox("Glucose Level", options=[1, 2, 3], format_func=lambda x: {1: "Normal", 2: "Above Normal", 3: "Well Above Normal"}[x])

    st.write("---")
    st.header("Lifestyle Habits")
    col3, col4, col5 = st.columns(3)
    
    with col3:
        smoke = st.radio("Smoking History?", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    with col4:
        alcohol = st.radio("Alcohol Intake?", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    with col5:
        active = st.radio("Physically Active?", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

    # Automated Feature Engineering (BMI Calculation)
    bmi = weight / ((height / 100) ** 2)
    
    # Constructing Structured Input Dataframe (Ensure columns precisely match original model layout)
    patient_data = pd.DataFrame([{
        'age_year': float(age),
        'gender': int(gender),
        'height': float(height),
        'weight': float(weight),
        'bp_hi_cleaned': float(sbp),
        'bp_lo_cleaned': float(dbp),
        'cholesterol': int(chol),
        'glucose': int(gluc),
        'smoke': int(smoke),
        'alcohol': int(alcohol),
        'active': int(active),
        'bmi': float(bmi)
    }])

    st.write("---")
    
    # Execution Button Trigger
    if st.button("Calculate CVD Risk Assessment", type="primary"):
        # Predict Class Probabilities
        probability = model.predict_proba(patient_data)[0][1]
        percent_probability = probability * 100
        
        # Display Results Dashboard Metrics
        st.subheader("Analysis Results:")
        st.metric(label="Estimated CVD Risk Probability", value=f"{percent_probability:.1f}%")
        
        if percent_probability >= 50.0:
            st.error("⚠️ Model Prediction Status: **HIGH RISK**")
            st.write("This patient presents significant clinical risk matching known cardiovascular complications.")
        else:
            st.success("✅ Model Prediction Status: **LOW RISK**")
            st.write("This patient falls below the baseline clinical classification threshold for CVD grouping.")
            
        st.info(f"Calculated Patient Baseline BMI: **{bmi:.1f}**")
