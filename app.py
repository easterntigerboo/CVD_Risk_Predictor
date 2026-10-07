import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

MODEL_FILE = Path(__file__).parent / "cvd_random_forest.joblib"   # same folder as app.py
TRAIN_AGE_MIN, TRAIN_AGE_MAX = 29, 64   # age range of the training data

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
        return joblib.load(MODEL_FILE)
    except FileNotFoundError:
        st.error(f"Error: '{MODEL_FILE.name}' not found next to app.py. Please check file placement.")
    except Exception as e:   # e.g. model saved with a different scikit-learn version
        st.error(f"The model file could not be loaded ({type(e).__name__}). "
                 "Make sure scikit-learn in this environment is the same version used to train it.")
    return None


model = load_prediction_model()

if model is not None:
    # --- Input Form Fields Layout ---
    st.header("Patient Demographics & Vitals")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age (Years)", min_value=18, max_value=100, value=50, step=1)
        gender = st.selectbox("Gender", options=[1, 2], format_func=lambda x: "Female" if x == 1 else "Male")
        height = st.number_input("Height (cm)", min_value=100, max_value=230, value=174, step=1)
        weight = st.number_input("Weight (kg)", min_value=30, max_value=250, value=83, step=1)

    with col2:
        sbp = st.number_input("Systolic Blood Pressure (mmHg)", min_value=60, max_value=240, value=125, step=1)
        dbp = st.number_input("Diastolic Blood Pressure (mmHg)", min_value=40, max_value=120, value=90, step=1)
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

    st.write("---")

    # Execution Button Trigger
    if st.button("Calculate CVD Risk Assessment", type="primary"):
        # --- Check the inputs before predicting ---
        if sbp <= dbp:
            st.error("Systolic (upper) pressure must be higher than diastolic (lower). Please check the blood pressure values.")
            st.stop()
        if not 12 <= bmi <= 60:
            st.error(f"BMI works out to {bmi:.1f}, outside the 12-60 range the model was trained on. Please check height and weight.")
            st.stop()
        if not TRAIN_AGE_MIN <= age <= TRAIN_AGE_MAX:
            st.warning(f"The model was trained on ages {TRAIN_AGE_MIN}-{TRAIN_AGE_MAX}. "
                       f"A result for age {age} is an extrapolation and less reliable.")

        # Every value we can supply, under every name the model might have been trained with
        available = {
            'age_year': float(age),
            'gender': int(gender),
            'height': float(height),
            'weight': float(weight),
            'bmi': float(bmi),
            'bmi_cleaned': float(bmi),
            'bp_hi_cleaned': float(sbp),
            'bp_lo_cleaned': float(dbp),
            'cholesterol': int(chol),
            'glucose': int(gluc),
            'smoke': int(smoke),
            'alcohol': int(alcohol),
            'active': int(active),
        }

        # Use exactly the columns the model was trained on, in the same order
        needed = list(model.feature_names_in_)
        missing = [c for c in needed if c not in available]
        if missing:
            st.error(f"The model expects inputs this form does not collect: {missing}")
            st.stop()
        patient_data = pd.DataFrame([{c: available[c] for c in needed}])[needed]

        # Predict Class Probabilities
        probability = model.predict_proba(patient_data)[0][1]
        percent_probability = probability * 100

        # Display Results Dashboard Metrics
        st.subheader("Analysis Results:")
        st.metric(label="Estimated CVD Risk Probability", value=f"{percent_probability:.1f}%")

        if percent_probability >= 50.0:
            st.error("⚠️ Model Prediction Status: **HIGH RISK**")
            st.write("The model estimates this profile as more likely than not to be in the CVD group.")
        else:
            st.success("✅ Model Prediction Status: **LOWER RISK**")
            st.write("The model estimates this profile as less likely than not to be in the CVD group.")

        st.info(f"Calculated Patient Baseline BMI: **{bmi:.1f}**")
        st.caption("Statistical estimate from a dataset (about 73% accuracy). Not a medical diagnosis.")