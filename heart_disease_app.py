import streamlit as st
import pandas as pd
import joblib
import os

# Set page configuration
st.set_page_config(
    page_title="Heart Disease Risk Predictor",
    page_icon="💓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("💓 Heart Disease Risk Predictor")
st.markdown("""
This clinical decision-support application uses a fine-tuned **XGBoost Classifier** pipeline 
to estimate a patient's risk of cardiovascular disease based on demographic profiles, lifestyle habits, and vital metrics.
""")

# Load the trained model pipeline
model_path = "best_heart_disease_model.pkl"

@st.cache_resource
def load_model(path):
    if os.path.exists(path):
        return joblib.load(path)
    return None

model = load_model(model_path)

if model is None:
    st.error(f"⚠️ Model file `{model_path}` not found in the directory! Please run the notebook first to generate the pickle file.")
else:
    # App layout - two columns
    col1, col2 = st.columns([2, 1.5], gap="large")

    with col1:
        st.subheader("📋 Patient Clinical & Lifestyle Details")
        st.markdown("Enter patient metrics below to evaluate cardiovascular risk.")
        
        # Grid layout for inputs
        grid1, grid2 = st.columns(2)
        
        with grid1:
            age = st.slider("Age (Years)", min_value=18, max_value=100, value=50, step=1)
            gender = st.selectbox("Gender", options=["Male", "Female"])
            glucose = st.slider("Glucose Level (mg/dL)", min_value=50, max_value=250, value=100, step=1)
            cholesterol = st.slider("Cholesterol Level (mg/dL)", min_value=80, max_value=400, value=200, step=1)
            heart_rate = st.slider("Heart Rate (bpm)", min_value=40, max_value=150, value=75, step=1)
            bmi = st.slider("Body Mass Index (BMI)", min_value=10.0, max_value=50.0, value=25.0, step=0.1)

        with grid2:
            systolic_bp = st.slider("Systolic Blood Pressure (mmHg)", min_value=80, max_value=220, value=120, step=1)
            diastolic_bp = st.slider("Diastolic Blood Pressure (mmHg)", min_value=50, max_value=130, value=80, step=1)
            
            st.markdown("---")
            smoking = st.radio("Do they smoke?", options=["Yes", "No"], horizontal=True)
            alcohol = st.radio("Do they consume alcohol regularly?", options=["Yes", "No"], horizontal=True)
            activity = st.selectbox("Physical Activity Level", options=["Low", "Medium", "High"])
            family_history = st.radio("Family History of Heart Disease?", options=["Yes", "No"], horizontal=True)

    with col2:
        st.subheader("🎯 Risk Assessment Output")
        
        # Package inputs into a DataFrame matching training schema
        input_data = pd.DataFrame([{
            'age': age,
            'gender': gender,
            'glucose_mg_dl': glucose,
            'cholesterol_mg_dl': cholesterol,
            'systolic_bp': systolic_bp,
            'diastolic_bp': diastolic_bp,
            'bmi': bmi,
            'heart_rate': heart_rate,
            'smoking': smoking,
            'alcohol_consumption': alcohol,
            'physical_activity': activity,
            'family_history': family_history
        }])
        
        # Predict
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]
        
        # Visual display cards
        st.markdown("### Estimated Risk Probability")
        
        # Display large percentage index
        if prediction == 1:
            st.metric(label="Risk Probability Score", value=f"{probability:.2%}", delta="Elevated Risk", delta_color="inverse")
            st.markdown(
                f'<div style="background-color:#FFD2D2;color:#D8000C;padding:20px;border-radius:10px;border-left:8px solid #D8000C;font-size:18px;font-weight:bold;">'
                f'⚠️ High Risk: The model predicts that this patient is at elevated risk of heart disease.'
                f'</div>', 
                unsafe_allow_html=True
            )
        else:
            st.metric(label="Risk Probability Score", value=f"{probability:.2%}", delta="Low/Normal Risk", delta_color="normal")
            st.markdown(
                f'<div style="background-color:#DFF2BF;color:#4F8A10;padding:20px;border-radius:10px;border-left:8px solid #4F8A10;font-size:18px;font-weight:bold;">'
                f'✅ Low Risk: The model predicts that this patient is at normal risk levels.'
                f'</div>',
                unsafe_allow_html=True
            )
            
        # Clinical Context Section
        st.markdown("---")
        st.markdown("### 💡 Interactive Recommendations")
        
        # Simple rule-based alerts based on input details
        recs = []
        if systolic_bp >= 140 or diastolic_bp >= 90:
            recs.append("🔴 **Hypertension Detected:** Patient BP is elevated. Monitor blood pressure closely and consider clinical intervention.")
        if cholesterol >= 240:
            recs.append("🔴 **Hypercholesterolemia Detected:** Patient cholesterol is high (>= 240 mg/dL). Consider dietary changes or lipid-lowering therapy.")
        if smoking == "Yes":
            recs.append("🚬 **Smoking Cessation:** Counseling on smoking cessation will substantially reduce future risk.")
        if activity == "Low":
            recs.append("🏃‍♂️ **Physical Inactivity:** Recommend regular moderate physical exercise (e.g., 150 minutes/week of brisk walking).")
            
        if recs:
            for rec in recs:
                st.markdown(rec)
        else:
            st.markdown("🟢 All vital markers and behavioral features appear to be within normal healthy ranges!")
            
    # Bottom section with feature info
    st.markdown("---")
    st.caption("Disclaimer: This tool is for educational and model presentation purposes only. It should not be used as a replacement for professional medical advice, diagnosis, or treatment.")
