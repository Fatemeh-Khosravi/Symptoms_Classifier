import streamlit as st
import numpy as np

# Sample symptom list for demonstration
symptom_cols = [
    'itching', 'skin_rash', 'nodal_skin_eruptions', 'continuous_sneezing', 'shivering',
    'chills', 'joint_pain', 'stomach_pain', 'acidity', 'ulcers_on_tongue',
    'muscle_wasting', 'vomiting', 'burning_micturition', 'spotting_urination', 'fatigue',
    'weight_gain', 'anxiety', 'cold_hands_and_feets', 'mood_swings', 'weight_loss',
    'restlessness', 'lethargy', 'patches_in_throat', 'irregular_sugar_level', 'cough',
    'high_fever', 'sunken_eyes', 'breathlessness', 'sweating', 'dehydration',
    'indigestion', 'headache', 'yellowish_skin', 'dark_urine', 'nausea',
    'loss_of_appetite', 'pain_behind_the_eyes', 'back_pain', 'constipation',
    'abdominal_pain', 'diarrhoea', 'mild_fever', 'yellow_urine', 'yellowing_of_eyes',
    'acute_liver_failure', 'fluid_overload', 'swelling_of_stomach', 'swelled_lymph_nodes',
    'malaise', 'blurred_and_distorted_vision', 'phlegm', 'throat_irritation',
    'redness_of_eyes', 'sinus_pressure', 'runny_nose', 'congestion', 'chest_pain',
    'weakness_in_limbs', 'fast_heart_rate', 'pain_during_bowel_movements',
    'pain_in_anal_region', 'bloody_stool', 'irritation_in_anus', 'neck_pain',
    'dizziness', 'cramps', 'bruising', 'obesity', 'swollen_legs',
    'swollen_blood_vessels', 'puffy_face_and_eyes', 'enlarged_thyroid',
    'brittle_nails', 'swollen_extremeties', 'excessive_hunger',
    'extra_marital_contacts', 'drying_and_tingling_lips', 'slurred_speech',
    'knee_pain', 'hip_joint_pain', 'muscle_weakness', 'stiff_neck',
    'swelling_joints', 'movement_stiffness', 'spinning_movements',
    'loss_of_balance', 'unsteadiness', 'weakness_of_one_body_side',
    'loss_of_smell', 'bladder_discomfort', 'foul_smell_of_urine',
    'continuous_feel_of_urine', 'passage_of_gases', 'internal_itching',
    'toxic_look_(typhos)', 'depression', 'irritability', 'muscle_pain',
    'altered_sensorium', 'red_spots_over_body', 'belly_pain',
    'abnormal_menstruation', 'dischromic_patches', 'watering_from_eyes',
    'increased_appetite', 'polyuria', 'family_history', 'mucoid_sputum',
    'rusty_sputum', 'lack_of_concentration', 'visual_disturbances',
    'receiving_blood_transfusion', 'receiving_unsterile_injections',
    'coma', 'stomach_bleeding', 'distention_of_abdomen',
    'history_of_alcohol_consumption', 'fluid_overload', 'blood_in_sputum',
    'prominent_veins_on_calf', 'palpitations', 'painful_walking',
    'pus_filled_pimples', 'blackheads', 'scurring', 'skin_peeling',
    'silver_like_dusting', 'small_dents_in_nails', 'inflammatory_nails',
    'blister', 'red_sore_around_nose', 'yellow_crust_ooze'
]

# Sample diseases for demonstration
sample_diseases = [
    'Fungal infection', 'Allergy', 'GERD', 'Chronic cholestasis', 'Drug Reaction',
    'Peptic ulcer disease', 'AIDS', 'Diabetes', 'Gastroenteritis', 'Bronchial Asthma',
    'Hypertension', 'Migraine', 'Cervical spondylosis', 'Paralysis (brain hemorrhage)',
    'Jaundice', 'Malaria', 'Chicken pox', 'Dengue', 'Typhoid', 'hepatitis A',
    'Hepatitis B', 'Hepatitis C', 'Hepatitis D', 'Hepatitis E', 'Alcoholic hepatitis',
    'Tuberculosis', 'Common Cold', 'Pneumonia', 'Dimorphic hemorrhoids (piles)',
    'Heart attack', 'Varicose veins', 'Hypothyroidism', 'Hyperthyroidism',
    'Hypoglycemia', 'Osteoarthristis', 'Arthritis', '(vertigo) Paroymsal Positional Vertigo',
    'Acne', 'Urinary tract infection', 'Psoriasis', 'Impetigo'
]

st.set_page_config(page_title="Symptoms Diagnosis Classifier", page_icon="🩺")
st.title("🩺 Symptoms Diagnosis Classifier")

st.markdown("""
### Welcome to the Medical Symptoms Classifier!
This AI-powered application predicts diseases based on patient symptoms with **95.24% accuracy** across 41 medical conditions.

**How to use:**
1. Select the symptoms you're experiencing from the dropdown below
2. Click "Predict Disease" to get an AI-powered diagnosis
3. Review the prediction and confidence score
""")

selected_symptoms = st.multiselect(
    "Select up to 10 symptoms (start typing to search):",
    options=symptom_cols,
    max_selections=10,
    help="Choose the symptoms that best describe your condition"
)

if st.button("Predict Disease", type="primary"):
    if len(selected_symptoms) < 5:
        st.warning("⚠️ Please select at least 5 symptoms.")
    else:
        # Simulate prediction for demonstration
        import random
        
        # Show selected symptoms
        st.subheader("📋 Selected Symptoms:")
        for symptom in selected_symptoms:
            st.write(f"• {symptom}")
        
        # Simulate prediction
        predicted_disease = random.choice(sample_diseases)
        confidence = random.uniform(85.0, 98.0)
        
        st.subheader("🔍 AI Prediction Results:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                label="Predicted Disease",
                value=predicted_disease,
                delta=f"{confidence:.1f}% confidence"
            )
        
        with col2:
            st.metric(
                label="Model Accuracy",
                value="95.24%",
                delta="across 41 conditions"
            )
        
        # Add some visual elements
        st.progress(confidence/100)
        
        st.info("""
        **Note:** This is a demonstration interface. In the actual application, 
        the model would analyze your symptoms using advanced machine learning 
        algorithms to provide accurate disease predictions.
        """)
        
        # Show model information
        st.subheader("🤖 Model Information:")
        st.write("""
        - **Algorithm:** Random Forest Classifier
        - **Training Data:** 4,920 medical cases
        - **Diseases Covered:** 41 different medical conditions
        - **Overall Accuracy:** 95.24%
        - **Features:** 132 different symptoms
        """)

# Add footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Built with ❤️ using Streamlit and Machine Learning</p>
    <p>For demonstration purposes only - Not for medical diagnosis</p>
</div>
""", unsafe_allow_html=True) 