import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from ibm_watson_machine_learning.foundation_models import Model
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
WATSONX_API_KEY = os.getenv('WATSONX_API_KEY')
WATSONX_PROJECT_ID = os.getenv('WATSONX_PROJECT_ID')

# Initialize IBM Granite model
@st.cache_resource
def init_granite_model():
    credentials = {
        "url": "https://us-south.ml.cloud.ibm.com",
        "apikey": WATSONX_API_KEY
    }
    # Debugging: Print the loaded project_id to verify
    print(f"Loaded project_id: {WATSONX_PROJECT_ID}")
    if not WATSONX_PROJECT_ID:
        raise ValueError("WATSONX_PROJECT_ID is not set. Please check your .env file.")
    model_id = "ibm/granite-3-3-8b-instruct"  # Updated to recommended model
    # Pass project_id as a top-level argument, not in params
    return Model(model_id=model_id, project_id=WATSONX_PROJECT_ID, credentials=credentials)

# Patient Query Function
@st.cache_data(show_spinner=False)
def answer_patient_query(query):
    model = init_granite_model()
    query_prompt = f"""
As a healthcare AI assistant, provide a helpful, accurate, and evidence-based response to the following patient question:
PATIENT QUESTION: {query}
Provide a clear, empathetic response that:
- Directly addresses the question
- Includes relevant medical facts
- Acknowledges limitations (when appropriate)
- Suggests when to seek professional medical advice
- Avoids making definitive diagnoses
- Uses accessible, non-technical language
RESPONSE:
"""
    answer = model.generate_text(prompt=query_prompt)
    # Ensure the response is complete and coherent
    if len(answer.strip()) < 50:  # Arbitrary threshold for response length
        answer += "\n\nIt is recommended to consult a healthcare professional for further guidance."
    return answer

# Disease Prediction Function
@st.cache_data(show_spinner=False)
def predict_disease(symptoms, age, gender, medical_history, avg_heart_rate, avg_bp_systolic, avg_bp_diastolic, avg_glucose, recent_symptoms):
    model = init_granite_model()
    prediction_prompt = f"""
As a medical AI assistant, predict potential health conditions based on the following patient data:
Current Symptoms: {symptoms}
Age: {age}
Gender: {gender}
Medical History: {medical_history}
Recent Health Metrics:
- Average Heart Rate: {avg_heart_rate} bpm
- Average Blood Pressure: {avg_bp_systolic}/{avg_bp_diastolic} mmHg
- Average Blood Glucose: {avg_glucose} mg/dL
- Recently Reported Symptoms: {recent_symptoms}
Format your response as:
1. Potential condition name
2. Likelihood (High/Medium/Low)
3. Brief explanation
4. Recommended next steps
Provide the top 3 most likely conditions based on the data provided.
"""
    prediction = model.generate_text(prompt=prediction_prompt)
    return prediction

# Treatment Plan Function
@st.cache_data(show_spinner=False)
def generate_treatment_plan(condition, age, gender, medical_history):
    model = init_granite_model()
    treatment_prompt = f"""
As a medical AI assistant, generate a personalized treatment plan for the following scenario:
Patient Profile:
- Condition: {condition}
- Age: {age}
- Gender: {gender}
- Medical History: {medical_history}
Create a comprehensive, evidence-based treatment plan that includes:
1. Recommended medications (include dosage guidelines if appropriate)
2. Lifestyle modifications
3. Follow-up testing and monitoring
4. Dietary recommendations
5. Physical activity guidelines
6. Mental health considerations
Format this as a clear, structured treatment plan that follows current medical guidelines while being personalized to this patient's specific needs.
"""
    treatment_plan = model.generate_text(prompt=treatment_prompt)
    return treatment_plan

# Generate sample patient data for analytics
def generate_sample_data():
    np.random.seed(42)
    dates = pd.date_range(end=pd.Timestamp.today(), periods=90)
    heart_rate = np.random.normal(74, 5, 90)
    bp_systolic = np.random.normal(121, 8, 90)
    bp_diastolic = np.random.normal(80, 5, 90)
    glucose = np.random.normal(101, 15, 90)
    symptoms = np.random.choice(['None', 'Headache', 'Nausea', 'Chest Pain', 'Dizziness', 'Fatigue'], 90, p=[0.3,0.2,0.1,0.15,0.1,0.15])
    sleep = np.random.normal(7, 1, 90)
    df = pd.DataFrame({
        'date': dates,
        'heart_rate': heart_rate,
        'bp_systolic': bp_systolic,
        'bp_diastolic': bp_diastolic,
        'glucose': glucose,
        'symptom': symptoms,
        'sleep': sleep
    })
    return df

# Sidebar: Patient Profile
def patient_profile():
    st.sidebar.header('Patient Profile')
    name = st.sidebar.text_input('Name', value='')
    age = st.sidebar.number_input('Age', min_value=0, max_value=120, value=30)
    gender = st.sidebar.selectbox('Gender', ['Male', 'Female', 'Other'])
    medical_history = st.sidebar.text_area('Medical History', value='')
    current_medications = st.sidebar.text_area('Current Medications', value='')
    allergies = st.sidebar.text_input('Allergies', value='')
    return name, age, gender, medical_history, current_medications, allergies

# Patient Chat UI
def display_patient_chat():
    st.header('🩺 24/7 Patient Support')
    st.write('Ask any health-related question for immediate assistance.')
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []
    user_input = st.text_input('Ask your health question...')
    if st.button('Send') and user_input:
        st.session_state['chat_history'].append({'role': 'user', 'content': user_input})
        with st.spinner('AI is responding...'):
            ai_response = answer_patient_query(user_input)
        st.session_state['chat_history'].append({'role': 'ai', 'content': ai_response})
    for msg in st.session_state['chat_history']:
        if msg['role'] == 'user':
            st.info(msg['content'])
        else:
            st.success(msg['content'])

# Disease Prediction UI
def display_disease_prediction(age, gender, medical_history, df):
    st.header('🔬 Disease Prediction System')
    st.write('Enter symptoms and patient data to receive potential condition predictions.')
    symptoms = st.text_area('Current Symptoms', placeholder='Describe symptoms in detail (e.g., persistent headache for 3 days, fatigue, mild fever of 99.5°F)')
    avg_heart_rate = round(df['heart_rate'].mean(), 1)
    avg_bp_systolic = round(df['bp_systolic'].mean(), 1)
    avg_bp_diastolic = round(df['bp_diastolic'].mean(), 1)
    avg_glucose = round(df['glucose'].mean(), 1)
    recent_symptoms = ', '.join(df['symptom'].tail(7).unique())
    if st.button('Generate Prediction') and symptoms:
        with st.spinner('AI is analyzing your symptoms...'):
            prediction = predict_disease(symptoms, age, gender, medical_history, avg_heart_rate, avg_bp_systolic, avg_bp_diastolic, avg_glucose, recent_symptoms)
        st.subheader('Potential Conditions')
        st.write(prediction)

# Treatment Plan UI
def display_treatment_plans(age, gender, medical_history):
    st.header('💊 Personalized Treatment Plan Generator')
    st.write('Generate customized treatment recommendations based on specific conditions.')
    condition = st.text_input('Medical Condition', value='')
    if st.button('Generate Treatment Plan') and condition:
        with st.spinner('AI is generating your treatment plan...'):
            plan = generate_treatment_plan(condition, age, gender, medical_history)
        st.subheader('Personalized Treatment Plan')
        st.write(plan)

# Health Analytics Dashboard
def display_health_analytics(df, patient_results):
    st.header('📊 Health Analytics Dashboard')
    st.write('Visualize and analyze patient health data trends based on AI predictions.')

    # Heart Rate Trend
    fig_hr = go.Figure()
    fig_hr.add_trace(go.Scatter(x=df['date'], y=df['heart_rate'], mode='lines', name='Heart Rate'))
    fig_hr.update_layout(title='Heart Rate Trend (90-Day)', xaxis_title='Date', yaxis_title='Heart Rate (bpm)')

    # Blood Pressure Trend
    fig_bp = go.Figure()
    fig_bp.add_trace(go.Scatter(x=df['date'], y=df['bp_systolic'], mode='lines', name='Systolic'))
    fig_bp.add_trace(go.Scatter(x=df['date'], y=df['bp_diastolic'], mode='lines', name='Diastolic'))
    fig_bp.update_layout(title='Blood Pressure Trend (90-Day)', xaxis_title='Date', yaxis_title='Blood Pressure (mmHg)')

    # Glucose Trend
    fig_glucose = go.Figure()
    fig_glucose.add_trace(go.Scatter(x=df['date'], y=df['glucose'], mode='lines', name='Blood Glucose'))
    fig_glucose.add_trace(go.Scatter(x=df['date'], y=[125]*len(df), mode='lines', name='Reference', line=dict(dash='dash', color='red')))
    fig_glucose.update_layout(title='Blood Glucose Trend (90-Day)', xaxis_title='Date', yaxis_title='Blood Glucose (mg/dL)')

    # Symptom Frequency Pie
    symptom_counts = df['symptom'].value_counts()
    fig_symptom = go.Figure(data=[go.Pie(labels=symptom_counts.index, values=symptom_counts.values)])
    fig_symptom.update_layout(title='Symptom Frequency (90-Day)')

    # Metrics Summary
    st.subheader('Health Metrics Summary')
    col1, col2, col3, col4 = st.columns(4)
    col1.metric('Avg. Heart Rate', f"{df['heart_rate'].mean():.1f} bpm", f"{df['heart_rate'].iloc[-1] - df['heart_rate'].iloc[0]:+.1f}")
    col2.metric('Avg. Blood Pressure', f"{df['bp_systolic'].mean():.1f}/{df['bp_diastolic'].mean():.1f}", f"{df['bp_systolic'].iloc[-1] - df['bp_systolic'].iloc[0]:+.1f}")
    col3.metric('Avg. Blood Glucose', f"{df['glucose'].mean():.1f} mg/dL", f"{df['glucose'].iloc[-1] - df['glucose'].iloc[0]:+.1f}")
    col4.metric('Avg. Sleep', f"{df['sleep'].mean():.1f} hours", f"{df['sleep'].iloc[-1] - df['sleep'].iloc[0]:+.1f}")

    st.plotly_chart(fig_hr, use_container_width=True)
    st.plotly_chart(fig_bp, use_container_width=True)
    st.plotly_chart(fig_glucose, use_container_width=True)
    st.plotly_chart(fig_symptom, use_container_width=True)

    # Display AI Predictions
    st.subheader('AI-Generated Health Insights')
    for result in patient_results:
        st.write(f"- {result}")

    # Warning Caption
    st.caption('This is an AI prediction. You must consult a doctor for professional medical advice.')

# Main App
st.set_page_config(page_title='HealthAI - Intelligent Healthcare Assistant', page_icon='🩺', layout='wide')
st.markdown('<style>body {background-color: #f6fafd;}</style>', unsafe_allow_html=True)

name, age, gender, medical_history, current_medications, allergies = patient_profile()
df = generate_sample_data()

menu = st.sidebar.radio('Navigate', ['Patient Chat', 'Disease Prediction', 'Treatment Plans', 'Health Analytics'])

if menu == 'Patient Chat':
    display_patient_chat()
elif menu == 'Disease Prediction':
    display_disease_prediction(age, gender, medical_history, df)
elif menu == 'Treatment Plans':
    display_treatment_plans(age, gender, medical_history)
elif menu == 'Health Analytics':
    # Dummy patient results for analytics display
    dummy_patient_results = [
        "Heart rate, blood pressure, and blood glucose levels are all within normal range.",
        "No current symptoms reported.",
        "No current medications."
    ]
    display_health_analytics(df, dummy_patient_results)
