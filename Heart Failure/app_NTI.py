import numpy as np
import streamlit as st
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Heart Failure Prediction",
    page_icon="❤️"
)

st.header('Heart Failure Prediction ❤️')
st.write('This app predicts the risk of heart failure based on patient data.')

model = joblib.load('heart_failure_rf.pkl')
scaler = joblib.load('scaler.pkl')


def convert_to_numeric(value):
    return 1 if value == 'Yes' else 0

def convert_sex(value):
    return 1 if value == 'Male' else 0

col1, col2 = st.columns(2)

with col1:
    age = st.number_input('Age' , 20 , 100, 50)
    creatinine_phosphokinase = st.number_input('Creatinine Phosphokinase' , 0 , 10000, 100)
    ejection_fraction = st.number_input('Ejection Fraction' , 0 , 100, 50)
    platelets = st.number_input('Platelets' , 0 , 1000000, 200000)
    serum_sodium = st.number_input('Serum Sodium' , 100.0 , 200.0, 140.0)
    smoking = convert_to_numeric(st.selectbox('Smoking' , ['No' , 'Yes']))
    
with col2:
    anaemia = convert_to_numeric(st.selectbox('Anaemia' , ['No' , 'Yes']))
    diabetes = convert_to_numeric(st.selectbox('Diabetes' , ['No' , 'Yes']))
    high_blood_pressure = convert_to_numeric(st.selectbox('High Blood Pressure' , ['No' , 'Yes']))
    serum_creatinine = st.number_input('Serum Creatinine' , 0.0 , 10.0, 1.0)
    sex = convert_sex(st.selectbox('Sex' , ['Male' , 'Female']))
    time = st.number_input('Time', 0 , 365, 90)

input_data = np.array([
            age, anaemia, creatinine_phosphokinase, diabetes, 
            ejection_fraction, high_blood_pressure, platelets, 
            serum_creatinine, serum_sodium, sex, smoking, time
        ]).reshape(1, -1)


input_data = scaler.transform(input_data)

if st.button('Predict'):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0] 
        
    st.write(f"Prediction Confidence: {np.max(probability)*100:.2f}%")
        
    if prediction == 1:
        st.error("Patient has a high risk of a heart failure event.")
    else:
        st.success("Patient has a low risk of a heart failure event.")
        
    sns.barplot(x=['Low Risk', 'High Risk'], y=probability, palette='coolwarm')
    plt.title('Probability Heart Failure')
    plt.xlabel('Risk Level')
    plt.ylabel('Probability')
    st.pyplot(plt)
            