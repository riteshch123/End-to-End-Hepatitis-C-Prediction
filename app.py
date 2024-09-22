import streamlit as st
import requests
import os

FLASK_BACKEND_URL = os.getenv('flask_backend_url')

st.title('Hepatitis C Prediction')

# Collect user input (exclude 'Category')
Age = st.number_input('Age', min_value=0, max_value=120, value=32)
Sex = st.selectbox('Sex', ['m', 'f'])
ALB = st.number_input('Albumin (ALB)', min_value=0.0, value=38.5)
ALP = st.number_input('Alkaline phosphatase (ALP)', min_value=0.0, value=52.5)
ALT = st.number_input('Alanine transaminase (ALT)', min_value=0.0, value=7.7)
AST = st.number_input('Aspartate transaminase (AST)',
                      min_value=0.0, value=22.1)
BIL = st.number_input('Bilirubin (BIL)', min_value=0.0, value=7.5)
CHE = st.number_input('Cholinesterase (CHE)', min_value=0.0, value=6.93)
CHOL = st.number_input('Cholesterol (CHOL)', min_value=0.0, value=3.23)
CREA = st.number_input('Creatinine (CREA)', min_value=0.0, value=106.0)
GGT = st.number_input('Gamma-glutamyl transferase (GGT)',
                      min_value=0.0, value=12.1)
PROT = st.number_input('Total Protein (PROT)', min_value=0.0, value=69.0)

if st.button('Predict'):
    # Prepare data for the API
    data = {
        'Age': Age,
        'Sex': Sex,
        'ALB': ALB,
        'ALP': ALP,
        'ALT': ALT,
        'AST': AST,
        'BIL': BIL,
        'CHE': CHE,
        'CHOL': CHOL,
        'CREA': CREA,
        'GGT': GGT,
        'PROT': PROT
    }

    # Send data to the Flask API
    try:
        response = requests.post(f"{FLASK_BACKEND_URL}/predict", json=data)
        response.raise_for_status()  # Raises HTTPError for bad responses (4xx or 5xx)
        prediction = response.json()

        # Display the prediction
        if prediction['prediction'] == 1:
            st.success('The model predicts that the patient has Hepatitis.')
        else:
            st.success(
                'The model predicts that the patient does not have Hepatitis.')
    except requests.exceptions.HTTPError as http_err:
        st.error(f'HTTP error occurred: {http_err}')
        st.write(response.text)
    except Exception as err:
        st.error(f'An error occurred: {err}')
