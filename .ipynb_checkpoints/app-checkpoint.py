import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the trained model pipeline
try:
    pipeline = joblib.load('../models/final_model.pkl')
except FileNotFoundError:
    st.error("Model file 'heart_disease_model.pkl' not found. Please run the model export script first.")

# Title of the Streamlit app
st.title('Heart Disease Prediction App ❤️')
st.write('This app predicts the likelihood of heart disease based on user inputs.')

# Sidebar for user input
st.sidebar.header('User Input Features')

# Function to get user input
def get_user_input():
    age = st.sidebar.slider('Age', 29, 77, 54)
    sex = st.sidebar.selectbox('Sex', [0, 1], format_func=lambda x: 'Female' if x == 0 else 'Male')
    cp = st.sidebar.selectbox('Chest Pain Type', [1, 2, 3, 4])
    trestbps = st.sidebar.slider('Resting Blood Pressure', 94, 200, 130)
    chol = st.sidebar.slider('Serum Cholestoral in mg/dl', 126, 564, 246)
    fbs = st.sidebar.selectbox('Fasting Blood Sugar > 120 mg/dl', [0, 1])
    restecg = st.sidebar.selectbox('Resting ECG Results', [0, 1, 2])
    thalach = st.sidebar.slider('Maximum Heart Rate Achieved', 71, 202, 149)
    exang = st.sidebar.selectbox('Exercise Induced Angina', [0, 1])
    oldpeak = st.sidebar.slider('ST depression induced by exercise relative to rest', 0.0, 6.2, 1.0, 0.1)

    # Create a dictionary of the user's input
    user_data = {
        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalach': thalach,
        'exang': exang,
        'oldpeak': oldpeak
    }
    
    # Create a DataFrame from the dictionary
    features = pd.DataFrame(user_data, index=[0])
    return features

# Get user input
user_input_df = get_user_input()

# Display the user input on the main page
st.subheader('Your Input:')
st.write(user_input_df)

# Make prediction
if st.sidebar.button('Predict'):
    try:
        # Preprocess the input using the pipeline
        prediction = pipeline.predict(user_input_df)[0]
        prediction_proba = pipeline.predict_proba(user_input_df)[0, 1]

        # Display the prediction result
        st.subheader('Prediction:')
        if prediction == 1:
            st.error(f'Likelihood of Heart Disease: High. ({prediction_proba:.2%})')
        else:
            st.success(f'Likelihood of Heart Disease: Low. ({1 - prediction_proba:.2%})')
    except Exception as e:
        st.error(f"An error occurred during prediction. Check your model pipeline and input data. Error: {e}")

# Add data visualization for users
st.subheader('Data Visualization')

# Load the original data for plotting (this is simplified for the app)
try:
    # A simplified version of loading the data for the app. In a real scenario, you'd have a data source
    df_plot = pd.read_csv('../data/Heart_Disease.csv')
    df_plot['target'] = df_plot['target'].apply(lambda x: 1 if x > 0 else 0)
    
    # Histogram of Age Distribution
    st.write("### Age Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df_plot['age'], kde=True, ax=ax)
    st.pyplot(fig)

    # Countplot of Chest Pain Type
    st.write("### Chest Pain Type vs. Heart Disease")
    fig2, ax2 = plt.subplots()
    sns.countplot(data=df_plot, x='cp', hue='target', ax=ax2)
    plt.xticks([0, 1, 2, 3], ['Type 1', 'Type 2', 'Type 3', 'Type 4'])
    st.pyplot(fig2)

except FileNotFoundError:
    st.warning("Data file for visualizations not found. Displaying placeholders.")
    st.text("Please ensure 'Heart_Disease.csv' is in the '../data/Heart_Disease' directory to enable visualizations.")