import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

try:
    pipeline = joblib.load('../models/final_model.pkl')
except FileNotFoundError:
    st.error("Model file 'final_model.pkl' not found. Please train and export the model first.")
   
    st.stop()

st.title('Heart Disease Prediction App')
st.write('This app predicts the likelihood of heart disease based on user inputs.')

st.sidebar.header('User Input Features')

def get_user_input():
    age = st.sidebar.slider('Age', 29, 77, 54)
    sex = st.sidebar.selectbox('Sex', [0, 1], format_func=lambda x: 'Female' if x == 0 else 'Male')
    cp = st.sidebar.selectbox('Chest Pain Type (1–4)', [1, 2, 3, 4])
    trestbps = st.sidebar.slider('Resting Blood Pressure', 94, 200, 130)
    chol = st.sidebar.slider('Serum Cholesterol (mg/dl)', 126, 564, 246)
    fbs = st.sidebar.selectbox('Fasting Blood Sugar > 120 mg/dl', [0, 1])
    restecg = st.sidebar.selectbox('Resting ECG Results (0–2)', [0, 1, 2])
    thalach = st.sidebar.slider('Max Heart Rate Achieved', 71, 202, 149)
    exang = st.sidebar.selectbox('Exercise Induced Angina', [0, 1])
    oldpeak = st.sidebar.slider('ST Depression (oldpeak)', 0.0, 6.2, 1.0, 0.1)
    slope = st.sidebar.selectbox('Slope of Peak Exercise ST', [1, 2])
    ca = st.sidebar.selectbox('Number of Major Vessels (0–3)', [0, 1, 2, 3])
    thal = st.sidebar.selectbox('Thalassemia (1-3)', [1, 2, 3])

    user_data = {
        'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol,
        'fbs': fbs, 'restecg': restecg, 'thalach': thalach, 'exang': exang,
        'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
    }
    return pd.DataFrame(user_data, index=[0])

user_input_df = get_user_input()

st.subheader('Your Input:')
st.write(user_input_df)

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
        st.error(f"An error occurred during prediction. Error: {e}")

st.subheader('Data Visualization')

try:
    df_plot = pd.read_csv('../data/heart_disease.csv')
    df_plot['target'] = df_plot['target'].apply(lambda x: 1 if x > 0 else 0)

    # Age distribution
    st.write("### Age Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df_plot['age'], kde=True, ax=ax, color="skyblue")
    st.pyplot(fig)

    # Cholesterol distribution
    st.write("### Cholesterol Distribution")
    fig2, ax2 = plt.subplots()
    sns.histplot(df_plot['chol'], kde=True, ax=ax2, color="orange")
    st.pyplot(fig2)

    # Blood Pressure distribution
    st.write("### Resting Blood Pressure Distribution")
    fig3, ax3 = plt.subplots()
    sns.histplot(df_plot['trestbps'], kde=True, ax=ax3, color="green")
    st.pyplot(fig3)

    # Chest pain type vs target
    st.write("### Chest Pain Type vs Heart Disease")
    fig4, ax4 = plt.subplots()
    sns.countplot(data=df_plot, x='cp', hue='target', ax=ax4)
    ax4.set_xticklabels(['Type 1', 'Type 2', 'Type 3', 'Type 4'])
    st.pyplot(fig4)

    # Correlation heatmap
    st.write("### Correlation Heatmap")
    fig5, ax5 = plt.subplots(figsize=(10,6))
    sns.heatmap(df_plot.corr(), annot=True, cmap="coolwarm", ax=ax5)
    st.pyplot(fig5)

except FileNotFoundError:
    st.warning("Dataset not found. Please ensure 'data/heart_disease.csv' exists.")
