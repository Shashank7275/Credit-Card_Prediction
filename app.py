
import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained model
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Load the dataset to get column names and some stats if needed
df = pd.read_csv('credit card.csv')
# Drop the target variable
X = df.drop('Class', axis=1)


st.title('Credit Card Fraud Prediction')

st.header('Input Features')

# Create input fields for all the features
time = st.number_input('Time', value=0.0)
amount = st.number_input('Amount', value=0.0)

# Create sliders for the 'V' features
v_features = []
for i in range(1, 29):
    v_features.append(st.slider(f'V{i}', float(X[f'V{i}'].min()), float(X[f'V{i}'].max()), float(X[f'V{i}'].mean())))

# Predict button
if st.button('Predict'):
    # Create a dataframe from the inputs
    input_data = pd.DataFrame({
        'Time': [time],
        'V1': [v_features[0]],
        'V2': [v_features[1]],
        'V3': [v_features[2]],
        'V4': [v_features[3]],
        'V5': [v_features[4]],
        'V6': [v_features[5]],
        'V7': [v_features[6]],
        'V8': [v_features[7]],
        'V9': [v_features[8]],
        'V10': [v_features[9]],
        'V11': [v_features[10]],
        'V12': [v_features[11]],
        'V13': [v_features[12]],
        'V14': [v_features[13]],
        'V15': [v_features[14]],
        'V16': [v_features[15]],
        'V17': [v_features[16]],
        'V18': [v_features[17]],
        'V19': [v_features[18]],
        'V20': [v_features[19]],
        'V21': [v_features[20]],
        'V22': [v_features[21]],
        'V23': [v_features[22]],
        'V24': [v_features[23]],
        'V25': [v_features[24]],
        'V26': [v_features[25]],
        'V27': [v_features[26]],
        'V28': [v_features[27]],
        'Amount': [amount]
    })

    # Make prediction
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)

    st.subheader('Prediction')
    if prediction[0] == 1:
        st.write('**Fraudulent Transaction**')
    else:
        st.write('**Normal Transaction**')

    st.subheader('Prediction Probability')
    st.write(f'Probability of being a normal transaction: {prediction_proba[0][0]:.2f}')
    st.write(f'Probability of being a fraudulent transaction: {prediction_proba[0][1]:.2f}')
