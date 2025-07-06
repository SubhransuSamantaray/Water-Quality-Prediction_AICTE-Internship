# Import all the necessary libraries
import pandas as pd
import numpy as np
import joblib
import pickle
import streamlit as st

# Load the Model and Structure
model = joblib.load("xgb_pollution_model.pkl")
model_cols = joblib.load("xgb_model_columns.pkl")

# let create an User Interface
st.title("Water Pollutants Predictor")
st.write("Predict the water pollutants based on Year and Station ID")

#User inputs
year_input = st.number_input("Enter Year: ")
station_id = st.text_input("Enter Station ID: ")
# Predict button
if st.button("Predict"):
    if not station_id.strip():
        st.warning('Please enter the Station ID properly.')
    else:
        # Prepare the input
        input_df = pd.DataFrame({'year': [year_input], 'id': [station_id]})
        input_encoded = pd.get_dummies(input_df, columns=['id'])

        # Align columns with model's expected input
        for col in model_cols:
            if col not in input_encoded.columns:
                input_encoded[col] = 0
        input_encoded = input_encoded[model_cols]

        # Make the prediction
        prediction = model.predict(input_encoded)

        # Show results
        st.subheader(f"Predicted pollutant levels for the station {station_id} in {year_input}:")
        pollutants = ['O2', 'NO3', 'NO2', 'SO4', 'PO4', 'CL']
        for p, val in zip(pollutants, prediction[0]):
            st.write(f'{p}: {val:.2f}')
