import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load model
data = pickle.load(open('ames_model.pkl', 'rb'))
model = data['model']
features = data['features']

st.title("House Price Prediction")
st.title("(Ames Dataset)")

st.write("Enter basic house details:")

# User Inputs (only important features)
overall_qual = st.slider("Overall Quality (1-10)", 1, 10, 5)
gr_liv_area = st.number_input("Above Ground Living Area (sq ft)", value=1500)
garage_cars = st.number_input("Garage Capacity (cars)", value=2)
total_bsmt_sf = st.number_input("Basement Area (sq ft)", value=800)
year_built = st.number_input("Year Built", value=2000)

# Prediction
if st.button("Predict Price"):

    # Create a default input row (all zeros)
    input_df = pd.DataFrame(np.zeros((1, len(features))), columns=features)

    # Fill important features (ONLY if they exist)
    if 'Overall Qual' in input_df.columns:
        input_df['Overall Qual'] = overall_qual

    if 'Gr Liv Area' in input_df.columns:
        input_df['Gr Liv Area'] = gr_liv_area

    if 'Garage Cars' in input_df.columns:
        input_df['Garage Cars'] = garage_cars

    if 'Total Bsmt SF' in input_df.columns:
        input_df['Total Bsmt SF'] = total_bsmt_sf

    if 'Year Built' in input_df.columns:
        input_df['Year Built'] = year_built

    # Predict (log transform used)
    log_pred = model.predict(input_df)
    prediction = np.exp(log_pred)

    # Convert USD → INR
    price_inr = prediction[0] * 83

    # Convert to Lakhs
    price_lakhs = price_inr / 100000

    st.success(f"💰 Estimated Price: ₹{round(price_lakhs, 2)} Lakhs")