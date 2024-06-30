import streamlit as st
import pandas as pd
import pickle

# Load the trained model
with open('model.pkl', 'rb') as model_file:
    clf = pickle.load(model_file)

# Define your input fields
st.title("Bank Account Prediction")
age = st.number_input('Age', min_value=18, max_value=100, step=1)
income = st.number_input('Income', min_value=0)

# validation button clicked
if st.button("Predict"):
    input_data = pd.DataFrame({'age': [age], 'income': [income]})  # Add more fields as needed
    prediction = clf.predict(input_data)
    st.write(f"The prediction is: {prediction[0]}")
