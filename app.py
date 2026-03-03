import streamlit as st
import joblib
import numpy as np

st.title("Salary Prediction AI")

model = joblib.load("salary_model.pkl")

experience = st.slider("Years of Experience", 0, 15, 3)

if st.button("Predict Salary"):
    prediction = model.predict(np.array([[experience]]))
    st.success(f"Estimated Salary: ${int(prediction[0]):,}")
