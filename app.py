import streamlit as st
import librosa
import numpy as np
from tensorflow.keras.models import load_model
from utils import extract_mfcc

# Load the trained model
model = load_model('animal_health_model.h5')

# Streamlit app
st.title("Animal Health Detection from Sound")
st.write("Upload an audio file of an animal sound, and we'll tell if the animal is healthy or not!")

uploaded_file = st.file_uploader("Choose an audio file", type=["wav"])

if uploaded_file is not None:
    # Extract MFCC features from the uploaded file
    mfcc_features = extract_mfcc(uploaded_file)
    mfcc_features = np.expand_dims(mfcc_features, axis=0)  # Add batch dimension

    # Make a prediction using the model
    prediction = model.predict(mfcc_features)
    health_status = "Healthy" if prediction[0][0] < 0.5 else "Unhealthy"
    
    st.write(f"The animal is: {health_status}")

    # Optional: Add a feature to predict the temperature (if you have data for this).
    # For now, just a placeholder message:
    st.write("Temperature prediction feature will be added later.")
