import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Define the main directory
main_directory = 'FINAL_PROJECT'

# Paths to the phishing URL model and tokenizer files
phishing_model_path = os.path.join(main_directory, 'phishing_url', 'phishing_model.h5')
phishing_tokenizer_path = os.path.join(main_directory, 'phishing_url', 'phishing_tokenizer.pkl')

# Paths to the spam email model and tokenizer files
spam_model_path = os.path.join(main_directory, 'spam_email', 'my_model.h5')
spam_tokenizer_path = os.path.join(main_directory, 'spam_email', 'tokenizer.pkl')

# Load the phishing URL model and tokenizer
if not os.path.exists(phishing_model_path):
    st.error(f"Phishing URL model file not found at {phishing_model_path}")
else:
    logging.info(f"Loading phishing URL model from {phishing_model_path}")

if not os.path.exists(phishing_tokenizer_path):
    st.error(f"Phishing URL tokenizer file not found at {phishing_tokenizer_path}")
else:
    logging.info(f"Loading phishing URL tokenizer from {phishing_tokenizer_path}")

phishing_model = load_model(phishing_model_path)
with open(phishing_tokenizer_path, 'rb') as handle:
    phishing_tokenizer = pickle.load(handle)

# Load the spam email model and tokenizer
if not os.path.exists(spam_model_path):
    st.error(f"Spam email model file not found at {spam_model_path}")
else:
    logging.info(f"Loading spam email model from {spam_model_path}")

if not os.path.exists(spam_tokenizer_path):
    st.error(f"Spam email tokenizer file not found at {spam_tokenizer_path}")
else:
    logging.info(f"Loading spam email tokenizer from {spam_tokenizer_path}")

spam_model = load_model(spam_model_path)
with open(spam_tokenizer_path, 'rb') as handle:
    spam_tokenizer = pickle.load(handle)

# Function to preprocess the input text for phishing URL model
def preprocess_text_url(text, tokenizer, max_len):
    text = text.lower()
    sequences = tokenizer.texts_to_sequences([text])
    padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post')
    return padded_sequences

# Function to preprocess the input text for spam email model
def preprocess_text_email(text, tokenizer, max_len):
    text = text.lower()
    sequences = tokenizer.texts_to_sequences([text])
    padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post')
    return padded_sequences

# Streamlit app title
st.title('Spam and Phishing Detection')

# Tabs for different functionalities
tab1, tab2 = st.tabs(["Phishing URL Detection", "Spam Email Detection"])

with tab1:
    st.header("Phishing URL Detection")
    url_input = st.text_area("Enter the URL to check:")
    if st.button("Check URL"):
        if url_input:
            # Preprocess the input URL content with the correct max_len
            processed_input = preprocess_text_url(url_input, phishing_tokenizer, max_len=177)
            # Predict
            prediction = phishing_model.predict(processed_input)
            is_phishing = (prediction > 0.5).astype("int32")[0][0]
            
            if is_phishing:
                st.error("Warning: This URL is likely a phishing attempt!")
            else:
                st.success("This URL seems to be safe.")
        else:
            st.error("Please enter a URL to check.")

with tab2:
    st.header("Spam Email Detection")
    email_input = st.text_area("Enter the email content:")
    if st.button("Check Email"):
        if email_input:
            # Preprocess the input email content with the correct max_len
            processed_input = preprocess_text_email(email_input, spam_tokenizer, max_len=500)  # Use max_len=500
            # Predict
            prediction = spam_model.predict(processed_input)
            is_spam = (prediction > 0.5).astype("int32")[0][0]
            
            if is_spam:
                st.error("Warning: This email is likely spam!")
            else:
                st.success("This email seems to be safe.")
        else:
            st.error("Please enter email content to check.")
