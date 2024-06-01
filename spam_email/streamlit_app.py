import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import os

# Paths to the model and tokenizer files
model_path = 'spam_email/my_model.h5'
tokenizer_path = 'spam_email/tokenizer.pkl'

# Check if the files exist
if not os.path.exists(model_path):
    st.error(f"Model file not found at {model_path}")
if not os.path.exists(tokenizer_path):
    st.error(f"Tokenizer file not found at {tokenizer_path}")

# Load the model and tokenizer
model = load_model(model_path)
with open(tokenizer_path, 'rb') as handle:
    tokenizer = pickle.load(handle)

# Function to preprocess the input text
def preprocess_text(text, tokenizer, max_len):
    text = text.lower()
    sequences = tokenizer.texts_to_sequences([text])
    padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post')
    return padded_sequences

# Streamlit app
st.title("Spam Email Detection")

email_input = st.text_area("Enter the email content:")
if st.button("Check Email"):
    if email_input:
        # Preprocess the input email content with the correct max_len
        processed_input = preprocess_text(email_input, tokenizer, max_len=500)  # Use max_len=500
        # Predict
        prediction = model.predict(processed_input)
        is_spam = (prediction > 0.5).astype("int32")[0][0]
        
        if is_spam:
            st.error("Warning: This email is likely spam!")
        else:
            st.success("This email seems to be safe.")
    else:
        st.error("Please enter email content to check.")
