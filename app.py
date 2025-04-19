import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os



# Get the current working directory
model_path = os.getcwd()

# Load model and tokenizer
loaded_model = AutoModelForSequenceClassification.from_pretrained(model_path)
loaded_tokenizer = AutoTokenizer.from_pretrained(model_path)

# Define emotion classes
classes = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']

# Streamlit page configuration
st.set_page_config(page_title="Emotion Detection from Text", page_icon="⌨", layout="centered")

# Title
st.title("Emotion Detection from Text")

# Add a description
st.markdown("""
    This app uses a trained model to classify emotions from your text.
    Just type your text below, and it will tell you the predicted emotion.
""")

# Input text box
user_input = st.text_area("Enter text", height=100, placeholder="Type something here...")

# Button to trigger prediction
if st.button("Detect Emotion"):
    if user_input.strip():
        # Tokenize input and get model predictions
        input_encoded = loaded_tokenizer(user_input, return_tensors='pt')
        input_encoded = input_encoded.to('cpu')
        
        with torch.no_grad():
            outputs = loaded_model(**input_encoded)

        logits = outputs.logits
        pred = torch.argmax(logits, dim=1).item()

        # Display the prediction result
        st.success(f"Predicted Emotion: {classes[pred]}")
    else:
        st.error("Please enter some text to analyze.")
        
# Footer
st.markdown("""
    Feel free to share and use this app!
""")
