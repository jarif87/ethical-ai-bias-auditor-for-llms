import streamlit as st
import torch
from transformers import ElectraForSequenceClassification, AutoTokenizer
import numpy as np
import os

# Set page configuration for a clean look with default theme
st.set_page_config(
    page_title="Ethical AI Bias Auditor",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Sidebar for instructions
st.sidebar.title("Ethical AI Bias Auditor")
st.sidebar.markdown("""
This app detects biases in text using a fine-tuned ELECTRA model across six categories: Gender, Racial, Cultural, Age, Religion and Disability.
- Enter text in the main panel to analyze for biases.
- Predictions show bias labels with probabilities (threshold: 0.5).
""")

# Load model and tokenizer from local directories
@st.cache_resource

@st.cache_resource
def load_model_and_tokenizer():
    base_dir = os.path.dirname(__file__)
    model_path = os.path.join(base_dir, "electrabert_model")
    tokenizer_path = os.path.join(base_dir, "tokenizer")
    
    try:
        if not os.path.exists(model_path) or not os.path.exists(tokenizer_path):
            raise FileNotFoundError("Model or tokenizer directory not found.")
        
        model = ElectraForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)
        return model, tokenizer, device
    
    except Exception as e:
        st.error(f"Failed to load model or tokenizer: {str(e)}")
        return None, None, None



model, tokenizer, device = load_model_and_tokenizer()

# Prediction function (no "Text {idx}" prefix)
def predict_bias_custom(texts):
    if not model or not tokenizer:
        return ["Error: Model or tokenizer not loaded."]
    model.eval()
    results = []
    for text in texts:
        try:
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
            inputs = {key: value.to(device) for key, value in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                probs = torch.sigmoid(logits).cpu().numpy()[0]
            labels = ["Gender", "Racial", "Cultural", "Age", "Religion", "Disability"]
            result = [f"{label} Bias ({probs[i]:.2f})" if probs[i] >= 0.5 else f"No {label} Bias ({probs[i]:.2f})" 
                      for i, label in enumerate(labels)]
            results.append(", ".join(result))
        except Exception as e:
            results.append(f"Error processing text - {str(e)}")
    return results

# Main app layout
st.title("Ethical AI Bias Auditor")
st.markdown("Analyze text for potential biases using a fine-tuned ELECTRA model")

# Single Text Prediction Section
st.header("Enter Text for Bias Analysis")
user_input = st.text_area(
    "Input your text here:",
    placeholder="Enter text to analyze for biases, e.g., 'Women are often stereotyped as being less capable in leadership roles.'",
    height=150
)
if st.button("Analyze Text"):
    if user_input:
        with st.spinner("Analyzing text..."):
            predictions = predict_bias_custom([user_input])
            for prediction in predictions:
                st.success(prediction)
    else:
        st.warning("Please enter some text to analyze.")