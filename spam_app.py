import streamlit as st
import pickle
import pandas as pd
import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the trained model and vectorizer
with open("model.pkl", "rb") as model_file:
    model = pickle.load(model_file)
with open("vectorizer.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

nltk.download('stopwords')
stop_words = set(nltk.corpus.stopwords.words('english'))

def preprocess(text):
    """Preprocess the input text by converting to lowercase, removing punctuation and stopwords."""
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

def classify_message(message):
    """Classify the given message as spam or ham."""
    message = preprocess(message)
    message_vector = vectorizer.transform([message])
    prediction = model.predict(message_vector)
    return prediction[0]

# Streamlit UI
st.set_page_config(page_title="Spam Message Classifier", page_icon="📩", layout="centered")

st.title("📩 Spam Message Classifier")
st.markdown("### Detect whether a message is spam or ham in real-time!")
st.write("Type a message below and click 'Predict' to classify it.")

# User input
message = st.text_area("Enter your message here:", "", height=150)

if st.button("Predict", use_container_width=True):
    if message.strip():
        result = classify_message(message)
        if result == "spam":
            st.error("🚨 This message is classified as **Spam**!")
        else:
            st.success("✅ This message is classified as **Ham** (Not Spam).")
    else:
        st.warning("⚠️ Please enter a message to classify.")

# Add some styling
st.markdown(
    """
    <style>
        .stTextArea textarea {font-size: 18px; padding: 10px;}
        .stButton button {background-color: #4CAF50; color: white; font-size: 18px; padding: 10px;}
    </style>
    """,
    unsafe_allow_html=True
)
