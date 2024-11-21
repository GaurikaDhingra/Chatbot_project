# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 17:21:58 2024

@author: gauri
"""

## run streamlit in anaconda terminal 
## - streamlit run "D:/Gaurika Dhingra/Gaurika_CS/Chatbot_Project/app.py"

import streamlit as st
import joblib
import requests
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


# Download NLTK Data
nltk.download('stopwords')
nltk.download('punkt')

# Load the Model and Vectorizer
model = joblib.load("D:/Gaurika Dhingra/Gaurika_CS/Chatbot_Project/emotion_classifier_model.pkl")
vectorizer = joblib.load("D:/Gaurika Dhingra/Gaurika_CS/Chatbot_Project/vectorizer.pkl")

# Emotion Labels Mapping (from integer prediction to emotion name)
emotion_labels = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear"
}

# Preprocessing Function
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    tokens = word_tokenize(text)  # Tokenize
    tokens = [word for word in tokens if word not in stopwords.words('english')]  # Remove stopwords
    return ' '.join(tokens)

# # Fetch Quote Function
# def fetch_quote(emotion):
#     api_url = "https://api.quotable.io/random"
#     tag_map = {
#         "joy": "happiness",
#         "sadness": "sadness",
#         "love": "love",
#         "anger": "anger",
#         "fear": "courage"
#     }
#     tag = tag_map.get(emotion, "inspiration")
#     try:
#         response = requests.get(api_url, params={"tags": tag})
#         if response.status_code == 200:
#             data = response.json()
#             return f'"{data["content"]}" - {data["author"]}'
#         else:
#             return "Sorry, no quotes available at the moment."
#     except Exception as e:
#         return "An error occurred while fetching the quote."

# Streamlit App
st.title("Text Emotion Classifier")
st.subheader("Detect emotions like sadness, joy, love, anger, and fear from your text!")

# User Input
user_input = st.text_area("Enter your text:")

if st.button("Classify Emotion"):
    if user_input.strip():
        # Preprocess and classify
        preprocessed_text = preprocess_text(user_input)
        vectorized_text = vectorizer.transform([preprocessed_text])
        prediction = model.predict(vectorized_text)[0]
        
        # Map prediction to emotion label
        emotion_str = emotion_labels.get(prediction, "Unknown").capitalize()
        
        # Display Emotion
        st.success(f"The detected emotion is: **{emotion_str}**")
        
    else:
        st.warning("Please enter some text for classification!")
        
        
        