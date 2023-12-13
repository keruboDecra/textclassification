import streamlit as st
import re
import numpy as np
import nltk
import sklearn
import joblib

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.stem import WordNetLemmatizer

# Function to clean and preprocess text
def preprocess_text(text):
    text = re.sub(r'http\S+|www\S+|@\S+|#\S+|[^A-Za-z\s]', '', text)
    text = text.lower()
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]
    return ' '.join(tokens)

# Function to predict using the loaded model
def predict_cyberbullying(text):
    # Load the model and vectorizer
    model = joblib.load('random_forest_model.joblib')
    vectorizer = joblib.load('tfidf_vectorizer.joblib')

    # Preprocess the input text
    preprocessed_text = preprocess_text(text)

    # Transform the preprocessed text using the loaded vectorizer
    text_tfidf = vectorizer.transform([preprocessed_text])

    # Make prediction
    prediction = model.predict(text_tfidf)

    return prediction[0]

# Streamlit UI
st.title('Cyberbullying Detection App')

# Input text box
user_input = st.text_area("Enter a text:", "")

# Check if the user has entered any text
if user_input:
    # Make prediction
    prediction = predict_cyberbullying(user_input)

    # Display the prediction
    st.write(f"Prediction: {'Cyberbullying' if prediction == 1 else 'Not Cyberbullying'}")

