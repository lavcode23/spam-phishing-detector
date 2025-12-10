import streamlit as st
import joblib
import pandas as pd
from phishing_features import extract_phishing_features
from clean_text import clean_text

# Load models & vectorizer
spam_model = joblib.load("spam_model.pkl")
phishing_model = joblib.load("phishing_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

st.title("📧 Spam + Phishing Email Detector")
st.write("Enter an email below to classify it as **Ham / Spam / Phishing**.")

email_text = st.text_area("Enter Email Text")

if st.button("Predict"):
    if len(email_text.strip()) == 0:
        st.error("Please enter some text!")
    else:
        # Spam detection
        cleaned = clean_text(email_text)
        tfidf_input = vectorizer.transform([cleaned])
        spam_pred = spam_model.predict(tfidf_input)[0]

        # Phishing detection
        phish_features = extract_phishing_features(email_text)
        phish_df = pd.DataFrame([phish_features])
        phish_pred = phishing_model.predict(phish_df)[0]

        # Output
        st.subheader("📌 Result")
        if phish_pred == 1:
            st.error("⚠️ This email is likely **PHISHING**!")
        elif spam_pred == 1:
            st.warning("🚫 This email is **SPAM**.")
        else:
            st.success("✔ This email is safe (HAM).")
