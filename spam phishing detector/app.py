import streamlit as st
import joblib
import pandas as pd
from phishing_features import extract_features
from clean_text import clean_text

# Load saved models
tfidf = joblib.load("tfidf_vectorizer.pkl")
spam_clf = joblib.load("spam_model.pkl")
phish_clf = joblib.load("phishing_model.pkl")

st.title("📧 AI Email Spam + Phishing Detector")
st.write("Detect harmful emails using ML + NLP + Security features.")

email_input = st.text_area("Paste your email content here:")

if st.button("Analyze Email"):
    if email_input.strip() == "":
        st.warning("Please enter some email text.")
    else:
        # Clean text
        cleaned = clean_text(email_input)
        tfidf_vec = tfidf.transform([cleaned])
        
        # Phishing features
        phish_vec = pd.DataFrame(
            [extract_features(email_input)],
            columns=['url_count', 'domain_flag', 'risky_count', 'digit_count', 'length']
        )
        
        # Predictions
        spam_pred = spam_clf.predict(tfidf_vec)[0]
        phish_pred = phish_clf.predict(phish_vec)[0]

        st.subheader("🔍 Results:")
        
        if spam_pred == 1:
            st.error("🚨 **SPAM DETECTED**")
        elif phish_pred == 1:
            st.error("⚠️ **PHISHING DETECTED**")
        else:
            st.success("✅ SAFE EMAIL")
        
        st.subheader("📊 Security Feature Analysis:")
        st.json(phish_vec.to_dict(orient="records")[0])
