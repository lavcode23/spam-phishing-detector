import streamlit as st
import joblib
import re
import numpy as np

# -------------------------------
# LOAD MODELS
# -------------------------------
spam_model = joblib.load("models/spam_model.pkl")
tfidf_vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
phishing_model = joblib.load("models/phishing_model.pkl")

# -------------------------------
# SIMPLE PHISHING FEATURE EXTRACTOR
# (kept inside this file to avoid import errors)
# -------------------------------
def extract_phishing_features(url):
    url = url.lower()

    features = [
        len(url),
        url.count("."),
        1 if "https" in url else 0,
        1 if "login" in url else 0,
        1 if "verify" in url else 0,
        1 if "secure" in url else 0,
        1 if "update" in url else 0,
        1 if "account" in url else 0,
        1 if "bank" in url else 0,
        1 if "click" in url else 0,
    ]

    return np.array(features).reshape(1, -1)

# -------------------------------
# MODERN UI
# -------------------------------
st.set_page_config(
    page_title="Spam + Phishing Detector",
    page_icon="⚡",
    layout="centered"
)

st.markdown(
    """
    <h1 style='text-align:center; color:#4A90E2;'>⚡ Spam + Phishing Detection System</h1>
    <p style='text-align:center; font-size:17px; color:#666;'>
        AI powered email & URL safety analyzer
    </p>
    """,
    unsafe_allow_html=True
)

tabs = st.tabs(["📩 Spam Detector", "🔗 Phishing URL Detector"])

# -------------------------------
# TAB 1 - SPAM DETECTOR
# -------------------------------
with tabs[0]:
    st.subheader("📩 Email Spam Classifier")
    email_text = st.text_area("Enter Email Text", height=200)

    if st.button("Analyze Email"):
        if len(email_text.strip()) == 0:
            st.warning("Please enter an email message.")
        else:
            vector = tfidf_vectorizer.transform([email_text])
            prediction = spam_model.predict(vector)[0]
            prob = spam_model.predict_proba(vector)[0][1]

            st.markdown("### Result:")
            if prediction == 1:
                st.error("🚨 **This email is likely SPAM!**")
            else:
                st.success("✅ **This email looks safe.**")

            st.markdown("### Confidence:")
            st.progress(float(prob))

# -------------------------------
# TAB 2 - PHISHING DETECTOR
# -------------------------------
with tabs[1]:
    st.subheader("🔗 Phishing URL Classifier")
    url = st.text_input("Enter URL")

    if st.button("Analyze URL"):
        if len(url.strip()) == 0:
            st.warning("Please enter a URL.")
        else:
            features = extract_phishing_features(url)
            prediction = phishing_model.predict(features)[0]
            prob = phishing_model.predict_proba(features)[0][1]

            st.markdown("### Result:")
            if prediction == 1:
                st.error("🚨 **This URL is likely dangerous (Phishing)!**")
            else:
                st.success("✅ **This URL seems safe.**")

            st.markdown("### Confidence:")
            st.progress(float(prob))

# Footer
st.markdown(
    """
    <hr>
    <p style='text-align:center; color:#999; font-size:14px;'>
        Built by Lavisha Yadav • AI/ML & Cybersecurity
    </p>
    """,
    unsafe_allow_html=True
)
