import streamlit as st
import joblib
import re
import pandas as pd

# -----------------------------
# Load Models
# -----------------------------
spam_model = joblib.load("spam_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")
phishing_model = joblib.load("phishing_model.pkl")

# -----------------------------
# Full Phishing Feature Extractor  (25 features)
# -----------------------------
def extract_phishing_features(text):
    url_regex = r"(https?://\S+|www\.\S+)"
    email_regex = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"

    features = {
        "length": len(text),
        "num_digits": sum(c.isdigit() for c in text),
        "num_uppercase": sum(c.isupper() for c in text),
        "num_lowercase": sum(c.islower() for c in text),
        "num_spaces": text.count(" "),
        "num_special_chars": sum(not c.isalnum() and c != " " for c in text),
        "num_urls": len(re.findall(url_regex, text)),
        "num_emails": len(re.findall(email_regex, text)),
        "num_dots": text.count("."),
        "num_slashes": text.count("/"),
        "num_question_marks": text.count("?"),
        "num_exclamations": text.count("!"),
        "num_words": len(text.split()),
        "avg_word_len": (len(text) / len(text.split())) if len(text.split()) > 0 else 0,

        # Keyword flags
        "has_bank": int("bank" in text.lower()),
        "has_verify": int("verify" in text.lower()),
        "has_password": int("password" in text.lower()),
        "has_login": int("login" in text.lower()),
        "has_limited": int("limited" in text.lower()),
        "has_urgent": int("urgent" in text.lower()),
        "has_click": int("click" in text.lower()),
        "has_account": int("account" in text.lower()),
        "has_security": int("security" in text.lower()),
        "has_alert": int("alert" in text.lower()),
        "has_suspended": int("suspended" in text.lower()),
        "has_action_required": int("action required" in text.lower())
    }

    return pd.DataFrame([features])


# -----------------------------
# UI DESIGN
# -----------------------------
st.set_page_config(page_title="Spam + Phishing Detector", layout="wide")

st.markdown(
    """
    <h1 style='text-align: center; color: #2C3E50;'>🔐 Email Spam & Phishing Detector</h1>
    <p style='text-align: center; font-size: 18px;'>
        Instantly check if a message is <b>Safe</b>, <b>Spam</b>, or <b>Phishing Attempt</b>.
    </p>
    <hr>
    """,
    unsafe_allow_html=True,
)

# Input field
user_input = st.text_area("✉️ Enter the Email / Message Below:", height=200)

if st.button("🔍 Analyze Message"):
    if user_input.strip() == "":
        st.warning("Please enter a message first.")
    else:
        # -----------------------------
        # SPAM PREDICTION
        # -----------------------------
        spam_features = tfidf.transform([user_input])
        spam_pred = spam_model.predict(spam_features)[0]
        spam_prob = spam_model.predict_proba(spam_features)[0][1]

        # -----------------------------
        # PHISHING PREDICTION
        # -----------------------------
        phishing_features = extract_phishing_features(user_input)

        # If the model was trained with a DataFrame, it likely has feature_names_in_.
        # Reindex to the same columns (order) to avoid ValueError about feature names.
        try:
            if hasattr(phishing_model, "feature_names_in_"):
                expected = list(phishing_model.feature_names_in_)
                # reindex will add any missing columns with NaN -> we fill with 0
                phishing_features = phishing_features.reindex(columns=expected, fill_value=0)
                X_phish = phishing_features.values
            else:
                # If model wasn't trained with feature names, pass a numpy array
                X_phish = phishing_features.values

            phish_pred = phishing_model.predict(X_phish)[0]
            phish_prob = phishing_model.predict_proba(X_phish)[0][1]
        except Exception as e:
            # Show helpful debugging info in the app and stop gracefully
            st.error("Error when running phishing prediction. See details below.")
            st.exception(e)
            st.stop()

        # -----------------------------
        # Results Display
        # -----------------------------
        st.subheader("🔎 Results")

        # Spam Result
        if spam_pred == 1:
            st.error(f"📌 **Spam Detected!**  (Probability: {spam_prob:.2f})")
        else:
            st.success(f"📌 **Not Spam**  (Probability: {spam_prob:.2f})")

        # Phishing Result
        if phish_pred == 1:
            st.error(f"⚠️ **Phishing Attempt Detected!** (Probability: {phish_prob:.2f})")
        else:
            st.success(f"🛡️ **Not Phishing**  (Probability: {phish_prob:.2f})")

        st.info("Analysis complete.")

# Footer
st.markdown(
    """
    <hr>
    <p style='text-align: center; color: gray;'>
    Built with ❤️ by Lavisha | AI + ML Security Project
    </p>
    """,
    unsafe_allow_html=True,
