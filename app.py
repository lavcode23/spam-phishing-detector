import streamlit as st
import joblib
import numpy as np
import re
import string

# Load models & vectorizer
spam_model = joblib.load("spam_model.pkl")
phishing_model = joblib.load("phishing_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

st.set_page_config(page_title="AI Spam + Phishing Detector",
                   layout="centered",
                   page_icon="🛡️")

# ---------------------------
# ------- STYLING -----------
# ---------------------------
st.markdown("""
    <style>
        .safe-box {background: #E7F7ED; padding: 18px; border-radius: 12px; border-left: 10px solid #2ECC71;}
        .spam-box {background: #FDEDEC; padding: 18px; border-radius: 12px; border-left: 10px solid #E74C3C;}
        .phish-box {background: #FFF3CD; padding: 18px; border-radius: 12px; border-left: 10px solid #F1C40F;}
        .highlight {background-color: #FFF59D; padding: 2px 4px; border-radius: 4px;}
    </style>
""", unsafe_allow_html=True)


# ---------------------------
# ------- FUNCTIONS ----------
# ---------------------------

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", " URL ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text

def extract_phishing_features(text):
    text_lower = text.lower()
    return {
        "https_count": text_lower.count("https"),
        "http_count": text_lower.count("http"),
        "login_words": sum(word in text_lower for word in ["verify", "update", "login", "confirm", "password", "bank"]),
        "suspicious_domain": int(".xyz" in text_lower or ".top" in text_lower or ".zip" in text_lower)
    }


def color_words(text):
    suspicious = ["verify", "update", "login", "confirm", "password",
                  "urgent", "bank", "account", "click", "secure"]

    for word in suspicious:
        text = re.sub(fr"\b{word}\b",
                      f"<span class='highlight'>{word}</span>",
                      text, flags=re.I)
    return text


# ---------------------------
# ------- UI HEADER ----------
# ---------------------------

st.title("🛡️ AI Spam + Phishing Email Detector")
st.markdown("### Detects SPAM, PHISHING, and SAFE messages using Machine Learning + NLP")
st.markdown("---")


# ---------------------------
# ------- USER INPUT --------
# ---------------------------

user_text = st.text_area("📩 Paste the email or message you want to check:", height=180)

if st.button("🔍 Analyze"):
    if not user_text.strip():
        st.warning("Please enter a message.")
    else:

        # 1 — SPAM prediction
        cleaned = clean_text(user_text)
        tfidf_vec = vectorizer.transform([cleaned])
        spam_prob = spam_model.predict_proba(tfidf_vec)[0][1]
        spam_pred = spam_model.predict(tfidf_vec)[0]

        # 2 — Phishing prediction
        features = extract_phishing_features(user_text)
        f_values = np.array([[*features.values()]])
        phish_prob = phishing_model.predict_proba(f_values)[0][1]
        phish_pred = phishing_model.predict(f_values)[0]

        # ---------------------------
        # ------- RESULT UI ----------
        # ---------------------------

        st.markdown("### 🧾 **Result:**")

        # PHISHING (highest priority)
        if phish_pred == 1 and phish_prob > 0.65:
            st.markdown(f"""
            <div class='phish-box'>
                <h3>⚠️ PHISHING DETECTED</h3>
                <p>This message shows signs of **credential theft** or suspicious intent.</p>
                <b>Phishing Probability:</b> {phish_prob:.2f}
            </div>
            """, unsafe_allow_html=True)

        # SPAM
        elif spam_pred == "spam":
            st.markdown(f"""
            <div class='spam-box'>
                <h3>🚫 SPAM DETECTED</h3>
                <p>This looks like advertising, scam, or unwanted content.</p>
                <b>Spam Probability:</b> {spam_prob:.2f}
            </div>
            """, unsafe_allow_html=True)

        # SAFE
        else:
            st.markdown(f"""
            <div class='safe-box'>
                <h3>✅ This message looks SAFE</h3>
                <p>No major spam or phishing patterns detected.</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 🔎 Highlighted Text")
        st.markdown(color_words(user_text), unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 🧠 Model Insights")
        st.json({
            "Spam Probability": float(spam_prob),
            "Phishing Probability": float(phish_prob),
            "Phishing Features": features
        })


# ---------------------------
# ------- ABOUT SECTION -------
# ---------------------------

st.markdown("---")
st.markdown("#### 👩‍💻 Built by *Lavisha Yadav* — AI/ML Engineer")
st.markdown("GitHub: [lavcode23](https://github.com/lavcode23)")
