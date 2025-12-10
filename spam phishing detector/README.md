# Spam + Phishing Email Detection System (ML + NLP + Security)

A Machine Learning–powered system that detects **spam SMS** and **phishing emails** using classical NLP + ML + rule-based security checks.  
This project simulates a **real-world cybersecurity tool** that organizations use to protect users from fraud, scams, and malicious links.

---

## 🚀 Features
### 📩 Spam Detection (NLP + ML)
- Cleans text using preprocessing (lowercase, stopwords, punctuation removal)
- Converts text to vectors using **TF-IDF**
- Trained using **Multinomial Naive Bayes**
- Achieved **97% accuracy**
- Predicts: `Spam` or `Ham (Not Spam)`

### 🛡️ Phishing Email Detection
Uses both text + URL-based features:
- URL length
- Presence of "@" symbol
- Suspicious keywords (verify, login, reset, bank, account)
- Suspicious domain extensions (.xyz, .top, .click, .ru, etc.)
- Model trained using **XGBoost**
- Accuracy: **97%**

---

## 🧠 Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost
- TF-IDF Vectorizer
- BeautifulSoup (HTML parsing)
- Joblib (model saving)
- Streamlit (optional app)

---

## 📂 Project Structure

