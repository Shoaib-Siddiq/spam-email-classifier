"""Interactive Streamlit demo for the Spam Email Classifier.

Run:  streamlit run app/app.py
"""
import sys
from pathlib import Path

import joblib
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import clean_text

st.set_page_config(page_title="Spam Email Classifier", page_icon="📧", layout="wide")

st.title("📧 Spam Email / SMS Classifier")
st.caption("Real-time spam detection with TF-IDF + Linear SVM, trained on the UCI SMS Spam Collection "
           "(5,574 messages, 13.4% spam).")

@st.cache_resource
def load_model():
    return joblib.load(ROOT / "models" / "spam_classifier_best.joblib")

pipe = load_model()

example = st.selectbox(
    "Try an example:",
    [
        "Custom message…",
        "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's",
        "Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat...",
        "WINNER!! As a valued network customer you have been selected to receivea £900 prize reward! To claim call 09061701461. Claim code KL341. Valid 12 hours only.",
        "I've been searching for the right words to thank you for this breather. Thank you so much.",
        "URGENT! You have won a 1 week FREE membership in our £100000 prize Jackpot! Txt the word: CLAIM to No: 81010 T&C www.dbuk.net LCCLTD POBOX 4403LDNW1A7RW18",
    ],
)

message = st.text_area("Or type a message:", example if example != "Custom message…" else "", height=130)

if st.button("Classify", type="primary"):
    if not message.strip():
        st.warning("Please type a message first.")
    else:
        cleaned = clean_text(message)
        prob = pipe.predict_proba([cleaned])[0, 1]
        pred = int(pipe.predict([cleaned])[0])

        c1, c2 = st.columns(2)
        with c1:
            st.metric("Verdict", "🚫 SPAM" if pred == 1 else "✅ HAM", delta=None)
        with c2:
            st.metric("Spam probability", f"{prob:.2%}")
        st.progress(float(prob))

        if pred == 1:
            st.error("This message looks like spam — typical patterns detected.")
        else:
            st.success("This message looks legit.")

st.divider()
st.markdown("""
**How it works:** the message is normalized (lowercased, URLs/numbers tokenized, stop words removed),
converted to a TF-IDF vector (1–2 word n-grams), then scored by a Linear SVM trained on 5,574
real SMS messages. **98.65% accuracy · 92.6% spam recall · F1 0.95 · ROC-AUC 0.99.**
""")
