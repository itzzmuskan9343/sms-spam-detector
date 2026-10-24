import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

st.set_page_config(page_title="SMS Spam Detector", page_icon="📩")

@st.cache_resource
def load_model():
    df = pd.read_csv("train_data.csv")
    vec = TfidfVectorizer(stop_words="english", lowercase=True)
    X = vec.fit_transform(df["message"].astype(str))
    model = MultinomialNB(alpha=0.3).fit(X, df["label"])
    return vec, model

vec, model = load_model()

st.title("📩 SMS Spam Detector")
st.write("Paste any message and the model will tell you whether it is spam or not. Works with English and Hinglish messages.")

msg = st.text_area("Paste your message here", height=120)

if st.button("Check message"):
    if not msg.strip():
        st.warning("Please enter a message first.")
    else:
        prob = model.predict_proba(vec.transform([msg]))[0][1]
        if prob >= 0.5:
            st.error(f"🚨 Spam ({prob*100:.0f}% confidence)")
        else:
            st.success(f"✅ Not Spam ({(1-prob)*100:.0f}% confidence)")

with st.expander("About the model"):
    st.write("Built with TF-IDF and Naive Bayes, trained on the SMS Spam Collection dataset plus a small set of Hinglish examples. Hindi written in Devanagari script is not supported.")
