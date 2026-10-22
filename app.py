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
st.write("Koi bhi message paste karo, model batayega spam hai ya nahi. English aur Hinglish dono chalega.")

msg = st.text_area("Message yahan paste karo", height=120)

if st.button("Check karo"):
    if not msg.strip():
        st.warning("Pehle koi message likho.")
    else:
        prob = model.predict_proba(vec.transform([msg]))[0][1]
        if prob >= 0.5:
            st.error(f"🚨 Spam ({prob*100:.0f}% confidence)")
        else:
            st.success(f"✅ Not Spam ({(1-prob)*100:.0f}% confidence)")

with st.expander("Model ke baare mein"):
    st.write("TF-IDF + Naive Bayes, SMS Spam Collection dataset aur kuch Hinglish examples par trained. Devanagari (Hindi script) support nahi hai.")
