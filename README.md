# SMS Spam Detector

A machine learning app that tells whether an SMS message is spam or not. Works with English and Hinglish messages.

**Live app:** https://muskan-sms-spam-detector.streamlit.app/

## How it works
1. Text messages are converted into numbers using TF-IDF.
2. A Naive Bayes classifier is trained on labelled messages (spam / ham).
3. The model is deployed as a Streamlit web app for real-time prediction.

## Results
- Trained on the SMS Spam Collection dataset (5,572 messages), split 80% train and 20% test
- Naive Bayes reached 98.48% accuracy on 1,115 test messages
- The first model had 84% spam recall and failed on Hinglish and phishing messages
- After adding a small set of Hinglish and phishing examples, spam recall improved to 91%

## Tech stack
Python, pandas, scikit-learn, Streamlit

## Limitations
- The Hinglish examples are few (80) and self-written, so this is a proof of concept
- The main dataset is old English SMS data
- Hindi written in Devanagari script is not supported

## Files
- `app.py`: Streamlit app
- `train_data.csv`: training data
- `requirements.txt`: dependencies
