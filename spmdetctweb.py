import streamlit as st
import joblib
import string
from string import punctuation
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from scipy.sparse import hstack
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
nltk.download("punkt")

# Load models
model = joblib.load("/Users/clean/spam-detector/spam_detection_model_compressed.pkl")
vectorizer = joblib.load("/Users/clean/spam-detector/tfidf_vectorizer.pkl")
scaler = joblib.load("/Users/clean/spam-detector/scaler.pkl")

# Preprocessing
ps = PorterStemmer()
def preprocess(text):
    text_nopunct = "".join([char for char in text if char not in string.punctuation])
    tokens = re.split(r"\W+", text_nopunct)
    text_nostop = [word.lower() for word in tokens if word not in stopwords.words('english')]
    stemmed = [ps.stem(word) for word in text_nostop]
    return " ".join(stemmed)

#LOGGING FUNCTION
import datetime

def log_user_input(user_input, prediction):
    with open("user_logs.txt", "a") as log_file:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_file.write(f"{timestamp} | Input: {user_input[:300]} | Prediction: {prediction}\n")

# USER INTERFACE
st.title("JAMIU'S SPAM DETECTION MODEL")
user_input = st.text_area("Input your message here:")

# ACTION BASED ON BUTTON
if st.button("Predict"):
    #Preprocess text
    clean_text = preprocess(user_input)
    tfidf = vectorizer.transform([clean_text]) 

    #Engineer features
    length = len(user_input) - user_input.count(" ")
    punct = round((sum([1 for char in user_input if char in string.punctuation]) / length), 3) * 100 if length > 0 else 0
    scaled = scaler.transform([[length, punct]])

    #Combine TF-IDF and features
    features = hstack([tfidf, scaled])

    #Predict
    prediction = model.predict(features)[0]
    label = "📬 Ham" if prediction == 0 else "🚫 Spam"
    st.success(f"This message is: {label}")

     # 📝 Log input and result
    log_user_input(user_input, "Spam" if prediction == 1 else "Ham")
