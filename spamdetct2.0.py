import pandas as pd
import numpy as np
import string
import re
import joblib
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from scipy.sparse import hstack
from datasets import load_dataset

# LOAD AND COMBINE DATASETS

# 1.SMS Spam Collection
df_sms = pd.read_csv(r'/Users/clean/Downloads/sms+spam+collection/SMSSpamCollection', sep='\t', header=None, names=["label", "text"])
df_sms['label'] = df_sms['label'].map({'ham': 0, 'spam': 1})

# 2.SpamDam
df_spamdam = pd.read_csv(r"/Users/clean/Downloads/SpamDam_twitter_data.csv")
df_spamdam = df_spamdam.rename(columns={"text": "text", "label": "label"})
# SpamDam is already labeled 1 (spam)

# 3.alusci/sms-otp-spam-dataset from HuggingFace
hug_dataset = load_dataset("alusci/sms-otp-spam-dataset")
df_hug = hug_dataset['train'].to_pandas()
df_hug = df_hug.rename(columns={"sms_text": "text", "label": "label"})
df_hug["label"] = df_hug["label"].map({"spam": 1, "valid": 0})

# 4.Combine all datasets
df = pd.concat([
    df_sms[["text", "label"]],
    df_spamdam[["text", "label"]],
    df_hug[["text", "label"]]
], ignore_index=True)

#DATA CLEANING
df = df.dropna(subset=["label"])
df["label"] = df["label"].astype(int)

#%% EXPLORATIVE DATA
#number of rows and columns in dataset:
print('NUMBER OF ROWS:', len(df),
      "NUMBER OF COLUMNS:", len(df.columns))
#number of ham and spam:
print("NUMBER OF HAM:", len(df[df["label"]=='ham']),
      "NUMBER OF SPAM:", len(df[df["label"]=='spam']))
#number of null values:
print("NUMBER OF LABEL NULL VALUES", df["label"].isnull().sum(),
      "NUMBER OF TEXT NULL VALUES", df["text"].isnull().sum())

# DATA PREPROCESSING
ps = PorterStemmer()
def clean_text(text):
    text_nopunct = "".join([char for char in text if char not in string.punctuation])
    tokens = re.split(r"\W+", text_nopunct)
    text_nostop = [word.lower() for word in tokens if word not in stopwords.words("english")]
    stemmed = [ps.stem(word) for word in text_nostop]
    return stemmed

df["cleantext"] = df["text"].apply(clean_text)
df["cleantext_str"] = df["cleantext"].apply(lambda x: " ".join(x))

# TF-IDF Vectorization
tfidf_vect = TfidfVectorizer(ngram_range=(1, 2), max_df=0.95, min_df=2)
tfidf_matrix = tfidf_vect.fit_transform(df["cleantext_str"])

# Feature engineering: text length and punctuation %
df["text_length"] = df["text"].apply(lambda x: len(x) - x.count(" "))
df["perc_punct"] = df["text"].apply(lambda x: round(
    sum([1 for char in x if char in string.punctuation]) / (len(x) - x.count(" ")), 3) * 100 if (len(x) - x.count(" ")) > 0 else 0
)

# Scale additional features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df[["text_length", "perc_punct"]])

# Combine TF-IDF + engineered features
x = hstack([tfidf_matrix, scaled_features])
y = df["label"].values

# Train/test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=42)

# Apply SMOTE to balance data
sm = SMOTE(random_state=42, k_neighbors=3)
x_train, y_train = sm.fit_resample(x_train, y_train)

# Logistic Regression with GridSearchCV
log_param = {
    'C': [0.01, 0.1, 1, 10],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear'],
    'class_weight': ['balanced']
}
log_grid = GridSearchCV(LogisticRegression(), log_param, scoring='f1', cv=5, verbose=2, n_jobs=-1)
log_grid.fit(x_train, y_train)
best_log = log_grid.best_estimator_

# Random Forest with GridSearchCV
rf_param = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}
rf_grid = GridSearchCV(RandomForestClassifier(random_state=42, class_weight='balanced'),
                       rf_param, scoring='f1_weighted', cv=5, verbose=2, n_jobs=-1)
rf_grid.fit(x_train, y_train)
best_rf = rf_grid.best_estimator_

# Combine with Voting Classifier
voting_clf = VotingClassifier(estimators=[
    ("LogReg", best_log),
    ("RandomForest", best_rf)
], voting='soft')

voting_clf.fit(x_train, y_train)
y_pred = voting_clf.predict(x_test)

# Evaluate model
print("Voting accuracy score:", accuracy_score(y_test, y_pred))
print("Classification report:\n", classification_report(y_test, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

# Save models
joblib.dump(voting_clf, "spam_detection_model.pkl")
joblib.dump(tfidf_vect, "tfidf_vectorizer.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Models saved successfully!")
