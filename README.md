🚫 Spam Detection Web App

This is a simple, interactive spam detection model built with Streamlit and Machine Learning, designed to classify messages as either Ham (📬 legitimate) or Spam (🚫 unwanted) using natural language processing and behavioral features.

📌 Table of Contents

- [About the Project](#about-the-project)
- [Features](#features)
- [How It Works](#how-it-works)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Tech Stack](#tech-stack)
- [How to Run Locally](#how-to-run-locally)
- [Try it Live](#try-it-live)
- [Contributing](#contributing)
- [License](#license)

📖 About the Project

The goal of this project is to build a practical and accessible spam classification system that can be deployed on the web for everyday users to test their messages. It combines traditional NLP techniques with engineered features to improve accuracy.

✨ Features

- ✏️ Text input from the user
- 🤖 Machine learning prediction
- 🔍 Preprocessing pipeline (tokenization, stemming, stopword removal)
- 🧠 TF-IDF vectorization + feature engineering
- 📊 Real-time prediction using a trained model
- 📝 Input logging for feedback analysis

⚙️ How It Works

1. User inputs a message.
2. Message is preprocessed:
   - Lowercase, punctuation removal
   - Tokenization
   - Stopword removal
   - Stemming
3. Features extracted:
   - TF-IDF vector
   - Message length (excluding whitespace)
   - Punctuation percentage
4. All features are scaled and fed to the trained model.
5. Output is displayed to the user and logged.

🧠 Model Architecture

- Text Vectorization: TF-IDF
- Features:
  - Message Length
  - Punctuation Percentage
- Model Type: Logistic Regression (or specify actual model used)
- Scaler: StandardScaler for numeric features

---

📊 Dataset

This project was trained and evaluated on a combination of three datasets to improve accuracy, robustness, and generalizability:

1. UCI SMS Spam Collection Dataset
   - ~5,500 labeled SMS messages  
   - Classes: `ham`, `spam`  
   - [Link](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)

2. Kaggle SMS Spam Dataset 
   - ~5,000+ labeled text messages  
   - Includes more variety in spam phrases, slang, and length.

3. Custom User-Generated Dataset
   - Collected and annotated manually  
   - Includes modern examples (e.g. WhatsApp spam, email pitch scams, bulk ads, etc.)  
   - Helped fine-tune the model for current spam styles

The combined dataset was preprocessed (cleaned, tokenized, and stemmed), vectorized using TF-IDF, and augmented with engineered features before model training.


🛠️ Tech Stack

- Python 3.8+
- Streamlit
- Scikit-learn
- NLTK
- Joblib
- SciPy

🚀 How to Run Locally

```bash
# Clone the repo
git clone https://github.com/SageAnalyst/spam-detector.git
cd spam-detector

# Install dependencies
pip install -r requirements.txt

# Download nltk data (optional for local run)
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"

# Run the app
streamlit run spmdetctweb.py
