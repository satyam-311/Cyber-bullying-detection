
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score

nltk.download('stopwords')
nltk.download('wordnet')

st.set_page_config(page_title="Cyberbullying Detection", layout="wide")
st.title("Cyberbullying Detection App")

def clean_text(text):
    text = re.sub(r"[^a-zA-Z#]", " ", text)
    text = ' '.join([w for w in text.split() if len(w) > 3])
    lemmatizer = WordNetLemmatizer()
    text = ' '.join([lemmatizer.lemmatize(w) for w in text.split()])
    return text

@st.cache_data
def preprocess(df):
    df = df.copy()
    df['label'] = df['label'].replace(-1, 1)
    df['headline'] = df['headline'].astype(str).apply(clean_text)
    return df

@st.cache_data
def vectorize_data(X_train, X_test):
    tfidf = TfidfVectorizer(stop_words=stopwords.words('english'))
    X_train_vec = tfidf.fit_transform(X_train)
    X_test_vec = tfidf.transform(X_test)
    with open("tfidf_vector.pkl", "wb") as f:
        pickle.dump(tfidf, f)
    return X_train_vec, X_test_vec

def train_and_evaluate(X_train, y_train, X_test, y_test):
    model = LinearSVC()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    with open("LinearSVC.pkl", "wb") as f:
        pickle.dump(model, f)
    return classification_report(y_test, preds, output_dict=True), preds

file = st.file_uploader("Upload CSV File (with 'headline' and 'label')", type=["csv"])

if file:
    try:
        df = pd.read_csv(file)
        st.subheader("📄 Data Preview")
        st.dataframe(df.head())

        if 'headline' in df.columns and 'label' in df.columns:
            df_clean = preprocess(df)
            X_train, X_test, y_train, y_test = train_test_split(df_clean['headline'], df_clean['label'], test_size=0.33, random_state=42)
            X_train_vec, X_test_vec = vectorize_data(X_train, X_test)

            st.subheader("🚀 Training Model...")
            report, preds = train_and_evaluate(X_train_vec, y_train, X_test_vec, y_test)

            st.subheader("📈 Evaluation Report")
            st.json(report)

            st.subheader("📊 Class Distribution")
            fig, ax = plt.subplots()
            sns.countplot(x='label', data=df_clean, ax=ax)
            st.pyplot(fig)
        else:
            st.error("Missing 'headline' or 'label' column in uploaded CSV.")
    except Exception as e:
        st.error(f"Error processing file: {e}")
else:
    st.info("👈 Upload a CSV file to start.")
