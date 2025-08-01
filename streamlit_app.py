
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score

# NLTK downloads
nltk.download('stopwords')
nltk.download('wordnet')

# Page config
st.set_page_config(page_title="Cyberbullying Detection", layout="wide")
st.title("Cyberbullying Detection App")

# Clean text
def clean_text(text):
    text = re.sub(r"[^a-zA-Z#]", " ", text)
    text = ' '.join([w for w in text.split() if len(w) > 3])
    lemmatizer = WordNetLemmatizer()
    text = ' '.join([lemmatizer.lemmatize(w) for w in text.split()])
    return text

# Load custom stopwords
@st.cache_data
def load_stopwords():
    with open("stopwords.txt", "r") as f:
        return f.read().splitlines()

# Preprocess
@st.cache_data
def preprocess(df):
    df = df.copy()
    # Label mapping: -1 = cyberbullying → 1, 0 = non-bullying → 0
    df['label'] = df['label'].replace(-1, 1)
    df['headline'] = df['headline'].astype(str).apply(clean_text)
    return df

# Vectorize using TF-IDF
@st.cache_data
def vectorize_data(X_train, X_test, stopword_list):
    tfidf = TfidfVectorizer(stop_words=stopword_list)
    X_train_vec = tfidf.fit_transform(X_train)
    X_test_vec = tfidf.transform(X_test)
    with open("tfidf_vector.pkl", "wb") as f:
        pickle.dump(tfidf, f)
    return X_train_vec, X_test_vec

# Train & evaluate model
def train_and_evaluate(X_train, y_train, X_test, y_test):
    model = LinearSVC()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    with open("LinearSVC.pkl", "wb") as f:
        pickle.dump(model, f)
    return classification_report(y_test, preds, output_dict=True), preds

# UI section
st.markdown("Upload your own dataset, or use the default Hinglish cyberbullying dataset.")

use_default = st.checkbox("Use built-in Hinglish dataset")

if use_default:
    try:
        df = pd.read_csv("final_dataset_hinglish.csv")
        st.success("Using built-in dataset: `final_dataset_hinglish.csv`")
    except Exception as e:
        st.error(f"Could not load default dataset: {e}")
        st.stop()
else:
    uploaded_file = st.file_uploader("Upload your CSV file (must have 'headline' and 'label')", type=["csv"])
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("CSV file loaded successfully.")
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
            st.stop()
    else:
        st.info("👈 Upload a CSV file or use the default dataset to begin.")
        st.stop()

# Main logic
st.subheader("📄 Data Preview")
st.dataframe(df.head())

# Show original label distribution
st.subheader("🔍 Original Label Distribution")
fig1, ax1 = plt.subplots()
sns.countplot(x='label', data=df, ax=ax1)
st.pyplot(fig1)

if 'headline' in df.columns and 'label' in df.columns:
    df_clean = preprocess(df)

    # Show cleaned label distribution
    st.subheader("✅ Cleaned Label Distribution")
    fig2, ax2 = plt.subplots()
    sns.countplot(x='label', data=df_clean, ax=ax2)
    st.pyplot(fig2)

    X_train, X_test, y_train, y_test = train_test_split(df_clean['headline'], df_clean['label'], test_size=0.33, random_state=42)
    stopword_list = load_stopwords()
    X_train_vec, X_test_vec = vectorize_data(X_train, X_test, stopword_list)

    st.subheader("🚀 Training Model...")
    report, preds = train_and_evaluate(X_train_vec, y_train, X_test_vec, y_test)

    st.subheader("📈 Evaluation Report")
    st.json(report)

else:
    st.error("CSV must contain both 'headline' and 'label' columns.")
