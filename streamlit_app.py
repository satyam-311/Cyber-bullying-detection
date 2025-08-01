
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

st.title("Text Classification: Predict on Any Dataset")

# Step 1: Load and train on fixed dataset
@st.cache_data
def train_model():
    # Load your training dataset
    train_df = pd.read_csv("aggression_parsed_dataset.csv")  # Replace with your actual path
    X_train = train_df["headline"].astype(str)
    y_train = train_df["label"]

    # Vectorize and train
    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    model = LogisticRegression()
    model.fit(X_train_vec, y_train)

    return model, vectorizer

model, vectorizer = train_model()
st.success("Model trained on aggression_parsed_dataset.csv")

# Step 2: Upload new dataset for prediction
uploaded_file = st.file_uploader("Upload a new dataset for prediction", type=["csv"])
if uploaded_file:
    new_df = pd.read_csv(uploaded_file)
    st.write("Preview of uploaded data:")
    st.dataframe(new_df.head())

    # Step 3: Auto-detect text column
    text_col = None
    for col in new_df.columns:
        if new_df[col].dtype == "object":
            text_col = col
            break

    if text_col:
        st.success(f"Detected text column: `{text_col}`")

        # Step 4: Predict
        X_new = new_df[text_col].astype(str)
        X_new_vec = vectorizer.transform(X_new)
        predictions = model.predict(X_new_vec)

        # Show results
        new_df["prediction"] = predictions
        st.subheader("Predictions")
        st.dataframe(new_df[[text_col, "prediction"]])

        # Optional: Download results
        # st.download_button("Download predictions", new_df.to_csv(index=False), "predictions.csv")

    else:
        st.error("No text column found in uploaded dataset.")
