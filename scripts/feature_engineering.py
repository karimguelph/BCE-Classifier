import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os

# Paths for the data files
INPUT_FILE = os.path.join("data", "processed_emails.csv")
FEATURES_FILE = os.path.join("data", "features.npy")
LABELS_FILE = os.path.join("data", "labels.npy")
VECTORIZER_FILE = os.path.join("models", "vectorizer.pkl")  # New path for saving vectorizer

def load_data(file_path):
    """
    Load the processed dataset into a pandas DataFrame.
    """
    try:
        data = pd.read_csv(file_path)
        print(f"Loaded dataset with {data.shape[0]} rows and {data.shape[1]} columns.")
        return data
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None

def clean_nan_messages(data):
    """
    Ensure no NaN values are in the 'cleaned_message' column.
    """
    data = data.dropna(subset=["cleaned_message"])  # Drop rows where 'cleaned_message' is NaN
    print(f"Removed rows with NaN values. Remaining rows: {data.shape[0]}.")
    return data

def extract_features(data, max_features=5000):
    """
    Extract TF-IDF features from the email text.
    - Uses cleaned_message column to extract text-based features.
    """
    print("Extracting features using TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english")
    X = vectorizer.fit_transform(data["cleaned_message"])
    print(f"Extracted {X.shape[1]} features from the text data.")
    return X, vectorizer

def extract_labels(data):
    """
    Extract labels from the dataset.
    - Assumes 'label' column exists in the data for labeling.
    """
    if "label" not in data.columns:
        raise ValueError("The dataset does not contain a 'label' column.")
    labels = data["label"].values
    print(f"Extracted {len(labels)} labels.")
    return labels

def save_features_labels(features, labels):
    """
    Save features and labels as .npy files.
    """
    np.save(FEATURES_FILE, features)
    np.save(LABELS_FILE, labels)
    print(f"Features saved to {FEATURES_FILE}")
    print(f"Labels saved to {LABELS_FILE}")

def save_vectorizer(vectorizer):
    """
    Save the vectorizer as a .pkl file for later use.
    """
    with open(VECTORIZER_FILE, "wb") as f:
        pickle.dump(vectorizer, f)
    print(f"Vectorizer saved to {VECTORIZER_FILE}")

if __name__ == "__main__":
    print("Starting feature engineering...")

    # Step 1: Load the processed dataset
    data = load_data(INPUT_FILE)
    if data is not None:
        # Step 2: Remove rows with NaN in 'cleaned_message'
        data = clean_nan_messages(data)

        # Step 3: Extract features using TF-IDF
        features, vectorizer = extract_features(data)

        # Step 4: Extract labels
        labels = extract_labels(data)

        # Step 5: Save features and labels
        save_features_labels(features.toarray(), labels)

        # Step 6: Save the vectorizer
        save_vectorizer(vectorizer)

    print("Feature engineering completed!")
