import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import os

# Paths for the data files
INPUT_FILE = os.path.join("data", "processed_emails.csv")
FEATURES_FILE = os.path.join("data", "features.npy")
LABELS_FILE = os.path.join("data", "labels.npy")

def load_data(file_path):
    """Load the processed dataset into a pandas DataFrame."""
    try:
        data = pd.read_csv(file_path)
        print(f"Loaded dataset with {data.shape[0]} rows and {data.shape[1]} columns.")
        return data
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None

def extract_features(data, max_features=5000):
    """Extract TF-IDF features from the email text."""
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english")
    X = vectorizer.fit_transform(data["cleaned_message"])
    print(f"Extracted {X.shape[1]} features.")
    return X, vectorizer

def save_features_labels(features, labels):
    """Save features and labels as .npy files."""
    np.save(FEATURES_FILE, features)
    np.save(LABELS_FILE, labels)
    print(f"Features saved to {FEATURES_FILE}")
    print(f"Labels saved to {LABELS_FILE}")

if __name__ == "__main__":
    print("Starting feature engineering...")

    # Step 1: Load the processed dataset
    data = load_data(INPUT_FILE)
    if data is not None:
        # Step 2: Extract features using TF-IDF
        features, vectorizer = extract_features(data)

        # Step 3: Generate labels (if labeled data is present, e.g., a 'label' column)
        # For now, we're assuming all emails are legitimate since no labels exist.
        # Replace this with actual label extraction logic if labeled data is available.
        labels = np.zeros(data.shape[0])  # Default to 0 (legitimate)

        # Step 4: Save features and labels
        save_features_labels(features.toarray(), labels)

    print("Feature engineering completed!")
