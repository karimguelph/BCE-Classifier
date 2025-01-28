import pandas as pd
import re
import os

# Paths for the data files
LEGIT_FILE = os.path.join("data", "legit.csv")
PHISHING_FILE = os.path.join("data", "phishing.csv")
OUTPUT_FILE = os.path.join("data", "processed_emails.csv")

def load_and_merge_data(legit_path, phishing_path):
    """Load legitimate and phishing emails, and merge them into a single DataFrame."""
    try:
        legit = pd.read_csv(legit_path)
        phishing = pd.read_csv(phishing_path)
        print(f"Loaded {legit.shape[0]} legitimate emails and {phishing.shape[0]} phishing emails.")
        
        # Assign labels: 0 for legitimate, 1 for phishing
        legit["label"] = 0
        phishing["label"] = 1

        # Combine the datasets
        data = pd.concat([legit, phishing], ignore_index=True)
        print(f"Merged dataset contains {data.shape[0]} rows and {data.shape[1]} columns.")
        return data
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None

def clean_text(text):
    """Clean text by removing special characters, links, and extra spaces."""
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # Remove special characters
    text = re.sub(r"\s+", " ", text)  # Replace multiple spaces with a single space
    return text.strip()

def preprocess_data(data):
    """Clean and preprocess the data."""
    if "body" not in data.columns:  # Replace 'body' if the actual column name differs
        raise ValueError("The dataset does not have the expected column for email content.")

    # Clean the email content
    data["cleaned_message"] = data["body"].apply(lambda x: clean_text(str(x)))
    
    # Drop duplicates and missing values
    data.drop_duplicates(subset=["cleaned_message"], inplace=True)
    data.dropna(subset=["cleaned_message"], inplace=True)
    
    print(f"After cleaning, dataset has {data.shape[0]} rows and {data.shape[1]} columns.")
    return data

def save_data(data, file_path):
    """Save the processed data to a CSV file."""
    data.to_csv(file_path, index=False)
    print(f"Processed data saved to {file_path}")

if __name__ == "__main__":
    print("Starting data preprocessing...")

    # Step 1: Load and merge the datasets
    raw_data = load_and_merge_data(LEGIT_FILE, PHISHING_FILE)
    if raw_data is not None:
        # Step 2: Preprocess the data
        processed_data = preprocess_data(raw_data)

        # Step 3: Save the processed data
        save_data(processed_data, OUTPUT_FILE)

    print("Data preprocessing completed!")
