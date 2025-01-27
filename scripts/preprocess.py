import pandas as pd
import re
import os

# Paths for the data files
INPUT_FILE = os.path.join("data", "emails.csv")
OUTPUT_FILE = os.path.join("data", "processed_emails.csv")

def load_data(file_path):
    """Load the dataset into a pandas DataFrame."""
    try:
        data = pd.read_csv(file_path, encoding="latin1")
        print(f"Loaded dataset with {data.shape[0]} rows and {data.shape[1]} columns.")
        return data
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None

def clean_text(text):
    """Clean text by removing special characters, links, and extra spaces."""
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # Remove special characters
    text = re.sub(r"\s+", " ", text)  # Replace multiple spaces with a single space
    return text.strip()

def preprocess_data(data):
    """Clean and preprocess the data."""
    if "message" not in data.columns:
        raise ValueError("The dataset does not have a 'message' column.")
    
    # Clean the email content
    data["cleaned_message"] = data["message"].apply(lambda x: clean_text(str(x)))
    
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

    # Step 1: Load the raw dataset
    raw_data = load_data(INPUT_FILE)
    if raw_data is not None:
        # Step 2: Preprocess the data
        processed_data = preprocess_data(raw_data)

        # Step 3: Save the processed data
        save_data(processed_data, OUTPUT_FILE)

    print("Data preprocessing completed!")
