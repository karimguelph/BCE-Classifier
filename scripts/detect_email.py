import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import os

# File paths
MODEL_FILE = os.path.join("models", "email_classifier.pkl")
VECTORIZER_FILE = os.path.join("models", "vectorizer.pkl")

def load_model_and_vectorizer(model_path, vectorizer_path):
    """Load the trained model and the vectorizer."""
    try:
        with open(model_path, 'rb') as model_file:
            model = pickle.load(model_file)
        with open(vectorizer_path, 'rb') as vectorizer_file:
            vectorizer = pickle.load(vectorizer_file)
        print("Model and vectorizer loaded successfully!")
        return model, vectorizer
    except FileNotFoundError as e:
        print(f"Error loading model or vectorizer: {e}")
        return None, None

def preprocess_input(subject, body):
    """Combine and preprocess the subject and body of the email."""
    email_content = f"{subject} {body}".strip()
    return email_content

def predict_email(model, vectorizer, email_content):
    """Make a prediction for the given email content."""
    # Transform the input email using the vectorizer
    email_vector = vectorizer.transform([email_content])
    
    # Get predictions and probabilities
    prediction = model.predict(email_vector)[0]
    prediction_proba = model.predict_proba(email_vector)[0]
    
    # Map prediction to label
    label_map = {0: "Legitimate", 1: "Phishing"}
    prediction_label = label_map[prediction]
    confidence = round(prediction_proba[prediction] * 100, 2)
    
    return prediction_label, confidence

if __name__ == "__main__":
    print("Loading model and vectorizer...")
    model, vectorizer = load_model_and_vectorizer(MODEL_FILE, VECTORIZER_FILE)
    
    if model and vectorizer:
        print("Ready to classify emails!")
        while True:
            print("\nEnter the email details:")
            try:
                subject = input("Subject: ")
                body = input("Body: ")
            except EOFError:
                print("Input ended unexpectedly. Exiting...")
                break
            
            # Preprocess the input
            email_content = preprocess_input(subject, body)
            
            # Make prediction
            prediction, confidence = predict_email(model, vectorizer, email_content)
            
            # Display the result
            print("\n--- Prediction ---")
            print(f"Input Email: \nSubject: {subject}\nBody: {body}")
            print(f"Prediction: {prediction}")
            print(f"Confidence: {confidence}%")
            
            # Ask if the user wants to test another email
            retry = input("\nDo you want to classify another email? (yes/no): ").strip().lower()
            if retry != "yes":
                print("Exiting...")
                break
