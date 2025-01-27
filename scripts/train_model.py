import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
import os

# File paths
FEATURES_FILE = os.path.join("data", "features.npy")
LABELS_FILE = os.path.join("data", "labels.npy")
MODEL_FILE = os.path.join("models", "email_classifier.pkl")

def load_data(features_file, labels_file):
    """Load features and labels from .npy files."""
    features = np.load(features_file)
    labels = np.load(labels_file)
    print(f"Loaded features of shape {features.shape} and labels of shape {labels.shape}.")
    return features, labels

def train_model(X_train, y_train):
    """Train a Random Forest model with GridSearchCV."""
    print("Starting model training...")
    rf = RandomForestClassifier(random_state=42)
    
    # Define hyperparameter grid
    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [10, 20, None],
        "min_samples_split": [2, 5, 10],
    }
    
    # Grid search with cross-validation
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring="accuracy", verbose=2)
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print("Model training completed!")
    return grid_search.best_estimator_

def save_model(model, model_file):
    """Save the trained model to a file."""
    with open(model_file, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to {model_file}")

if __name__ == "__main__":
    print("Loading data...")
    
    # Step 1: Load features and labels
    X, y = load_data(FEATURES_FILE, LABELS_FILE)
    
    # Step 2: Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Data split into {X_train.shape[0]} training samples and {X_test.shape[0]} testing samples.")
    
    # Step 3: Train the model
    model = train_model(X_train, y_train)
    
    # Step 4: Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy on test set: {accuracy * 100:.2f}%")
    
    # Step 5: Save the trained model
    save_model(model, MODEL_FILE)
