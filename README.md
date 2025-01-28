# **BEC Email Classifier: Detecting Phishing with Machine Learning**
> A project built on the **Enron Dataset** containing 500,000+ emails, showcasing how machine learning can detect phishing attempts with precision and reliability.

---

## **Table of Contents**
1. [Project Overview](#project-overview)
2. [Objectives](#objectives)
3. [Key Steps](#key-steps)
   - [Data Preparation](#data-preparation)
   - [Feature Engineering](#feature-engineering)
   - [Model Training](#model-training)
   - [Evaluation](#evaluation)
   - [Real-Time Detection](#real-time-detection)
4. [Algorithms Used](#algorithms-used)
5. [Setup and Execution](#setup-and-execution)
6. [Key Findings and Results](#key-findings-and-results)
7. [Future Improvements](#future-improvements)

---

## **Project Overview**
In this project, I created a **BEC Email Classifier** to detect phishing attempts in emails by:
- Leveraging **machine learning models** like Random Forests.
- Preprocessing and extracting text-based features using **TF-IDF Vectorization**.
- Building a **real-time detection pipeline** for new emails.

This end-to-end project focuses on real-world applicability and provides robust results through detailed data preparation, feature engineering, and model evaluation.

---

## **Objectives**
The goal was to:
1. Detect phishing emails in real-time with high accuracy.
2. Build an end-to-end machine learning pipeline.
3. Use a real-world dataset for training and evaluation.

---

## **Key Steps**

### **1. Data Preparation**
The dataset consisted of raw emails with columns like `sender`, `subject`, `body`, and `label`. I performed:
- **Cleaning**: Removed duplicates and NaN values.
- **Preprocessing**: Combined `subject` and `body` columns for meaningful text features.
- **Output**: Saved the cleaned data to `processed_emails.csv`.

### **2. Feature Engineering**
I used **TF-IDF Vectorization** to extract features from email text:
- Set `max_features=5000` to limit vocabulary size.
- Stored the vectorizer as `vectorizer.pkl` for real-time predictions.
- Split data into `features.npy` and `labels.npy`.

### **3. Model Training**
I trained a **Random Forest Classifier**:
- **Hyperparameters**:
  - `max_depth`: Tested [10, 20, None].
  - `n_estimators`: Tested [50, 100, 200].
  - `min_samples_split`: Tested [2, 5, 10].
- Used **GridSearchCV** for optimal hyperparameter selection.
- Saved the model as `email_classifier.pkl`.

### **4. Evaluation**
I evaluated the model using:
- **Metrics**: Accuracy, Precision, Recall, F1-score.
- **Visualizations**:
  - Confusion Matrix: Shows True Positives, True Negatives, False Positives, and False Negatives.
  - AUC-ROC Curve: Assesses model performance at various thresholds.

### **5. Real-Time Detection**
Implemented a script for classifying new emails:
- User inputs the email `subject` and `body`.
- The model outputs:
  - Prediction: **Legitimate** or **Phishing**.
  - Confidence score (e.g., 95%).

---

## **Algorithms Used**
- **TF-IDF Vectorization**: Converts text into numerical features based on term frequency and inverse document frequency.
- **Random Forest Classifier**: A robust ensemble learning method for classification tasks.

---

## **Setup and Execution**
### **Step 1: Install Dependencies**
Run the following command to install all required libraries:
```bash
pip install -r requirements.txt
```
