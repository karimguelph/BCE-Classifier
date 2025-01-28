# **BEC Email Classifier using Machine Learning**
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
8. [Datasets Used](#datasets-used)
9. [Challenges Faced](#challenges-faced)
10. [Demo Screenshots](#demo-screenshots)



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
- **GridSearchCV**: Used for optimal hyperparameter selection, ensuring the best combination of parameters like `n_estimators`, `max_depth`, and `min_samples_split` for the Random Forest model.


---

## **Setup and Execution**
### **Step 1: Install Dependencies**
Run the following command to install all required libraries:
```bash
pip install -r requirements.txt
```
### **Step 2: Run Scripts**
Follow these steps to execute the project in sequence:

1. **Data Preparation**: Preprocess the raw dataset and save it for feature extraction.  
```bash
python scripts/preprocess.py
```


2. **Feature Engineering**: Extract numerical features from email text using TF-IDF and save the vectorized output.  
```bash
python scripts/feature_engineering.py
```


3. **Model Training**: Train the Random Forest Classifier with GridSearchCV and save the trained model.  
```bash
python scripts/train_model.py
```


4. **Evaluation**: Evaluate the model on the test dataset and generate performance metrics and visualizations.  
```bash
python scripts/evaluate_model.py
```

5. **Real-Time Detection**: Test the model with custom email inputs to classify them as "Legitimate" or "Phishing".  
```bash
python scripts/detect_email.py
```

---

## **Key Findings and Results**
### **1. Performance Metrics**
- **Accuracy**: Achieved ~99.58% accuracy on the test set.
- **AUC-ROC Score**: 1.00, indicating perfect discrimination between legitimate and phishing emails.

### **2. Visual Results**
- **Confusion Matrix**:
  Shows how many emails were correctly classified as legitimate or phishing.  
  ![Confusion Matrix](outputs/confusion_matrix.png)
  - **Explanation**:  
  The confusion matrix provides insights into the model's performance by showing the number of correct and incorrect classifications:
  - **True Positives (TP)**: Emails correctly classified as phishing.
  - **True Negatives (TN)**: Emails correctly classified as legitimate.
  - **False Positives (FP)**: Legitimate emails incorrectly classified as phishing.
  - **False Negatives (FN)**: Phishing emails incorrectly classified as legitimate.  
  This helps identify areas where the model might need improvement.


- **AUC-ROC Curve**:  
  Demonstrates the model's performance at various classification thresholds.  
  ![AUC-ROC Curve](outputs/roc_curve.png)
  - **Explanation**:  
  The AUC-ROC curve evaluates the model's ability to distinguish between classes (legitimate vs. phishing). A perfect score of **1.0** indicates that the model is highly effective in separating the two classes without overlap.


---

## **Challenges Faced**

1. **Dataset Limitations and Overfitting**:  
   When starting with the **[Enron Email Dataset](https://www.kaggle.com/datasets/wcukierski/enron-email-dataset)**, I trained the model for several hours and achieved an accuracy of **100%**.  
   - I later realized the Enron dataset contained **only legitimate emails**, causing the model to overfit and classify everything as "Legitimate."  
   - It was a hard lesson but essential for identifying the limitations of real-world datasets.

2. **Switching to a New Dataset**:  
   After discovering the issue, I transitioned to using the **[Human-LLM Generated Phishing and Legitimate Emails Dataset](https://www.kaggle.com/datasets/francescogreco97/human-llm-generated-phishing-legitimate-emails)**.  
   - This dataset has **balanced classes** (2000 phishing emails and 2000 legitimate emails), which provided a much better foundation for training and evaluation.  
   - It also includes **LLM-generated phishing and legitimate emails**, ensuring the model can handle modern phishing tactics.

3. **Future Enhancements**:  
   My plan is to integrate the **[Phishing Email Dataset](https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset/data)**, which has **82,500 phishing emails**, with a balanced sample (~82,000 emails) from the Enron dataset.  
   - This will allow the classifier to handle larger, more diverse datasets and avoid bias while ensuring equal representation of phishing and legitimate emails.
   - This step is critical to achieving a robust, production-ready phishing detection system.

---

## **Future Improvements**

1. **Dataset Expansion**:
   - One of the major lessons from this project was realizing the limitations of using only the **[Enron Email Dataset](https://www.kaggle.com/datasets/wcukierski/enron-email-dataset)**, as it contained only legitimate emails. Moving forward, I plan to use datasets like:
     - **[Human-LLM Generated Phishing and Legitimate Emails Dataset](https://www.kaggle.com/datasets/francescogreco97/human-llm-generated-phishing-legitimate-emails)**: This balanced dataset already improved the model significantly, handling both human and AI-generated phishing emails.
     - **[Phishing Email Dataset](https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset/data)**: This dataset includes over **82,500 phishing emails**, which I’ll combine with a balanced subset of around 82,000 emails from the Enron dataset. The goal is to train a more robust and scalable model capable of handling diverse and nuanced phishing attempts.

2. **Interactive Interface**:
   - A future goal is to build a **web application** or **mobile app** for real-time email classification. This will allow users to input email content directly and receive predictions instantly.
   - I also want to implement **visual explanations**, like highlighting suspicious words or links in emails, to provide users with actionable insights.

3. **Testing at Scale**:
   - After fine-tuning the model with the expanded datasets, I plan to test it on larger-scale email datasets to evaluate its performance in real-world scenarios.
   - The ultimate goal is to create a production-ready phishing email detection system with broad applicability across industries.

---

## **Datasets Used**

This project utilized multiple datasets to develop and improve the phishing email classifier:

1. **[Enron Email Dataset](https://www.kaggle.com/datasets/wcukierski/enron-email-dataset)**  
   - Contains over 500,000 legitimate emails from the Enron Corporation.  
   - Initially used for training the model but later discovered to contain only legitimate emails, which skewed the results.  

2. **[Human-LLM Generated Phishing and Legitimate Emails Dataset](https://www.kaggle.com/datasets/francescogreco97/human-llm-generated-phishing-legitimate-emails)**  
   - A balanced multiclass dataset of **4000 emails** containing:
     - **1000 Human-generated legitimate emails.**
     - **1000 Human-generated phishing emails.**
     - **1000 LLM-generated legitimate emails.**
     - **1000 LLM-generated phishing emails.**  
   - Used as the primary dataset for training the current working model.

3. **[Phishing Email Dataset](https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset/data)**  
   - Contains **82,500 phishing emails** from diverse phishing categories.  
   - Planned for future enhancements to combine with a balanced subset from the Enron dataset for a more robust classifier.
---

## **Demo Screenshots**

Below are some screenshots and examples from the project, showcasing different stages of the pipeline and the model's predictions.

### **1. Data Cleaning**
- The raw email data was cleaned and preprocessed from the previous mode, resulting in the `processed_emails.csv` file. This step ensured that only meaningful features were passed to the model.

-Previous Model.
![image](https://github.com/user-attachments/assets/fec24a4f-febc-4049-9513-9241944d8a3d)

-The new Model.
![image](https://github.com/user-attachments/assets/b59f1cc1-7427-4b36-9d97-b142cbfa5353)

---

### **2. Feature Engineering and Model Training**
![image](https://github.com/user-attachments/assets/61948c22-0095-4faf-94e2-41c4fa9cb9c6)

![image](https://github.com/user-attachments/assets/4c6cdfdf-763d-411e-bf22-3281e6ab0c45)

![image](https://github.com/user-attachments/assets/7bf63370-a471-4fad-98eb-02d61ee6259b)

---
### **3. Evaluate Model**
![image](https://github.com/user-attachments/assets/0f14db9b-08b1-4b84-878a-ae93240b01fc)

---
### **4. Feature Engineering with added vectorizer saving feature for real-time detection**

![image](https://github.com/user-attachments/assets/d8519f9b-7bdf-4c5e-b76a-0af0380fff84)

---
### **5. Example Phishing Emails**
Here are a few examples of phishing emails used in the testing process:

#### **Example 1:**
- **Subject**: Urgent: Verify Your Account Information  
- **Body**: Dear Customer, We detected unusual activity in your account. Please verify your account information immediately to avoid suspension. Copy and paste this link into your browser to confirm your details: http://fakebankingsecure.com Regards, Support Team


#### **Example 2:**
- **Subject**: Your Payment Has Been Declined  
- **Body**: Dear User, Your recent payment could not be processed. To continue using our service, please update your billing information immediately. Copy and paste the following link into your browser: http://secure-billpay.com Sincerely, Payment Processing Team
---

### **6. Correctly Identified Phishing Emails**
Here’s an example of the model successfully identifying the phishing emails:
![image](https://github.com/user-attachments/assets/26c9250a-5912-4cc1-af9a-da131d72cb02)

---

### **7. Correctly Identified Legitimate Emails**
Here’s an example of the model correctly classifying a legitimate email:
![image](https://github.com/user-attachments/assets/8b2a1370-4018-4be2-9922-eaba27a6aee4)
