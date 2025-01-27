import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

import numpy as np
import pickle
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

# Load the trained model
print("Loading the trained model...")
with open('models/email_classifier.pkl', 'rb') as f:
    model = pickle.load(f)

# Load test dataset
print("Loading test data...")
X_test = np.load('data/features.npy')[413920:]
y_test = np.load('data/labels.npy')[413920:]

# Check test data label distribution
print(f"Test set label distribution: {np.unique(y_test, return_counts=True)}")

# Make predictions
print("Making predictions...")
y_pred = model.predict(X_test)

# Handle predict_proba safely
try:
    y_pred_proba = model.predict_proba(X_test)[:, 1]
except (IndexError, AttributeError):
    print("Warning: Model's predict_proba returned only one class or is unavailable.")
    y_pred_proba = np.zeros(len(y_pred))  # Dummy probabilities

# Evaluate metrics
print("Calculating evaluation metrics...")
try:
    report = classification_report(
        y_test, y_pred,
        target_names=['Legitimate', 'BEC Attempt'],
        labels=[0, 1],
        zero_division=0
    )
except ValueError as e:
    print(f"Error generating classification report: {e}")
    report = "Unable to generate report due to single-class issue."

conf_matrix = confusion_matrix(y_test, y_pred, labels=[0, 1])
roc_auc = roc_auc_score(y_test, y_pred_proba) if len(set(y_test)) > 1 else 0.0

# Save the evaluation report to a markdown file
print("Saving evaluation report...")
with open("evaluation_report.md", "w") as f:
    f.write("# Evaluation Report\n")
    f.write(f"### Model Performance Metrics\n")
    f.write(f"- **ROC AUC Score**: {roc_auc:.2f}\n\n")
    f.write("### Classification Report\n")
    f.write(f"```\n{report}\n```\n")
    f.write("\n### Confusion Matrix\n")
    f.write("![Confusion Matrix](outputs/confusion_matrix.png)\n")
    f.write("\n### AUC-ROC Curve\n")
    f.write("![AUC-ROC Curve](outputs/roc_curve.png)\n")

# Plot and save confusion matrix
print("Plotting confusion matrix...")
class_names = ['Legitimate', 'BEC Attempt']
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('outputs/confusion_matrix.png')
plt.close()

# Plot and save ROC curve
print("Plotting ROC curve...")
if len(set(y_test)) > 1:
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.figure()
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.4f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.savefig('outputs/roc_curve.png')
    plt.close()

print("Evaluation completed!")
