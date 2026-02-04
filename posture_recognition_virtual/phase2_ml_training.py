import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle

# Load dataset
X = np.load("data/processed/X.npy")
y = np.load("data/processed/y.npy")

print("Loaded feature matrix:", X.shape)
print("Loaded labels:", y.shape)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

# Train Decision Tree
dt_model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=None,
    random_state=42
)

dt_model.fit(X_train, y_train)
print("✅ Decision Tree training completed")

# Prediction
y_pred = dt_model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save model
os.makedirs("models/trained", exist_ok=True)

with open("models/trained/dt_model.pkl", "wb") as f:
    pickle.dump(dt_model, f)

print("\n✅ Trained model saved as models/trained/dt_model.pkl")

