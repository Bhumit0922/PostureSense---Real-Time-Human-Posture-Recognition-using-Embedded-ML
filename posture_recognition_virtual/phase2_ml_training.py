import numpy as np
import os
import pickle
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ==============================
# 1️⃣ Load Dataset
# ==============================

X = np.load("data/processed/X.npy")
y = np.load("data/processed/y.npy")

print("Loaded feature matrix:", X.shape)
print("Loaded labels:", y.shape)

# ==============================
# 2️⃣ Train-Test Split
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

# ==============================
# 3️⃣ Feature Scaling
# ==============================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==============================
# 4️⃣ Hyperparameter Tuning (Optional but Recommended)
# ==============================

print("\n🔍 Performing Hyperparameter Tuning...")

param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 10, 20],
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

print("✅ Best Parameters Found:", grid_search.best_params_)

# ==============================
# 5️⃣ Cross Validation Score
# ==============================

cv_scores = cross_val_score(best_model, X_train, y_train, cv=5)

print("\nCross Validation Accuracy: {:.2f}%".format(cv_scores.mean() * 100))

# ==============================
# 6️⃣ Final Training
# ==============================

best_model.fit(X_train, y_train)
print("✅ Random Forest training completed")

# ==============================
# 7️⃣ Prediction
# ==============================

y_pred = best_model.predict(X_test)

# ==============================
# 8️⃣ Evaluation
# ==============================

accuracy = accuracy_score(y_test, y_pred)

print("\n==============================")
print("🎯 FINAL TEST RESULTS")
print("==============================")
print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ==============================
# 9️⃣ Feature Importance
# ==============================

print("\n📊 Feature Importance Analysis")

importances = best_model.feature_importances_

plt.figure()
plt.title("Feature Importance")
plt.bar(range(len(importances)), importances)
plt.xlabel("Feature Index")
plt.ylabel("Importance")
plt.show()

# ==============================
# 🔟 Save Model & Scaler
# ==============================

os.makedirs("models/trained", exist_ok=True)

with open("models/trained/rf_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

with open("models/trained/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("\n✅ Trained model saved as models/trained/rf_model.pkl")
print("✅ Scaler saved as models/trained/scaler.pkl")
