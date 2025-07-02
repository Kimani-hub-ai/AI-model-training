# Task 1: Iris Species Classification using Decision Tree Classifier

# Step 1: Import necessary libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

# Step 2: Load the Iris dataset
iris = load_iris()
X = iris.data  # Features (sepal and petal measurements)
y = iris.target  # Labels (species: 0=setosa, 1=versicolor, 2=virginica)

# Step 3: Simulate missing data (for learning purposes only)
# Introduce a few NaN values randomly
rng = np.random.default_rng(seed=42)
missing_indices = rng.choice(X.size, size=10, replace=False)  # Random 10 values
X.ravel()[missing_indices] = np.nan

# Step 4: Handle missing values using mean imputation
imputer = SimpleImputer(strategy="mean")
X_imputed = imputer.fit_transform(X)

# Step 5: Encode target labels (already numeric, but here's how it would work if they were strings)
# In this case, we skip label encoding because y is already encoded

# Step 6: Split dataset into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# Step 7: Initialize and train a Decision Tree Classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Step 8: Make predictions on test data
y_pred = clf.predict(X_test)

# Step 9: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred, average='weighted')

# Step 10: Output metrics
print("=== Model Evaluation ===")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print("\nDetailed Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=iris.target_names))
