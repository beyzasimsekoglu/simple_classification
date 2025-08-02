# src/iris_classification.py

# 1. Import necessary libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

import pandas as pd

# 2. Load the Iris dataset
iris = load_iris()
X = iris.data  # Features (measurements)
y = iris.target  # Labels (flower types)

# 3. Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Create a K-Nearest Neighbors classifier
knn = KNeighborsClassifier(n_neighbors=3)

# 5. Train the classifier
knn.fit(X_train, y_train)

# 6. Make predictions on the test set
y_pred = knn.predict(X_test)

# 7. Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# 8. (Optional) Show some predictions
print("First 5 predictions:", y_pred[:5])
print("First 5 true labels:", y_test[:5])




