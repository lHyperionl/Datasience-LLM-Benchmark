```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# --- 1. Load the dataset ---
# Assume 'iris.csv' is in the current working directory.
try:
    df = pd.read_csv('iris.csv')
except FileNotFoundError:
    print("Error: 'iris.csv' not found. Please ensure the file is in the current directory.")
    exit()

# --- 2. Define features X ---
# Select the specified feature columns.
feature_columns = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
X = df[feature_columns]

# --- 3. Define the target y ---
# Select the 'Species' column as the target.
y = df['Species']

# --- 4. Encode the categorical Species target variable into numerical labels ---
# Initialize LabelEncoder.
label_encoder = LabelEncoder()
# Fit and transform the 'Species' column.
y_encoded = label_encoder.fit_transform(y)

# --- 5. Split the data (X and encoded y) into training (70%) and testing (30%) sets ---
# Use random_state=42 for reproducibility.
# stratify=y_encoded ensures that the proportion of target classes is the same in both train and test sets.
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
)

# --- 6. Train a K-Nearest Neighbors classifier with n_neighbors=3 on the training data ---
# Initialize the KNN classifier.
knn_classifier = KNeighborsClassifier(n_neighbors=3)
# Train the classifier on the training data.
knn_classifier.fit(X_train, y_train)

# --- 7. Make predictions on the test set and report the accuracy score ---
# Make predictions on the test features.
y_pred = knn_classifier.predict(X_test)

# Calculate the accuracy score.
accuracy = accuracy_score(y_test, y_pred)

# Print the accuracy.
print(f"K-Nearest Neighbors Classifier Accuracy: {accuracy:.4f}")

```