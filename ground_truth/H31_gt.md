```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix

# --- 1. Load data ---
try:
    df = pd.read_csv('spotify_2023.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: 'spotify_2023.csv' not found. Please ensure the CSV file is in the current directory.")
    exit() # Exit if the file is not found

# --- 2. Convert `streams` to numeric (coerce errors to NaN). Drop rows where `streams` is NaN. ---
# Convert 'streams' column to numeric, coercing non-numeric values to NaN
df['streams'] = pd.to_numeric(df['streams'], errors='coerce')

# Drop rows where 'streams' is NaN (after conversion)
initial_rows = df.shape[0]
df.dropna(subset=['streams'], inplace=True)
rows_after_streams_clean = df.shape[0]
print(f"Dropped {initial_rows - rows_after_streams_clean} rows with non-numeric or missing 'streams' values.")

# --- 3. Create a binary target variable `is_popular` ---
# Calculate the 75th percentile of the 'streams' column
streams_75th_percentile = df['streams'].quantile(0.75)
print(f"75th percentile of 'streams': {streams_75th_percentile:,.0f}")

# Create the 'is_popular' target variable
# 1 if streams are above the 75th percentile, 0 otherwise
df['is_popular'] = (df['streams'] > streams_75th_percentile).astype(int)
print(f"Target variable 'is_popular' created. Class distribution:\n{df['is_popular'].value_counts()}")

# --- 4. Select features ---
# Define the numerical features to be used
numerical_features = ['danceability_%', 'energy_%', 'valence_%', 'acousticness_%', 'bpm', 'artist_count']
# Define the categorical features to be one-hot encoded
categorical_features = ['key', 'mode']

# Combine all selected features
all_features = numerical_features + categorical_features

# Ensure all selected features exist in the DataFrame
missing_features = [col for col in all_features if col not in df.columns]
if missing_features:
    print(f"Error: The following specified features are missing from the dataset: {missing_features}")
    exit()

# Prepare the feature matrix X and target vector y
X = df[all_features].copy() # Use .copy() to avoid SettingWithCopyWarning
y = df['is_popular']

# --- 5. One-hot encode categorical features `key` and `mode`. Combine these with other selected features. ---
# Perform one-hot encoding on the specified categorical features
X = pd.get_dummies(X, columns=categorical_features, drop_first=False)
print(f"Features after one-hot encoding: {X.shape[1]} columns.")

# --- 6. Split data into training (80%) and testing (20%) sets (`random_state=42`). ---
# Split the data, stratifying by 'is_popular' to maintain class distribution in both sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nData split into training and testing sets:")
print(f"Training set shape: X_train={X_train.shape}, y_train={y_train.shape}")
print(f"Testing set shape: X_test={X_test.shape}, y_test={y_test.shape}")

# --- 7. Train a Logistic Regression model (default parameters, `max_iter=1000`). ---
# Initialize the Logistic Regression model
# Set max_iter to 1000 as required, and random_state for reproducibility
model = LogisticRegression(max_iter=1000, random_state=42)

# Train the model using the training data
print("\nTraining Logistic Regression model...")
model.fit(X_train, y_train)
print("Model training complete.")

# --- 8. Evaluate using F1-score (macro), ROC AUC score. Display the confusion matrix for the test set. ---
# Make predictions on the test set
y_pred = model.predict(X_test)
# Get prediction probabilities for ROC AUC score (probability of the positive class)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Calculate F1-score (macro average)
f1 = f1_score(y_test, y_pred, average='macro')
print(f"\nEvaluation Metrics:")
print(f"F1-score (macro): {f1:.4f}")

# Calculate ROC AUC score
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"ROC AUC Score: {roc_auc:.4f}")

# Display the confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix for the Test Set:")
print(cm)
print("\nInterpretation of Confusion Matrix:")
print("Rows represent actual classes (0: Not Popular, 1: Popular)")
print("Columns represent predicted classes (0: Not Popular, 1: Popular)")
print(f"[[True Negatives  False Positives]")
print(f" [False Negatives True Positives]]")
```