```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# --- 1. Load the dataset and handle missing values ---

# Load the dataset
try:
    df = pd.read_csv('video_games_sales.csv')
except FileNotFoundError:
    print("Error: 'video_games_sales.csv' not found. Please ensure the file is in the current directory.")
    exit()

# Display initial info and missing values
# print("Initial DataFrame Info:")
# df.info()
# print("\nMissing values before handling:")
# print(df.isnull().sum())

# Handle missing values
# Drop rows where 'Global_Sales' is NaN as it's crucial for the target variable
df.dropna(subset=['Global_Sales'], inplace=True)

# For 'Year', 'Publisher', and regional sales, dropping NaNs is a reasonable approach
# as these are critical for feature creation and model training.
# If there were many NaNs, imputation strategies (e.g., mode for categorical, median/mean for numerical)
# would be considered, but for this task, dropping is simpler and robust.
df.dropna(subset=['Year', 'Publisher', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales'], inplace=True)

# Convert 'Year' to integer type after dropping NaNs
df['Year'] = df['Year'].astype(int)

# print("\nMissing values after handling:")
# print(df.isnull().sum())
# print("\nDataFrame Info after handling missing values:")
# df.info()

# --- 2. Create a binary target variable 'High_Sales' ---

# Define 'High_Sales': 1 if Global_Sales > 1.0 million, else 0
df['High_Sales'] = (df['Global_Sales'] > 1.0).astype(int)

# print("\n'High_Sales' distribution:")
# print(df['High_Sales'].value_counts())

# --- 3. Prepare features by encoding categorical variables ---

# Identify categorical columns for encoding
categorical_cols = ['Platform', 'Genre', 'Publisher']

# Initialize LabelEncoder
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le # Store encoder for potential inverse_transform if needed

# print("\nDataFrame head after label encoding:")
# print(df[categorical_cols].head())

# --- 4. Create additional features ---

# Create 'Total_Regional_Sales'
df['Total_Regional_Sales'] = df['NA_Sales'] + df['EU_Sales'] + df['JP_Sales'] + df['Other_Sales']

# Create 'Years_Since_Release'
# Assuming current year is 2023 as per prompt
CURRENT_YEAR = 2023
df['Years_Since_Release'] = CURRENT_YEAR - df['Year']

# Handle potential negative or zero years since release if CURRENT_YEAR is less than game's release year
# (though unlikely with typical datasets, good practice to consider)
df['Years_Since_Release'] = df['Years_Since_Release'].apply(lambda x: max(0, x))

# print("\nDataFrame head with new features:")
# print(df[['Total_Regional_Sales', 'Years_Since_Release']].head())

# --- Select features and target ---
# Features to be used for training
features = [
    'Platform', 'Genre', 'Publisher', 'NA_Sales', 'EU_Sales',
    'JP_Sales', 'Other_Sales', 'Total_Regional_Sales', 'Years_Since_Release'
]
target = 'High_Sales'

X = df[features]
y = df[target]

# print("\nFeatures (X) shape:", X.shape)
# print("Target (y) shape:", y.shape)

# --- 5. Split the data into training and testing sets (80-20 split) ---

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# print("\nTraining set shape (X_train, y_train):", X_train.shape, y_train.shape)
# print("Testing set shape (X_test, y_test):", X_test.shape, y_test.shape)

# --- 6. Train a Random Forest classifier with 100 estimators ---

# Initialize the Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1) # n_jobs=-1 uses all available cores

# Train the model
print("\nTraining Random Forest Classifier...")
rf_classifier.fit(X_train, y_train)
print("Training complete.")

# --- 7. Evaluate the model ---

# Make predictions on the test set
y_pred = rf_classifier.predict(X_test)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("\n--- Model Evaluation ---")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print("\nConfusion Matrix:")
print(conf_matrix)
print("  (True Negatives, False Positives)")
print("  (False Negatives, True Positives)")

# --- 8. Display feature importance ---

# Get feature importances from the trained model
feature_importances = rf_classifier.feature_importances_

# Create a pandas Series for better visualization
importance_df = pd.Series(feature_importances, index=features)

# Sort feature importances in descending order
sorted_importance = importance_df.sort_values(ascending=False)

print("\n--- Feature Importance ---")
print(sorted_importance)
```