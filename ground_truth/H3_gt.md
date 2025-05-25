```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report

# --- 1. Load the dataset ---
try:
    df = pd.read_csv('winequality-red.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: 'winequality-red.csv' not found. Please ensure the file is in the current directory.")
    exit() # Exit if the dataset is not found

# --- 2. Create binary target variable ---
# 'good' wines have quality >= 7 (assigned 1), 'bad' wines have quality < 7 (assigned 0)
df['is_good_wine'] = (df['quality'] >= 7).astype(int)
print(f"\nBinary target variable 'is_good_wine' created. Value counts:\n{df['is_good_wine'].value_counts()}")

# Separate features (X) and target (y)
# Drop the original 'quality' column as it's now replaced by 'is_good_wine'
X = df.drop(['quality', 'is_good_wine'], axis=1)
y = df['is_good_wine']

# --- 3. Preprocess the data ---

# Handle missing values (if any)
# Check for missing values
if X.isnull().sum().sum() > 0:
    print("\nMissing values detected. Imputing with median...")
    # A simple strategy: fill missing values with the median of each column
    # For this specific dataset, there are typically no missing values, but this is good practice.
    for col in X.columns:
        if X[col].isnull().any():
            X[col].fillna(X[col].median(), inplace=True)
    print("Missing values handled.")
else:
    print("\nNo missing values found in the dataset.")

# Scale features appropriately
# Initialize StandardScaler
scaler = StandardScaler()

# --- 4. Split data into 80/20 train/test sets ---
# Use stratify=y to ensure the same proportion of target classes in both train and test sets
# Use random_state for reproducibility
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nData split into training and testing sets:")
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

# Fit the scaler on the training data and transform both training and test data
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("\nFeatures scaled using StandardScaler.")

# Convert scaled arrays back to DataFrame for easier feature importance interpretation later
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X.columns, index=X_train.index)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X.columns, index=X_test.index)


# --- 5. Train a logistic regression model ---
# Initialize Logistic Regression model
# solver='liblinear' is good for small datasets and handles L1/L2 regularization
# max_iter increased to ensure convergence
model = LogisticRegression(random_state=42, solver='liblinear', max_iter=1000)

# Train the model
model.fit(X_train_scaled_df, y_train)
print("\nLogistic Regression model trained successfully.")

# --- 6. Evaluate performance ---
# Make predictions on the scaled test set
y_pred = model.predict(X_test_scaled_df)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("\n--- Model Evaluation on Test Set ---")
print(f"Accuracy: {accuracy:.4f}")
print(f"F1-Score: {f1:.4f}")
print("\nClassification Report:\n", report)

# --- 7. Report the most important features according to the model coefficients ---
# For Logistic Regression, coefficients indicate feature importance.
# Higher absolute coefficient values mean higher importance.
coefficients = model.coef_[0]
feature_names = X.columns

# Create a DataFrame to store feature names and their coefficients
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': coefficients
})

# Calculate absolute coefficients for ranking importance
feature_importance_df['Abs_Coefficient'] = np.abs(feature_importance_df['Coefficient'])

# Sort features by absolute coefficient in descending order
feature_importance_df = feature_importance_df.sort_values(by='Abs_Coefficient', ascending=False)

print("\n--- Most Important Features (by absolute coefficient) ---")
print(feature_importance_df[['Feature', 'Coefficient', 'Abs_Coefficient']].round(4))

# Optional: Print interpretation of top features
print("\nInterpretation of coefficients:")
print("A positive coefficient indicates that as the feature value increases, the likelihood of the wine being 'good' (quality >= 7) increases.")
print("A negative coefficient indicates that as the feature value increases, the likelihood of the wine being 'good' (quality >= 7) decreases.")
print("The magnitude of the coefficient indicates the strength of the relationship.")
```