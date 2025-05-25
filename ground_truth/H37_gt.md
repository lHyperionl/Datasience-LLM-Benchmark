```python
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score

# --- 1. Load Data ---
# Assume 'mobiles_2025.csv' is in the current directory
try:
    df = pd.read_csv('mobiles_2025.csv')
    print("Dataset loaded successfully.")
    print(f"Initial dataset shape: {df.shape}")
except FileNotFoundError:
    print("Error: 'mobiles_2025.csv' not found. Please ensure the CSV file is in the same directory.")
    exit() # Exit if the dataset cannot be loaded

# Create a copy to avoid SettingWithCopyWarning
df_cleaned = df.copy()

# --- 2. Clean `Launched Price (USA)` and Create Binary Target `is_expensive_usa` ---
print("\n--- Cleaning 'Launched Price (USA)' and creating target variable ---")
# Remove 'USD ' and commas, then convert to numeric
df_cleaned['Launched Price (USA)_cleaned'] = df_cleaned['Launched Price (USA)'].astype(str).str.replace('USD ', '', regex=False).str.replace(',', '', regex=False)
df_cleaned['Launched Price (USA)_cleaned'] = pd.to_numeric(df_cleaned['Launched Price (USA)_cleaned'], errors='coerce')

# Create binary target `is_expensive_usa`
# 1 if cleaned `Launched Price (USA)` > 1000, else 0
df_cleaned['is_expensive_usa'] = df_cleaned['Launched Price (USA)_cleaned'].apply(lambda x: 1 if x > 1000 else (0 if pd.notna(x) else np.nan))

# Drop rows where `is_expensive_usa` is NaN (i.e., original price was unparseable)
initial_rows = df_cleaned.shape[0]
df_cleaned.dropna(subset=['is_expensive_usa'], inplace=True)
df_cleaned['is_expensive_usa'] = df_cleaned['is_expensive_usa'].astype(int) # Convert to integer type
print(f"Dropped {initial_rows - df_cleaned.shape[0]} rows due to unparseable 'Launched Price (USA)'.")
print(f"Current dataset shape after target cleaning: {df_cleaned.shape}")

# --- 3. Clean Features ---
print("\n--- Cleaning specified features ---")

# Mobile Weight: remove 'g', to numeric
df_cleaned['Mobile Weight_cleaned'] = df_cleaned['Mobile Weight'].astype(str).str.replace('g', '', regex=False).str.strip()
df_cleaned['Mobile Weight_cleaned'] = pd.to_numeric(df_cleaned['Mobile Weight_cleaned'], errors='coerce')
print("Cleaned 'Mobile Weight'.")

# RAM: remove 'GB', to numeric
df_cleaned['RAM_cleaned'] = df_cleaned['RAM'].astype(str).str.replace('GB', '', regex=False).str.strip()
df_cleaned['RAM_cleaned'] = pd.to_numeric(df_cleaned['RAM_cleaned'], errors='coerce')
print("Cleaned 'RAM'.")

# Battery Capacity: remove 'mAh', to numeric
df_cleaned['Battery Capacity_cleaned'] = df_cleaned['Battery Capacity'].astype(str).str.replace('mAh', '', regex=False).str.strip()
df_cleaned['Battery Capacity_cleaned'] = pd.to_numeric(df_cleaned['Battery Capacity_cleaned'], errors='coerce')
print("Cleaned 'Battery Capacity'.")

# Screen Size: remove ' inches', to numeric
df_cleaned['Screen Size_cleaned'] = df_cleaned['Screen Size'].astype(str).str.replace(' inches', '', regex=False).str.strip()
df_cleaned['Screen Size_cleaned'] = pd.to_numeric(df_cleaned['Screen Size_cleaned'], errors='coerce')
print("Cleaned 'Screen Size'.")

# Front Camera and Back Camera: extract first numerical MP value (default 0 if none)
def extract_mp(camera_str):
    if pd.isna(camera_str):
        return 0.0
    match = re.search(r'(\d+(\.\d+)?)\s*MP', str(camera_str), re.IGNORECASE)
    if match:
        return float(match.group(1))
    return 0.0 # Default to 0 if no MP value found

df_cleaned['Front Camera_cleaned'] = df_cleaned['Front Camera'].apply(extract_mp)
df_cleaned['Back Camera_cleaned'] = df_cleaned['Back Camera'].apply(extract_mp)
print("Cleaned 'Front Camera' and 'Back Camera'.")

# --- 4. Handle Categorical Features: One-hot encode `Company Name` and `Processor` ---
print("\n--- One-hot encoding 'Company Name' and 'Processor' ---")
# Ensure these columns exist before encoding
categorical_cols = ['Company Name', 'Processor']
for col in categorical_cols:
    if col not in df_cleaned.columns:
        print(f"Warning: Column '{col}' not found for one-hot encoding. Skipping.")
        categorical_cols.remove(col)

df_encoded = pd.get_dummies(df_cleaned, columns=categorical_cols, drop_first=False, dummy_na=False)
print(f"Dataset shape after one-hot encoding: {df_encoded.shape}")

# --- 5. Define features `X` and target `y` ---
print("\n--- Defining features (X) and target (y) ---")

# List of cleaned numerical features
numerical_features = [
    'Mobile Weight_cleaned',
    'RAM_cleaned',
    'Front Camera_cleaned',
    'Back Camera_cleaned',
    'Battery Capacity_cleaned',
    'Screen Size_cleaned',
    'Launched Year' # 'Launched Year' is already numeric
]

# Get the names of the one-hot encoded columns
encoded_feature_cols = [col for col in df_encoded.columns if col.startswith('Company Name_') or col.startswith('Processor_')]

# Combine all feature columns
all_features = numerical_features + encoded_feature_cols

# Define X and y
X = df_encoded[all_features]
y = df_encoded['is_expensive_usa']

# Drop rows with any NaNs in features or target after all transformations
initial_rows_xy = X.shape[0]
# Combine X and y to drop NaNs consistently across both
combined_df = pd.concat([X, y], axis=1)
combined_df.dropna(inplace=True)

X = combined_df[X.columns]
y = combined_df['is_expensive_usa']

print(f"Dropped {initial_rows_xy - X.shape[0]} rows due to NaNs in features or target after final cleaning.")
print(f"Final X shape: {X.shape}, y shape: {y.shape}")

# Check if there's enough data left
if X.shape[0] == 0:
    print("Error: No data remaining after NaN removal. Cannot proceed with model training.")
    exit()

# --- 6. Split data (80% train, 20% test, random_state=42) ---
print("\n--- Splitting data into training and testing sets ---")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

# --- 7. Train a RandomForestClassifier ---
print("\n--- Training RandomForestClassifier ---")
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1) # n_jobs=-1 uses all available cores
model.fit(X_train, y_train)
print("RandomForestClassifier trained successfully.")

# --- 8. Evaluate using F1-score (macro average) and ROC AUC score ---
print("\n--- Evaluating model performance ---")

# Make predictions on the test set
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1] # Probability of the positive class (1)

# Calculate F1-score (macro average)
f1_macro = f1_score(y_test, y_pred, average='macro')
print(f"F1-score (macro average): {f1_macro:.4f}")

# Calculate ROC AUC score
# Check if there's only one class in y_test, which would make ROC AUC undefined
if len(np.unique(y_test)) < 2:
    print("ROC AUC score cannot be calculated: Test set contains only one class.")
else:
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"ROC AUC Score: {roc_auc:.4f}")

print("\n--- Classification task completed ---")
```