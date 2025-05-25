```python
import pandas as pd
import numpy as np

# --- 1. Load the dataset ---
# Assume 'penguins.csv' is in the current directory.
try:
    df = pd.read_csv('penguins.csv')
    print("Dataset loaded successfully. Original shape:", df.shape)
except FileNotFoundError:
    print("Error: 'penguins.csv' not found. Please ensure the file is in the current directory.")
    exit() # Exit if the dataset cannot be loaded

# --- 2. Select features and target variable ---
# Define the features and the target variable as specified.
features_to_select = [
    'culmen_length_mm',
    'culmen_depth_mm',
    'flipper_length_mm',
    'body_mass_g',
    'island',
    'sex'
]
target_variable = 'species'

# Create a copy of the DataFrame with only the selected columns to avoid SettingWithCopyWarning.
df_processed = df[features_to_select + [target_variable]].copy()

print("\nSelected features and target. Initial processed shape:", df_processed.shape)

# --- 3. Impute NA values for numerical features with the mean ---
numerical_cols = [
    'culmen_length_mm',
    'culmen_depth_mm',
    'flipper_length_mm',
    'body_mass_g'
]

print("\nImputing numerical NA values with column means...")
for col in numerical_cols:
    if df_processed[col].isnull().any():
        mean_val = df_processed[col].mean()
        df_processed[col].fillna(mean_val, inplace=True)
        print(f"  - Imputed NA values in '{col}' with mean: {mean_val:.2f}")
    else:
        print(f"  - No NA values found in '{col}'.")

# --- 4. Impute and encode the 'sex' column ---
# Impute NA values with the string 'Unknown'.
print("\nProcessing 'sex' column...")
if df_processed['sex'].isnull().any():
    df_processed['sex'].fillna('Unknown', inplace=True)
    print("  - Imputed NA values in 'sex' with 'Unknown'.")
else:
    print("  - No NA values found in 'sex'.")

# Map 'MALE' to 0, 'FEMALE' to 1, and 'Unknown' to -1.
sex_mapping = {'MALE': 0, 'FEMALE': 1, 'Unknown': -1}
# Ensure all values in 'sex' are strings before mapping to avoid errors with mixed types.
df_processed['sex'] = df_processed['sex'].astype(str).map(sex_mapping)

# Check if any values were not mapped (e.g., if there were other unexpected strings)
# and handle them if necessary (e.g., by mapping them to -1 or dropping rows).
# For this task, we assume only 'MALE', 'FEMALE', 'Unknown' are present after imputation.
if df_processed['sex'].isnull().any():
    print("  - Warning: Some 'sex' values could not be mapped and are now NaN. Mapping them to -1.")
    df_processed['sex'].fillna(-1, inplace=True) # Fallback for any unmapped values

print("  - Mapped 'sex' column: 'MALE'->0, 'FEMALE'->1, 'Unknown'->-1.")

# --- 5. Apply one-hot encoding to the 'island' column ---
print("\nApplying one-hot encoding to 'island' column...")
# pd.get_dummies automatically handles string columns and creates new columns.
# The original 'island' column will be dropped.
df_processed = pd.get_dummies(df_processed, columns=['island'], prefix='island')
print("  - 'island' column one-hot encoded.")

# --- Separate features (X) and target (y) ---
X = df_processed.drop(columns=[target_variable])
y = df_processed[target_variable]

# --- 6. Print the shape of the resulting feature matrix and the first 5 rows ---
print("\n--- Processing Complete ---")
print("Shape of the resulting feature matrix (X):", X.shape)
print("\nFirst 5 rows of the processed features (X):")
print(X.head())

# Optional: Print value counts for the target variable to confirm it's ready
print("\nTarget variable (species) value counts:")
print(y.value_counts())
```