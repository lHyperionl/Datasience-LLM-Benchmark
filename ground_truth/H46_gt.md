```python
import pandas as pd
from sklearn.impute import KNNImputer
import numpy as np

# --- 1. Load the dataset using pandas ---
try:
    df = pd.read_csv('penguins.csv')
    print("Dataset loaded successfully.")
    print("Initial NA values per column:")
    print(df.isnull().sum())
    print("-" * 30)
except FileNotFoundError:
    print("Error: 'penguins.csv' not found. Please ensure the file is in the current directory.")
    exit() # Exit the script if the file is not found

# --- 2. K-Nearest Neighbors imputation for specified numerical columns ---

# Define the numerical columns for KNN imputation
numerical_cols_for_knn = ['culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g']

# Initialize KNNImputer with n_neighbors=3 as specified
knn_imputer = KNNImputer(n_neighbors=3)

# Apply KNN imputation to the selected numerical columns
# The imputer expects a 2D array and returns a NumPy array.
# We convert the result back to a DataFrame and assign it to the original columns.
# Ensure that the columns selected for imputation are indeed numerical.
# If any of these columns contain non-numeric data (e.g., strings), KNNImputer will raise an error.
# Assuming standard numerical data types for these columns.
df[numerical_cols_for_knn] = knn_imputer.fit_transform(df[numerical_cols_for_knn])

print("KNN imputation applied to numerical columns.")
print("-" * 30)

# --- 3. Impute NA values in 'sex' using the mode ---

# Define the categorical column for mode imputation
categorical_col_for_mode = 'sex'

# Calculate the mode of the 'sex' column.
# .mode() returns a Series, and [0] selects the first mode in case of multiple modes.
# Check if the mode can be calculated (i.e., if the column is not entirely NA).
if not df[categorical_col_for_mode].mode().empty:
    mode_sex = df[categorical_col_for_mode].mode()[0]
    # Fill NA values in the 'sex' column with its calculated mode
    df[categorical_col_for_mode].fillna(mode_sex, inplace=True)
    print(f"Mode imputation applied to '{categorical_col_for_mode}' column using mode: '{mode_sex}'.")
else:
    print(f"Warning: '{categorical_col_for_mode}' column is entirely NA or empty; mode imputation skipped.")

print("-" * 30)

# --- 4. Print the number of remaining NA values in each column to verify ---

print("Number of remaining NA values in each column after all imputations:")
print(df.isnull().sum())

# Optional: Verify that the target columns have no NAs
print("\nVerification for specific imputed columns:")
print(f"NA in culmen_length_mm: {df['culmen_length_mm'].isnull().sum()}")
print(f"NA in culmen_depth_mm: {df['culmen_depth_mm'].isnull().sum()}")
print(f"NA in flipper_length_mm: {df['flipper_length_mm'].isnull().sum()}")
print(f"NA in body_mass_g: {df['body_mass_g'].isnull().sum()}")
print(f"NA in sex: {df['sex'].isnull().sum()}")
```