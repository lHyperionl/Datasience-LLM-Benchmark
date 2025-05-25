```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# --- 1. Load the dataset ---
# Assume 'diabetes.csv' is in the current directory.
try:
    df = pd.read_csv('diabetes.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: 'diabetes.csv' not found. Please ensure the file is in the current directory.")
    exit()

# Store original shape for comparison
original_shape = df.shape

# --- 2. Identify columns with zero values that represent missing data ---
# As per the problem description, these columns should not have zero values.
cols_to_replace_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

# --- 3. Replace zero values in these columns with NaN ---
# This makes it easier to handle them as missing data.
for col in cols_to_replace_zero:
    if col in df.columns:
        df[col] = df[col].replace(0, np.nan)
    else:
        print(f"Warning: Column '{col}' not found in the dataset. Skipping zero replacement for this column.")

# --- 4. Handle missing values by filling them with the median of each respective column ---
# Median is robust to outliers compared to mean.
for col in cols_to_replace_zero:
    if col in df.columns:
        # Calculate median only from non-NaN values
        median_val = df[col].median()
        df[col].fillna(median_val, inplace=True)
        # print(f"Filled missing values in '{col}' with median: {median_val}")
    # No else needed here, as the warning was already printed in step 3.

# --- 5. Remove any duplicate rows if they exist ---
initial_rows = df.shape[0]
df.drop_duplicates(inplace=True)
rows_after_duplicates = df.shape[0]
if initial_rows > rows_after_duplicates:
    print(f"Removed {initial_rows - rows_after_duplicates} duplicate rows.")
else:
    print("No duplicate rows found.")

# --- 6. Create a new column 'BMI_Category' ---
# Categorizes BMI into 'Underweight', 'Normal', 'Overweight', and 'Obese'.
# Ensure 'BMI' column exists before creating the category.
if 'BMI' in df.columns:
    bins = [0, 18.5, 24.9, 29.9, np.inf] # np.inf handles values >= 30
    labels = ['Underweight', 'Normal', 'Overweight', 'Obese']
    df['BMI_Category'] = pd.cut(df['BMI'], bins=bins, labels=labels, right=True)
    print("Created 'BMI_Category' column.")
else:
    print("Warning: 'BMI' column not found. Skipping 'BMI_Category' creation.")

# --- 7. Scale numerical features using StandardScaler ---
# Identify numerical columns to scale. 'Outcome' is typically the target variable and not scaled.
# 'BMI_Category' is a categorical column and should not be scaled.
numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
if 'Outcome' in numerical_cols:
    numerical_cols.remove('Outcome') # Exclude target variable from scaling

# Initialize StandardScaler
scaler = StandardScaler()

# Apply scaling to the identified numerical columns
if numerical_cols:
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    print(f"Scaled numerical features: {numerical_cols}")
else:
    print("No numerical columns found for scaling (excluding 'Outcome').")

# --- 8. Display the cleaned dataset's shape and first 5 rows ---
print("\n--- Cleaned Dataset Information ---")
print(f"Original dataset shape: {original_shape}")
print(f"Cleaned dataset shape: {df.shape}")
print("\nFirst 5 rows of the cleaned dataset:")
print(df.head())

```