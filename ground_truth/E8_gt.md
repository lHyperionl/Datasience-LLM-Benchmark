```python
import pandas as pd
import numpy as np

# --- 1. Load the dataset ---
# Assume the CSV file 'water_potability.csv' is in the current directory
try:
    df = pd.read_csv('water_potability.csv')
except FileNotFoundError:
    print("Error: 'water_potability.csv' not found. Please ensure the file is in the current directory.")
    exit()

# --- 5. Create a summary showing before cleaning statistics ---
print("--- Data Cleaning Summary ---")
print("\n--- Before Cleaning ---")
print(f"Initial Dataset Shape: {df.shape}")

print("\nMissing Values Before Cleaning:")
# Identify columns with missing values
missing_values_before = df.isnull().sum()
# Display only columns that have missing values
print(missing_values_before[missing_values_before > 0])

# --- 2. Fill missing values using appropriate strategies (mean for numerical columns) ---
# Identify numerical columns that currently have missing values
# We select numerical columns and then filter those that have any NaN values
numerical_cols_with_nan = df.select_dtypes(include=np.number).columns[df.isnull().any()].tolist()

if numerical_cols_with_nan:
    print("\nFilling missing values:")
    for col in numerical_cols_with_nan:
        # Fill missing values with the mean of the column
        df[col].fillna(df[col].mean(), inplace=True)
        print(f"  - Filled missing values in '{col}' with its mean.")
else:
    print("\nNo numerical columns with missing values found to fill.")

# --- 3. Check for and remove any duplicate rows ---
initial_rows_after_imputation = df.shape[0]
# Count duplicate rows across all columns
num_duplicates = df.duplicated().sum()

if num_duplicates > 0:
    df.drop_duplicates(inplace=True)
    print(f"\nRemoved {num_duplicates} duplicate rows.")
else:
    print("\nNo duplicate rows found.")

# --- 4. Verify that all values are within reasonable ranges for water quality parameters ---
# Define reasonable physical ranges for water quality parameters based on common sense and domain knowledge.
# Values outside these ranges are likely measurement errors or physically impossible.
# pH: typically between 0 and 14
# Other parameters (Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic_carbon, Trihalomethanes, Turbidity):
#   These represent concentrations or physical properties that cannot be negative.

initial_rows_before_range_check = df.shape[0]

# Apply range filters to remove physically impossible values
# pH check: values must be between 0 and 14
df = df[(df['ph'] >= 0) & (df['ph'] <= 14)]

# Check for non-negative values for other relevant parameters
positive_params = ['Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity',
                   'Organic_carbon', 'Trihalomethanes', 'Turbidity']
for col in positive_params:
    # Ensure the column exists before applying the filter
    if col in df.columns:
        df = df[df[col] >= 0]

rows_removed_due_to_range = initial_rows_before_range_check - df.shape[0]
if rows_removed_due_to_range > 0:
    print(f"Removed {rows_removed_due_to_range} rows due to out-of-reasonable-range values.")
else:
    print("No rows removed due to out-of-reasonable-range values.")

# --- 5. Create a summary showing after cleaning statistics ---
print("\n--- After Cleaning ---")
print(f"Cleaned Dataset Shape: {df.shape}")

print("\nMissing Values After Cleaning:")
# Verify no missing values remain
missing_values_after = df.isnull().sum()
# Display only columns that still have missing values (should be none if successful)
print(missing_values_after[missing_values_after > 0])
if missing_values_after.sum() == 0:
    print("No missing values remaining.")

# --- 6. Display the first 5 rows of the cleaned dataset ---
print("\nFirst 5 rows of the cleaned dataset:")
print(df.head())
```