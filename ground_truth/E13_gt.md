```python
import pandas as pd
import numpy as np

# --- Configuration ---
DATASET_PATH = 'diabetes.csv'
TARGET_COLUMN = 'Outcome' # Assuming 'Outcome' is the target column for diabetes status

# Columns where zero values might represent missing data (common in this dataset)
# Pregnancies and Outcome can legitimately be zero.
COLUMNS_TO_CHECK_FOR_ZERO_AS_MISSING = [
    'Glucose',
    'BloodPressure',
    'SkinThickness',
    'Insulin',
    'BMI'
]

# --- 1. Load the dataset and display the first 10 rows ---
try:
    df = pd.read_csv(DATASET_PATH)
    print("--- 1. First 10 rows of the dataset ---")
    print(df.head(10))
    print("\n" + "="*50 + "\n")
except FileNotFoundError:
    print(f"Error: The file '{DATASET_PATH}' was not found. Please ensure it's in the current directory.")
    exit()
except Exception as e:
    print(f"An error occurred while loading the dataset: {e}")
    exit()

# --- 2. Show the shape of the dataset (rows and columns) ---
print("--- 2. Shape of the dataset (rows, columns) ---")
print(df.shape)
print("\n" + "="*50 + "\n")

# --- 3. List all column names and their data types ---
print("--- 3. Column names and their data types ---")
print(df.info()) # df.info() provides column names, non-null counts, and dtypes
print("\n" + "="*50 + "\n")

# --- 4. Calculate the percentage of diabetic vs non-diabetic patients (Outcome column) ---
if TARGET_COLUMN in df.columns:
    print(f"--- 4. Percentage of patients by '{TARGET_COLUMN}' status ---")
    outcome_counts = df[TARGET_COLUMN].value_counts()
    outcome_percentages = df[TARGET_COLUMN].value_counts(normalize=True) * 100

    print("Counts:")
    print(outcome_counts)
    print("\nPercentages:")
    print(outcome_percentages.round(2)) # Round to 2 decimal places for readability
    print("\n" + "="*50 + "\n")
else:
    print(f"--- 4. '{TARGET_COLUMN}' column not found. Cannot calculate percentages. ---")
    print("\n" + "="*50 + "\n")


# --- 5. Display the count of missing values for each column ---
print("--- 5. Count of missing values for each column ---")
missing_values = df.isnull().sum()
print(missing_values[missing_values > 0] if missing_values.sum() > 0 else "No missing values found.")
print("\n" + "="*50 + "\n")

# --- 6. Show basic summary statistics for all numerical columns ---
print("--- 6. Basic summary statistics for numerical columns ---")
print(df.describe())
print("\n" + "="*50 + "\n")

# --- 7. Identify any columns with zero values that might represent missing data ---
print("--- 7. Count of zero values in specific columns (potential missing data) ---")
zero_counts = {}
for col in COLUMNS_TO_CHECK_FOR_ZERO_AS_MISSING:
    if col in df.columns:
        zero_count = (df[col] == 0).sum()
        if zero_count > 0:
            zero_counts[col] = zero_count
    else:
        print(f"Warning: Column '{col}' not found in the dataset.")

if zero_counts:
    for col, count in zero_counts.items():
        print(f"Column '{col}': {count} zero values ({np.round(count / len(df) * 100, 2)}%)")
else:
    print("No suspicious zero values found in the specified columns.")
print("\n" + "="*50 + "\n")
```