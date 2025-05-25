```python
import pandas as pd

# Define the dataset file name
FILE_NAME = 'heart.csv'

# --- Step 1: Load the dataset ---
# Load the CSV file into a pandas DataFrame.
# Includes error handling for FileNotFoundError if the file is not present.
try:
    df = pd.read_csv(FILE_NAME)
    print(f"Successfully loaded '{FILE_NAME}' dataset.\n")
except FileNotFoundError:
    print(f"Error: The file '{FILE_NAME}' was not found. Please ensure it is in the current directory.")
    # Exit the script as further operations depend on the DataFrame
    exit()
except Exception as e:
    print(f"An error occurred while loading the dataset: {e}")
    exit()

# --- Step 2: List all column names and identify the data type of each column ---
print("--- Column Names and Data Types ---")
print("Column Names:")
# Iterate through column names and print each one
for col in df.columns:
    print(f"- {col}")

print("\nData Types:")
# Print the data types of all columns
print(df.dtypes)
print("-" * 40 + "\n")

# --- Step 3: Calculate and report basic descriptive statistics for specified numerical columns ---
numerical_cols_for_stats = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR']
print(f"--- Descriptive Statistics for Numerical Columns: {', '.join(numerical_cols_for_stats)} ---")

# Check if all specified numerical columns exist in the DataFrame
existing_numerical_cols = [col for col in numerical_cols_for_stats if col in df.columns]
missing_numerical_cols = [col for col in numerical_cols_for_stats if col not in df.columns]

if missing_numerical_cols:
    print(f"Warning: The following specified numerical columns were not found in the dataset: {', '.join(missing_numerical_cols)}")
    print("Descriptive statistics will be calculated only for existing columns.\n")

if existing_numerical_cols:
    # Calculate descriptive statistics (mean, median, std, min, max) using .describe()
    # .describe() provides count, mean, std, min, 25%, 50% (median), 75%, max
    # We select the relevant statistics for reporting.
    stats = df[existing_numerical_cols].describe().loc[['mean', '50%', 'std', 'min', 'max']]
    # Rename '50%' to 'median' for clarity
    stats = stats.rename(index={'50%': 'median'})
    print(stats)
else:
    print("No valid numerical columns found to calculate descriptive statistics.")
print("-" * 40 + "\n")

# --- Step 4: Report the count of occurrences for each value in the 'HeartDisease' column ---
heart_disease_column_name = 'HeartDisease'
print(f"--- Value Counts for '{heart_disease_column_name}' Column ---")

# Check if the 'HeartDisease' column exists in the DataFrame
if heart_disease_column_name in df.columns:
    print("Interpretation: 0 = No Heart Disease, 1 = Heart Disease")
    # Use .value_counts() to get the count of occurrences for each unique value
    print(df[heart_disease_column_name].value_counts())
else:
    print(f"Error: The column '{heart_disease_column_name}' was not found in the dataset. Cannot report value counts.")
print("-" * 40 + "\n")
```