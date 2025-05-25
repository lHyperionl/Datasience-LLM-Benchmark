```python
import pandas as pd

# --- 1. Load the iris.csv dataset ---
# Assume iris.csv is in the current directory.
# A try-except block is used for robust file loading.
try:
    df = pd.read_csv('iris.csv')
    print("Dataset 'iris.csv' loaded successfully.\n")
except FileNotFoundError:
    print("Error: 'iris.csv' not found. Please ensure the file is in the current directory.")
    # Exit the script if the file is not found, as subsequent operations will fail.
    exit()
except Exception as e:
    print(f"An unexpected error occurred while loading the dataset: {e}")
    exit()

# --- 2. List all column names and identify the data type of each column ---
print("--- Column Names and Data Types ---")
print("Column Names:", df.columns.tolist())
print("\nData Types:")
print(df.dtypes)
print("-" * 40 + "\n")

# --- 3. Calculate and report basic descriptive statistics for specified numerical columns ---
numerical_columns = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']

print("--- Descriptive Statistics for Numerical Columns ---")
# Check if all specified numerical columns exist in the DataFrame.
# This handles cases where column names might be misspelled or missing.
existing_numerical_columns = [col for col in numerical_columns if col in df.columns]

if not existing_numerical_columns:
    print("None of the specified numerical columns were found in the dataset.")
else:
    # Report any missing columns from the specified list.
    missing_cols = set(numerical_columns) - set(existing_numerical_columns)
    if missing_cols:
        print(f"Warning: The following specified numerical columns were not found: {', '.join(missing_cols)}")
        print("Calculating statistics for available numerical columns.\n")

    # Calculate mean, median, standard deviation, min, and max using .agg() for clarity.
    # .describe() also provides these, but .agg() allows for explicit selection.
    descriptive_stats = df[existing_numerical_columns].agg(['mean', 'median', 'std', 'min', 'max'])
    print(descriptive_stats)

print("-" * 40 + "\n")

# --- 4. Report the number of unique species in the 'Species' column and their respective counts ---
print("--- Unique Species and Their Counts ---")
species_column_name = 'Species'

# Check if the 'Species' column exists in the DataFrame.
if species_column_name in df.columns:
    unique_species_counts = df[species_column_name].value_counts()
    print(f"Number of unique species: {len(unique_species_counts)}")
    print("\nCounts of each species:")
    print(unique_species_counts)
else:
    print(f"Error: The '{species_column_name}' column was not found in the dataset.")

print("-" * 40 + "\n")
```