```python
import pandas as pd

# Define the path to the dataset
file_path = 'clash_royale_cards.csv'

# --- 1. Load the dataset ---
try:
    df = pd.read_csv(file_path)
    print(f"Dataset '{file_path}' loaded successfully.")
except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found. Please ensure it's in the current directory.")
    # Exit the script if the file is not found, as subsequent operations will fail.
    exit()
except Exception as e:
    print(f"An unexpected error occurred while loading the dataset: {e}")
    exit()

# --- 2. List all column names and identify the data type of each column ---
print("\n--- Column Names and Data Types ---")
print("Column Names:")
# Iterate and print each column name for clear listing
for col in df.columns:
    print(f"- {col}")

print("\nData Types of Each Column:")
# Print the Series containing column names and their respective data types
print(df.dtypes)

# --- 3. Calculate and report basic descriptive statistics for specified numerical columns ---
# Define the list of numerical columns for which to calculate statistics
numerical_cols_for_stats = ['elixirCost', 'Win Rate', 'Usage']
print("\n--- Descriptive Statistics for Numerical Columns ---")

# Filter for columns that actually exist in the DataFrame to prevent KeyError
existing_numerical_cols = [col for col in numerical_cols_for_stats if col in df.columns]

if existing_numerical_cols:
    # Calculate mean, median, standard deviation, min, and max using the .agg() method
    # This provides a concise way to get multiple statistics for multiple columns
    descriptive_stats = df[existing_numerical_cols].agg(['mean', 'median', 'std', 'min', 'max'])
    print(descriptive_stats)
else:
    # Inform the user if none of the specified columns were found
    print(f"Warning: None of the specified numerical columns ({', '.join(numerical_cols_for_stats)}) were found in the dataset.")

# --- 4. Provide the total number of cards (rows) in the dataset ---
# Get the total number of rows using df.shape[0]
total_cards = df.shape[0]
print("\n--- Total Number of Cards ---")
print(f"Total number of cards (rows) in the dataset: {total_cards}")
```