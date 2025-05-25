```python
import pandas as pd
import sys # Required for sys.exit() to terminate the script gracefully on error

# --- Configuration ---
DATASET_PATH = 'imdb_top_250.csv'
# Define the numerical columns for which descriptive statistics are required
NUMERICAL_COLUMNS_FOR_STATS = ['Year', 'Rating', 'Runtime']

# --- Step 1: Load the dataset ---
# Use a try-except block to handle potential FileNotFoundError
try:
    df = pd.read_csv(DATASET_PATH)
    print(f"Successfully loaded '{DATASET_PATH}'.")
except FileNotFoundError:
    print(f"Error: The file '{DATASET_PATH}' was not found.")
    print("Please ensure the CSV file is in the same directory as this script.")
    sys.exit(1) # Exit the script if the file is not found

# --- Step 2: Preprocessing for 'Runtime' column if necessary ---
# The 'Runtime' column often comes in a string format like '142 min'.
# It needs to be converted to a numeric type for statistical calculations.
if 'Runtime' in df.columns:
    # Check if 'Runtime' is of object type (string)
    if df['Runtime'].dtype == 'object':
        print("\n--- Preprocessing 'Runtime' column ---")
        print("Converting 'Runtime' from string (e.g., '142 min') to numeric.")
        # Remove the ' min' suffix and convert the column to numeric.
        # errors='coerce' will convert any non-numeric values (after stripping ' min') to NaN.
        df['Runtime'] = pd.to_numeric(df['Runtime'].str.replace(' min', '', regex=False), errors='coerce')
    else:
        print("\n'Runtime' column is already numeric or not found, no conversion needed.")
else:
    print(f"\nWarning: 'Runtime' column not found in the dataset. Skipping runtime statistics.")
    # Remove 'Runtime' from the list of columns to analyze if it's missing from the DataFrame
    if 'Runtime' in NUMERICAL_COLUMNS_FOR_STATS:
        NUMERICAL_COLUMNS_FOR_STATS.remove('Runtime')

# --- Step 3: List all column names and their data types ---
print("\n--- Column Names and Data Types ---")
# df.info() provides a concise summary including column names, non-null counts, and data types.
df.info()

# --- Step 4: Calculate and report basic descriptive statistics ---
print("\n--- Descriptive Statistics for Numerical Columns ---")

# Filter the list of numerical columns to include only those that exist in the DataFrame
# and are actually numeric after any preprocessing.
existing_numeric_cols = [
    col for col in NUMERICAL_COLUMNS_FOR_STATS
    if col in df.columns and pd.api.types.is_numeric_dtype(df[col])
]

if existing_numeric_cols:
    # Use the .describe() method on the selected numerical columns.
    # Then, use .loc to select specific statistics: mean, median (50th percentile),
    # standard deviation (std), minimum (min), and maximum (max).
    descriptive_stats = df[existing_numeric_cols].describe().loc[['mean', '50%', 'std', 'min', 'max']]
    
    # Rename the '50%' index to 'median' for better readability
    descriptive_stats.rename(index={'50%': 'median'}, inplace=True)
    
    # Print the descriptive statistics, rounded to 2 decimal places for clarity
    print(descriptive_stats.round(2))
else:
    print("No numerical columns found for statistics among 'Year', 'Rating', 'Runtime' after checks.")

# --- Step 5: Report the total number of movies in the dataset ---
print("\n--- Total Number of Movies ---")
# The total number of movies is simply the number of rows in the DataFrame
total_movies = len(df)
print(f"Total number of movies in the dataset: {total_movies}")
```