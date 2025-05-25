```python
import pandas as pd

# Define the path to the dataset
DATASET_PATH = 'penguins.csv'

# --- 1. Load the penguins.csv dataset ---
# Attempt to load the dataset, handling potential FileNotFoundError
try:
    df = pd.read_csv(DATASET_PATH)
    print(f"Successfully loaded '{DATASET_PATH}'. Dataset shape: {df.shape}\n")
except FileNotFoundError:
    print(f"Error: The file '{DATASET_PATH}' was not found. Please ensure it's in the current directory.")
    # Exit the script if the file is not found as subsequent operations would fail
    exit()
except Exception as e:
    print(f"An unexpected error occurred while loading the dataset: {e}")
    exit()

# --- 2. Report the number of missing values (NA) for each column ---
print("--- Missing Values Report ---")
# Calculate the sum of null values for each column
missing_values_count = df.isnull().sum()
# Filter to show only columns with missing values
columns_with_missing_values = missing_values_count[missing_values_count > 0]

if not columns_with_missing_values.empty:
    print(columns_with_missing_values)
else:
    print("No missing values found in any column.")
print("\n")

# --- 3. For specific numerical columns, drop NAs and report descriptive statistics ---
# Define the list of numerical columns for this specific analysis
numerical_cols_for_analysis = [
    'culmen_length_mm',
    'culmen_depth_mm',
    'flipper_length_mm',
    'body_mass_g'
]

print("--- Descriptive Statistics for Filtered Numerical Columns ---")
# Store the original number of rows before filtering
original_rows_count = len(df)

# Drop rows where *any* of the four specified numerical columns have a missing value
# The 'subset' argument ensures that only NAs in these specific columns lead to row drops
df_filtered = df.dropna(subset=numerical_cols_for_analysis)

# Report the number of rows before and after filtering
print(f"Original number of rows: {original_rows_count}")
print(f"Number of rows after dropping NAs in specified columns: {len(df_filtered)}\n")

# Calculate and report basic descriptive statistics for these four columns
# Check if the filtered DataFrame is not empty to avoid errors
if not df_filtered.empty:
    # Use .agg() to calculate multiple statistics efficiently
    descriptive_stats = df_filtered[numerical_cols_for_analysis].agg(
        ['mean', 'median', 'std', 'min', 'max']
    )
    print(descriptive_stats)
else:
    print("No data remaining after dropping missing values for the specified columns.")
print("\n")

# --- 4. Count and report the number of unique values in 'species' and 'island' columns ---
print("--- Unique Values Report ---")

# Check if 'species' column exists before proceeding
if 'species' in df.columns:
    unique_species_count = df['species'].nunique()
    print(f"Number of unique values in 'species' column: {unique_species_count}")
else:
    print("Column 'species' not found in the dataset.")

# Check if 'island' column exists before proceeding
if 'island' in df.columns:
    unique_island_count = df['island'].nunique()
    print(f"Number of unique values in 'island' column: {unique_island_count}")
else:
    print("Column 'island' not found in the dataset.")
print("\n")

print("--- Analysis Complete ---")
```