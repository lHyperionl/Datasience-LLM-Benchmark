```python
import pandas as pd

# Define the dataset file name
DATASET_FILE = 'penguins.csv'

# Define the numerical columns for NA imputation and statistics calculation
NUMERICAL_COLUMNS = ['culmen_length_mm', 'culmen_depth_mm']

# --- 1. Read the dataset ---
try:
    df = pd.read_csv(DATASET_FILE)
except FileNotFoundError:
    print(f"Error: The file '{DATASET_FILE}' was not found. Please ensure it is in the current directory.")
    exit() # Exit if the file is not found, as subsequent operations will fail

# --- 2. For the numerical columns `culmen_length_mm` and `culmen_depth_mm`,
#        fill any NA values with the mean of their respective columns. ---

print("--- NA Imputation Summary ---")
for col in NUMERICAL_COLUMNS:
    if col in df.columns:
        # Calculate the mean of the column *before* filling NAs.
        # This ensures the mean is based on the existing valid data points.
        col_mean = df[col].mean()
        
        # Store initial NA count for reporting
        initial_na_count = df[col].isnull().sum()

        # Fill NA values with the calculated mean
        df[col] = df[col].fillna(col_mean)
        
        # Print a summary of the imputation for clarity
        print(f"Column '{col}':")
        print(f"  - Initial NA count: {initial_na_count}")
        print(f"  - Mean used for imputation: {col_mean:.2f}")
        print(f"  - NA count after imputation: {df[col].isnull().sum()}\n")
    else:
        print(f"Warning: Column '{col}' not found in the dataset. Skipping NA imputation for this column.\n")

# --- 3. Calculate and print the mean, median, and standard deviation
#        for the cleaned `culmen_length_mm` and `culmen_depth_mm` columns. ---

print("--- Statistics for Cleaned Numerical Columns ---")
for col in NUMERICAL_COLUMNS:
    if col in df.columns:
        mean_val = df[col].mean()
        median_val = df[col].median()
        std_val = df[col].std()

        print(f"Statistics for '{col}':")
        print(f"  - Mean: {mean_val:.2f}")
        print(f"  - Median: {median_val:.2f}")
        print(f"  - Standard Deviation: {std_val:.2f}\n")
    else:
        # This case should ideally not be reached if the column was checked during imputation
        print(f"Could not calculate statistics for '{col}' as it was not found or processed.\n")

# --- 4. Print the total number of rows and columns in the dataset. ---

print("--- Dataset Dimensions ---")
num_rows, num_cols = df.shape
print(f"Total number of rows: {num_rows}")
print(f"Total number of columns: {num_cols}")
```