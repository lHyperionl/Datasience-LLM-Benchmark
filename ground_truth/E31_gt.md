```python
import pandas as pd

# --- 1. Load the dataset ---
# Assume 'spotify_2023.csv' is in the current working directory.
try:
    df = pd.read_csv('spotify_2023.csv')
    print("Dataset 'spotify_2023.csv' loaded successfully.")
except FileNotFoundError:
    print("Error: 'spotify_2023.csv' not found. Please ensure the file is in the current directory.")
    exit() # Exit if the file is not found, as subsequent operations will fail.

# --- 2. List all column names and identify the data type of each ---
print("\n--- Column Names and Data Types ---")
# Using .info() provides a comprehensive summary including column names, non-null counts, and dtypes.
df.info()

# --- 3. Convert the 'streams' column to a numeric data type ---
# Handle potential errors by coercing non-numeric values to NaN.
print("\n--- Converting 'streams' to Numeric and Reporting NaNs ---")
initial_streams_dtype = df['streams'].dtype
df['streams'] = pd.to_numeric(df['streams'], errors='coerce')
print(f"Original 'streams' dtype: {initial_streams_dtype}")
print(f"New 'streams' dtype after conversion: {df['streams'].dtype}")

# Report any NaNs introduced or present after conversion.
nan_streams_count = df['streams'].isnull().sum()
print(f"Number of NaN values in 'streams' after conversion: {nan_streams_count}")

if nan_streams_count > 0:
    print("Sample rows with NaN 'streams' values (first 5):")
    print(df[df['streams'].isnull()].head())
else:
    print("No NaN values found in 'streams' after conversion.")

# --- 4. Calculate and report basic descriptive statistics for specified numerical columns ---
# Columns: 'streams' (after conversion), 'bpm', 'danceability_%', 'valence_%', 'energy_%'
print("\n--- Basic Descriptive Statistics for Numerical Columns ---")
numerical_cols_for_stats = ['streams', 'bpm', 'danceability_%', 'valence_%', 'energy_%']

# Filter for columns that actually exist in the DataFrame to prevent KeyError
existing_numerical_cols = [col for col in numerical_cols_for_stats if col in df.columns]

if existing_numerical_cols:
    # Use .describe() and select specific statistics (mean, median (50%), std, min, max)
    # Note: '50%' is the median in .describe() output.
    descriptive_stats = df[existing_numerical_cols].describe().loc[['mean', '50%', 'std', 'min', 'max']]
    print(descriptive_stats)
else:
    print("None of the specified numerical columns for statistics were found in the dataset.")

# --- 5. Report the total number of unique artists found in the 'artist(s)_name' column ---
print("\n--- Total Number of Unique Artists ---")
if 'artist(s)_name' in df.columns:
    unique_artists_count = df['artist(s)_name'].nunique()
    print(f"Total number of unique artists: {unique_artists_count}")
else:
    print("Column 'artist(s)_name' not found in the dataset.")
```