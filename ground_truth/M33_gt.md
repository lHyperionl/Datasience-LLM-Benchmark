```python
import pandas as pd

# --- Configuration ---
# Define the path to the dataset CSV file.
DATASET_PATH = 'spotify_2023.csv'
# Define the fixed reference date for calculating days since release.
FIXED_DATE_FOR_CALCULATION = '2023-12-31'

# --- 1. Load the dataset ---
# Load the CSV file into a pandas DataFrame.
# Includes basic error handling for file loading.
try:
    df = pd.read_csv(DATASET_PATH)
    print(f"Dataset '{DATASET_PATH}' loaded successfully. Shape: {df.shape}")
except FileNotFoundError:
    print(f"Error: The file '{DATASET_PATH}' was not found. Please ensure it's in the current directory.")
    # Exit the script if the file is not found.
    exit()
except Exception as e:
    print(f"An unexpected error occurred while loading the dataset: {e}")
    exit()

# --- 2. Combine `released_year`, `released_month`, and `released_day` into a single datetime column ---
# Create a string representation of the date in 'YYYY-MM-DD' format.
# .astype(str) converts numeric columns to string.
# .str.zfill(2) ensures month and day are zero-padded to two digits (e.g., '1' becomes '01'),
# which helps in consistent date string formatting.
df['release_date_str'] = df['released_year'].astype(str) + '-' + \
                         df['released_month'].astype(str).str.zfill(2) + '-' + \
                         df['released_day'].astype(str).str.zfill(2)

# Convert the combined date string to a datetime object.
# `errors='coerce'` is crucial here: it will set any invalid date strings (e.g., '2023-02-30', '2023-13-01')
# to NaT (Not a Time), handling potential errors gracefully as required.
df['release_date'] = pd.to_datetime(df['release_date_str'], errors='coerce')

# Drop the intermediate string column as it's no longer needed.
df = df.drop(columns=['release_date_str'])

# --- 3. Create a new feature named `days_since_release` ---
# Convert the fixed reference date string into a pandas datetime object.
fixed_date = pd.to_datetime(FIXED_DATE_FOR_CALCULATION)

# Calculate the difference between the fixed date and the `release_date`.
# This operation results in a pandas Timedelta Series.
time_difference = fixed_date - df['release_date']

# Extract the number of days from the Timedelta Series.
# `.dt.days` returns an integer for valid Timedelta values and NaN for NaT values (where release_date was NaT).
# Convert the resulting Series to pandas' nullable integer type ('Int64').
# This allows the column to contain integer values while also accommodating NaN (missing) values,
# fulfilling the requirement for the feature to be an integer.
df['days_since_release'] = time_difference.dt.days.astype('Int64')

# --- 4. Display the `track_name`, `release_date`, and `days_since_release` for the first 5 tracks ---
print("\n--- First 5 tracks with new features ---")
# Select and display the specified columns for the first 5 rows of the DataFrame.
print(df[['track_name', 'release_date', 'days_since_release']].head(5))

# Optional: Display data types of the newly created columns to confirm their types.
# print("\n--- Data types of relevant columns ---")
# print(df[['release_date', 'days_since_release']].dtypes)
```