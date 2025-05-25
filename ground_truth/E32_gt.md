```python
import pandas as pd

# Define the path to the dataset
DATASET_PATH = 'spotify_2023.csv'

# --- Step 1: Load the spotify_2023.csv dataset using pandas ---
try:
    df = pd.read_csv(DATASET_PATH)
except FileNotFoundError:
    print(f"Error: The file '{DATASET_PATH}' was not found. Please ensure it's in the current directory.")
    exit()
except Exception as e:
    print(f"An error occurred while loading the dataset: {e}")
    exit()

# --- Step 2: Ensure the 'streams' column is treated as numeric (coerce errors to NaN if necessary) ---
# This step is performed early to ensure correct filtering and sorting later.
# Using errors='coerce' will turn any non-numeric values into NaN.
df['streams'] = pd.to_numeric(df['streams'], errors='coerce')

# --- Step 3: Filter the dataset to include only songs where 'released_year' is 2023 ---
# Using .loc for explicit indexing and to avoid SettingWithCopyWarning.
df_2023 = df.loc[df['released_year'] == 2023]

# --- Step 4: From this filtered dataset, select the 'track_name', 'artist(s)_name', and 'streams' columns ---
# Select the desired columns from the filtered DataFrame.
selected_columns_df = df_2023[['track_name', 'artist(s)_name', 'streams']]

# --- Step 5: Sort the resulting data by the 'streams' column in descending order ---
# Drop rows where 'streams' might be NaN (due to coercion) before sorting,
# as NaN values would typically be placed at the end during sorting.
# If NaNs should be kept, this line can be omitted. For a clean top-streams list, dropping them is good.
sorted_df = selected_columns_df.dropna(subset=['streams'])

# Sort by 'streams' in descending order.
sorted_df = sorted_df.sort_values(by='streams', ascending=False)

# --- Step 6: Display the resulting data ---
# Display the top entries of the sorted DataFrame.
# You can adjust the number of rows to display, e.g., .head(10) for top 10.
print("Songs released in 2023, sorted by streams (descending):")
print(sorted_df)
```