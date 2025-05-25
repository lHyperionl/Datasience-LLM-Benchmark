```python
import pandas as pd

# --- Configuration ---
FILE_PATH = 'mobiles_2025.csv'
TARGET_COMPANY = 'Apple'
MAX_PRICE_INDIA = 100000

# --- Load the dataset ---
try:
    df = pd.read_csv(FILE_PATH)
    print(f"Dataset '{FILE_PATH}' loaded successfully. Shape: {df.shape}")
except FileNotFoundError:
    print(f"Error: The file '{FILE_PATH}' was not found. Please ensure it's in the current directory.")
    exit()
except Exception as e:
    print(f"An error occurred while loading the dataset: {e}")
    exit()

# --- 1) Clean the 'Launched Price (India)' column ---
# Convert to string first to handle potential non-string types gracefully
# Remove 'INR ' prefix
# Remove commas
# Convert to numeric, coercing errors to NaN
print("\nCleaning 'Launched Price (India)' column...")
df['Launched Price (India)'] = df['Launched Price (India)'].astype(str).str.replace('INR ', '', regex=False)
df['Launched Price (India)'] = df['Launched Price (India)'].str.replace(',', '', regex=False)
df['Launched Price (India)'] = pd.to_numeric(df['Launched Price (India)'], errors='coerce')
print("Finished cleaning 'Launched Price (India)'. NaNs introduced for errors.")

# --- 2) Clean the 'RAM' column ---
# Convert to string first to handle potential non-string types gracefully
# Remove 'GB' suffix
# Convert to numeric, coercing errors to NaN
print("Cleaning 'RAM' column...")
df['RAM'] = df['RAM'].astype(str).str.replace('GB', '', regex=False)
df['RAM'] = pd.to_numeric(df['RAM'], errors='coerce')
print("Finished cleaning 'RAM'. NaNs introduced for errors.")

# --- 3) Filter the dataset ---
# Filter for 'Company Name' is 'Apple'
# Filter for cleaned 'Launched Price (India)' less than 100,000
print(f"\nFiltering data for '{TARGET_COMPANY}' and price < {MAX_PRICE_INDIA}...")
filtered_df = df[
    (df['Company Name'] == TARGET_COMPANY) &
    (df['Launched Price (India)'] < MAX_PRICE_INDIA)
].copy() # Use .copy() to avoid SettingWithCopyWarning

if filtered_df.empty:
    print(f"No data found for '{TARGET_COMPANY}' with 'Launched Price (India)' less than {MAX_PRICE_INDIA}.")
else:
    print(f"Filtered data shape: {filtered_df.shape}")

    # --- 4) Select and display specified columns ---
    # Select 'Model Name', cleaned 'RAM', and cleaned 'Launched Price (India)'
    print("\nSelecting specified columns: 'Model Name', 'RAM', 'Launched Price (India)'...")
    selected_columns_df = filtered_df[['Model Name', 'RAM', 'Launched Price (India)']]

    # --- 5) Sort the resulting data ---
    # Sort by cleaned 'Launched Price (India)' in ascending order
    print("Sorting by 'Launched Price (India)' in ascending order...")
    sorted_df = selected_columns_df.sort_values(by='Launched Price (India)', ascending=True)

    # --- Display the final result ---
    print("\n--- Final Result ---")
    print(sorted_df)

```