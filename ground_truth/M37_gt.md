```python
import pandas as pd

# Define the path to the dataset
file_path = 'mobiles_2025.csv'

# --- Step 1: Load the mobiles_2025.csv dataset using pandas ---
try:
    df = pd.read_csv(file_path)
    print(f"Successfully loaded '{file_path}'. Initial shape: {df.shape}")
except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found. Please ensure it's in the current directory.")
    exit()
except Exception as e:
    print(f"An error occurred while loading the CSV: {e}")
    exit()

# --- Step 2: Clean the 'Launched Price (USA)' column ---
# Remove 'USD ' prefix and commas, then convert to numeric.
# Handle conversion errors by setting to NaN using errors='coerce'.
print("\nCleaning 'Launched Price (USA)' column...")
# Ensure the column is treated as string before applying string operations
df['Launched Price (USA)'] = df['Launched Price (USA)'].astype(str)
df['Cleaned Price (USD)'] = df['Launched Price (USA)'].str.replace('USD ', '', regex=False)
df['Cleaned Price (USD)'] = df['Cleaned Price (USD)'].str.replace(',', '', regex=False)
df['Cleaned Price (USD)'] = pd.to_numeric(df['Cleaned Price (USD)'], errors='coerce')
print(f"Number of NaN values introduced in 'Cleaned Price (USD)': {df['Cleaned Price (USD)'].isnull().sum()}")

# --- Step 3: Clean the 'Screen Size' column ---
# Remove ' inches' suffix and convert to numeric.
# Handle conversion errors by setting to NaN using errors='coerce'.
print("Cleaning 'Screen Size' column...")
# Ensure the column is treated as string before applying string operations
df['Screen Size'] = df['Screen Size'].astype(str)
df['Cleaned Screen Size (inches)'] = df['Screen Size'].str.replace(' inches', '', regex=False)
df['Cleaned Screen Size (inches)'] = pd.to_numeric(df['Cleaned Screen Size (inches)'], errors='coerce')
print(f"Number of NaN values introduced in 'Cleaned Screen Size (inches)': {df['Cleaned Screen Size (inches)'].isnull().sum()}")

# --- Step 4: Group the data by 'Company Name' and 'Launched Year' ---
# --- Step 5: For each group, calculate the average of cleaned 'Launched Price (USA)'
#             and the average of cleaned 'Screen Size'.
print("\nGrouping data and calculating averages...")
aggregated_data = df.groupby(['Company Name', 'Launched Year']).agg(
    Average_Launched_Price_USD=('Cleaned Price (USD)', 'mean'),
    Average_Screen_Size_inches=('Cleaned Screen Size (inches)', 'mean')
).reset_index() # reset_index to turn the grouped keys back into columns

# --- Step 6: Display the resulting aggregated data ---
print("\nAggregated Data (Average Price and Screen Size by Company and Launch Year):")
print(aggregated_data)

# Optional: Display some basic info about the aggregated data
print(f"\nShape of aggregated data: {aggregated_data.shape}")
print("\nData types of aggregated data:")
print(aggregated_data.dtypes)
```