```python
import pandas as pd

# Define the path to the dataset
# Assuming 'penguins.csv' is in the current working directory
file_path = 'penguins.csv'

# --- Step 1: Load the penguins.csv dataset ---
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found. Please ensure it's in the current directory.")
    exit()
except Exception as e:
    print(f"An error occurred while loading the CSV file: {e}")
    exit()

# --- Step 2: Filter the dataset ---
# Filter for records where 'species' is 'Adelie' AND 'island' is 'Torgersen'
filtered_df = df[(df['species'] == 'Adelie') & (df['island'] == 'Torgersen')]

# --- Step 3: Select specific columns ---
# Select only 'culmen_length_mm' and 'culmen_depth_mm'
selected_columns_df = filtered_df[['culmen_length_mm', 'culmen_depth_mm']]

# --- Step 4: Drop rows with missing values ---
# Drop rows where either 'culmen_length_mm' or 'culmen_depth_mm' has a missing value (NaN)
# The 'subset' argument ensures that NaNs are checked only in these two columns
cleaned_df = selected_columns_df.dropna(subset=['culmen_length_mm', 'culmen_depth_mm'])

# --- Step 5: Display the first 5 rows of the resulting DataFrame ---
print("First 5 rows of the processed DataFrame:")
print(cleaned_df.head(5))

# Optional: Display the shape of the final DataFrame to confirm processing
# print(f"\nShape of the final DataFrame: {cleaned_df.shape}")
```