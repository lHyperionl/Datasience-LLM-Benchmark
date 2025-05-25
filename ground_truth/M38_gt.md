```python
import pandas as pd
import re

def clean_camera_mp(camera_spec: str) -> int:
    """
    Cleans a camera specification string to extract the first sequence of digits
    as the Megapixel (MP) value.

    Args:
        camera_spec (str): The camera specification string (e.g., '12MP / 4K', '48MP + 12MP').

    Returns:
        int: The extracted MP value as an integer. Returns 0 if no digits are found
             or if the input is NaN/None.
    """
    # Handle NaN or non-string inputs by converting to string first
    if pd.isna(camera_spec):
        return 0
    
    spec_str = str(camera_spec)
    
    # Use regex to find the first sequence of one or more digits
    match = re.search(r'\d+', spec_str)
    
    if match:
        # If digits are found, convert the matched string to an integer
        return int(match.group(0))
    else:
        # If no digits are found, return 0
        return 0

# --- Main script execution ---

# 1. Load the dataset
try:
    df = pd.read_csv('mobiles_2025.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: 'mobiles_2025.csv' not found. Please ensure the file is in the current directory.")
    exit() # Exit if the file is not found

# 2. Apply the cleaning function to 'Front Camera' and 'Back Camera' columns
# Create new cleaned numeric columns
print("Cleaning camera specifications...")
df['Cleaned Front Camera'] = df['Front Camera'].apply(clean_camera_mp)
df['Cleaned Back Camera'] = df['Back Camera'].apply(clean_camera_mp)
print("Camera specifications cleaned.")

# 3. Create a new feature 'Total Camera MP'
# This is the sum of the cleaned front and back camera MP values
print("Calculating 'Total Camera MP'...")
df['Total Camera MP'] = df['Cleaned Front Camera'] + df['Cleaned Back Camera']
print("'Total Camera MP' calculated.")

# 4. Display the 'Model Name', 'Cleaned Front Camera', 'Cleaned Back Camera',
# and 'Total Camera MP' columns for the first 10 rows
print("\nDisplaying results for the first 10 rows:")
display_columns = ['Model Name', 'Cleaned Front Camera', 'Cleaned Back Camera', 'Total Camera MP']
print(df[display_columns].head(10).to_string()) # Using .to_string() for better console formatting
```