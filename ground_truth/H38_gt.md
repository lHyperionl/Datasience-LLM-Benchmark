```python
import pandas as pd
import numpy as np

def clean_price(price_str):
    """
    Cleans a price string by removing currency symbols and commas,
    then converts it to a numeric (float) type.
    Handles errors by returning NaN if conversion fails.

    Args:
        price_str (str or float): The price string to clean.

    Returns:
        float: The cleaned numeric price, or NaN if cleaning/conversion fails.
    """
    # Return NaN immediately if the input is already NaN
    if pd.isna(price_str):
        return np.nan
    try:
        # Ensure the input is treated as a string for string operations
        price_str = str(price_str)
        
        # Remove specified currency symbols
        currency_symbols = ['PKR', 'INR', 'CNY', 'USD', 'AED']
        for symbol in currency_symbols:
            price_str = price_str.replace(symbol, '')
        
        # Remove commas
        price_str = price_str.replace(',', '')
        
        # Convert to float after stripping any leading/trailing whitespace
        return float(price_str.strip())
    except ValueError:
        # Return NaN if conversion to float fails
        return np.nan

# --- Main Script ---

# 1. Load the dataset
try:
    df = pd.read_csv('mobiles_2025.csv')
except FileNotFoundError:
    print("Error: 'mobiles_2025.csv' not found. Please ensure the file is in the current directory.")
    exit()

# Define the price columns to be cleaned
price_columns = [
    'Launched Price (Pakistan)',
    'Launched Price (India)',
    'Launched Price (China)',
    'Launched Price (USA)',
    'Launched Price (Dubai)'
]

# 2. Apply the cleaning function to each specified price column
for col in price_columns:
    if col in df.columns:
        df[col] = df[col].apply(clean_price)
    else:
        print(f"Warning: Column '{col}' not found in the dataset. Skipping cleaning for this column.")

# 3. For each mobile phone model, calculate its 'Average Global Price'
# This average should only consider countries for which price data is available (not NaN)
# `skipna=True` ensures that NaN values are ignored when calculating the mean.
df['Average Global Price'] = df[price_columns].mean(axis=1, skipna=True)

# 4. Identify the top 5 mobile phone models with the highest 'Average Global Price'
# Sort the DataFrame by 'Average Global Price' in descending order
df_sorted_by_price = df.sort_values(by='Average Global Price', ascending=False)

# Select the top 5 models
top_5_models = df_sorted_by_price.head(5)

# 5. Display the 'Model Name' and their calculated 'Average Global Price'
# for these top 5 models, sorted by 'Average Global Price' descending.
# Filter out any models where Average Global Price might be NaN (e.g., if all price columns were NaN)
top_5_models_display = top_5_models[['Model Name', 'Average Global Price']].dropna(subset=['Average Global Price'])

# Print the results
print("Top 5 Mobile Phone Models by Average Global Price:")
print(top_5_models_display)
```