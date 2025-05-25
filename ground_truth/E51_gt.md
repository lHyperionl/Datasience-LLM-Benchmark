```python
import pandas as pd

# --- 1. Create the pandas DataFrame ---
# Define the data for the DataFrame
data = {
    'Date': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05', '2024-01-08']),
    'Ticker': ['TECHCO', 'TECHCO', 'TECHCO', 'TECHCO', 'TECHCO', 'TECHCO'],
    'Open': [150.0, 152.5, 151.0, 153.0, 155.5, 154.0],
    'High': [153.0, 153.5, 154.0, 156.0, 157.0, 156.5],
    'Low': [149.5, 150.5, 150.0, 152.5, 154.5, 153.0],
    'Close': [152.0, 151.0, 153.5, 155.0, 154.5, 156.0],
    'Volume': [1000000, 1200000, 1100000, 1300000, 900000, 1050000]
}

# Create the DataFrame
df = pd.DataFrame(data)

# Set 'Date' as the index of the DataFrame
# Using inplace=True modifies the DataFrame directly
df.set_index('Date', inplace=True)

# --- 2. Calculate the daily price range ---
# Calculate 'Daily_Range' by subtracting 'Low' from 'High'
df['Daily_Range'] = df['High'] - df['Low']

# --- 3. Find and print the 'Date' with the highest 'Close' price ---
# Use idxmax() to get the index (Date) of the maximum 'Close' price
date_highest_close = df['Close'].idxmax()

print(f"Date with the highest 'Close' price: {date_highest_close.strftime('%Y-%m-%d')}\n")

# --- 4. Print the DataFrame including the new 'Daily_Range' column ---
print("DataFrame with 'Daily_Range' column:")
print(df)
```