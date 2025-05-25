```python
import pandas as pd

# 1. Create the pandas DataFrame
# Define the raw data for stock prices
data = {
    'Date': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05', '2024-01-08', '2024-01-09', '2024-01-10']),
    'Ticker': ['TECHCO']*8,
    'Open': [150.0, 152.5, 151.0, 153.0, 155.5, 154.0, 156.0, 155.0],
    'High': [153.0, 153.5, 154.0, 156.0, 157.0, 156.5, 157.5, 156.5],
    'Low': [149.5, 150.5, 150.0, 152.5, 154.5, 153.0, 155.0, 154.0],
    'Close': [152.0, 151.0, 153.5, 155.0, 154.5, 156.0, 157.0, 154.0],
    'Volume': [1000000, 1200000, 1100000, 1300000, 900000, 1050000, 1150000, 950000]
}

# Create the DataFrame from the data
df = pd.DataFrame(data)

# Set 'Date' as the index of the DataFrame
# This is crucial for time-series operations and proper alignment
df.set_index('Date', inplace=True)

# 2. Calculate the 3-day Simple Moving Average (SMA) of the 'Close' price
# The .rolling() method provides rolling windows, and .mean() calculates the average
# for each window. By default, it handles initial days by producing NaN values
# until enough data points are available for the specified window (e.g., 3 days).
df['SMA_3Day'] = df['Close'].rolling(window=3).mean()

# 3. Calculate the percentage change in 'Volume' from the previous day
# The .pct_change() method calculates the percentage change between the current
# and a prior element. It automatically places a NaN for the first element
# as there is no previous day to compare with.
df['Volume_Pct_Change'] = df['Volume'].pct_change()

# 4. Print the resulting DataFrame with the new columns
print("DataFrame with calculated SMA_3Day and Volume_Pct_Change:")
print(df)
```