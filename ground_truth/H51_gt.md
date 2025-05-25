```python
import pandas as pd
import matplotlib.pyplot as plt

# --- 1. Create DataFrame and Set Index ---

# Provided data for TECHCO stock prices
data = {
    'Date': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05',
                            '2024-01-08', '2024-01-09', '2024-01-10', '2024-01-11', '2024-01-12']),
    'Ticker': ['TECHCO'] * 10,
    'Open': [150.0, 152.5, 151.0, 153.0, 155.5, 154.0, 156.0, 155.0, 153.0, 150.0],
    'High': [153.0, 153.5, 154.0, 156.0, 157.0, 156.5, 157.5, 156.5, 155.0, 152.0],
    'Low': [149.5, 150.5, 150.0, 152.5, 154.5, 153.0, 155.0, 154.0, 151.0, 149.0],
    'Close': [152.0, 151.0, 153.5, 155.0, 154.5, 156.0, 157.0, 154.0, 151.5, 150.5],
    'Volume': [1000000, 1200000, 1100000, 1300000, 900000, 1050000, 1150000, 950000, 1250000, 1300000]
}

# Create the pandas DataFrame
df = pd.DataFrame(data)

# Set 'Date' column as the DataFrame index
df.set_index('Date', inplace=True)

# Display the initial DataFrame (optional)
# print("Initial DataFrame:")
# print(df)
# print("\n" + "="*50 + "\n")

# --- 2. Calculate Daily Percentage Return ---

# Calculate daily percentage return using the 'Close' price
# Formula: (Close_t / Close_t-1 - 1) * 100
df['Daily_Return'] = df['Close'].pct_change() * 100

# --- 3. Calculate Simple Moving Averages (SMAs) ---

# Calculate the 3-day Simple Moving Average (SMA) for 'Close' price
df['SMA_3Day'] = df['Close'].rolling(window=3).mean()

# Calculate the 5-day Simple Moving Average (SMA) for 'Close' price
df['SMA_5Day'] = df['Close'].rolling(window=5).mean()

# Display DataFrame with new calculated columns (optional)
# print("DataFrame with Daily Return and SMAs:")
# print(df)
# print("\n" + "="*50 + "\n")

# --- 4. Generate and Display Line Plot ---

# Set up the plot figure size for better readability
plt.figure(figsize=(12, 7))

# Plot 'Close' price
plt.plot(df.index, df['Close'], label='Close Price', color='blue', linewidth=2)

# Plot 'SMA_3Day'
plt.plot(df.index, df['SMA_3Day'], label='SMA_3Day', color='red', linestyle='--', linewidth=1.5)

# Plot 'SMA_5Day'
plt.plot(df.index, df['SMA_5Day'], label='SMA_5Day', color='green', linestyle=':', linewidth=1.5)

# Set plot title and labels
plt.title('TECHCO Close Price & Moving Averages', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Price', fontsize=12)

# Add a legend to distinguish the lines
plt.legend(fontsize=10)

# Add a grid for better readability
plt.grid(True, linestyle='--', alpha=0.7)

# Adjust layout to prevent labels from overlapping
plt.tight_layout()

# Display the plot
plt.show()

# --- 5. Identify and Print Bullish Crossover Dates ---

# A bullish crossover occurs when the shorter-term SMA (SMA_3Day) crosses above the longer-term SMA (SMA_5Day).
# This means two conditions must be met:
# 1. On the current day, SMA_3Day is greater than SMA_5Day.
# 2. On the previous day, SMA_3Day was less than or equal to SMA_5Day.

# Calculate boolean series for the current day's condition
current_bullish_condition = (df['SMA_3Day'] > df['SMA_5Day'])

# Calculate boolean series for the previous day's condition
# .shift(1) moves the series values down by 1, so df.shift(1) gives previous day's values
previous_bearish_or_equal_condition = (df['SMA_3Day'].shift(1) <= df['SMA_5Day'].shift(1))

# Combine both conditions to find the exact crossover points
# The '&' operator performs element-wise logical AND.
# NaN values in either series will result in False for the combined condition, which correctly handles initial NaNs from rolling/shift.
bullish_crossovers = df.index[current_bullish_condition & previous_bearish_or_equal_condition].tolist()

# Print the identified dates
print("\nDates where SMA_3Day crossed above SMA_5Day (Bullish Signal):")
if bullish_crossovers:
    for date in bullish_crossovers:
        print(date.strftime('%Y-%m-%d')) # Format date for cleaner output
else:
    print("No bullish crossovers found in the given data.")

```