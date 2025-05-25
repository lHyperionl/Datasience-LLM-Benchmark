```python
import pandas as pd

# 1. Define the pandas DataFrame representing environmental sensor readings
# The data dictionary is provided in the task prompt.
data = {
    "Timestamp": ["2025-05-24 10:00", "2025-05-24 10:05", "2025-05-24 10:10", "2025-05-24 10:15", "2025-05-24 10:00", "2025-05-24 10:05", "2025-05-24 10:20", "2025-05-24 10:25"],
    "SensorID": ["S1", "S1", "S1", "S1", "S2", "S2", "S1", "S2"],
    "TemperatureCelsius": [22.5, 22.7, 22.6, 28.0, 21.9, 22.0, 15.0, 22.1],
    "HumidityPercent": [55, 56, 55, 57, 60, 59, 70, 58],
    "AirQualityIndex": [30, 32, 31, 33, 25, 26, 45, 28]
}
df = pd.DataFrame(data)

# Optional: Convert 'Timestamp' column to datetime objects for better time-series handling
# This is good practice, though not strictly necessary for this specific outlier detection method.
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# 2. Calculate the mean and standard deviation of "TemperatureCelsius" for all readings
temp_mean = df['TemperatureCelsius'].mean()
temp_std = df['TemperatureCelsius'].std()

print(f"Global Mean Temperature: {temp_mean:.2f} °C")
print(f"Global Standard Deviation of Temperature: {temp_std:.2f} °C")
print("-" * 40)

# Define the threshold for outlier detection (1.5 standard deviations)
threshold_multiplier = 1.5
upper_bound = temp_mean + threshold_multiplier * temp_std
lower_bound = temp_mean - threshold_multiplier * temp_std

print(f"Outlier Upper Bound (Mean + {threshold_multiplier}*STD): {upper_bound:.2f} °C")
print(f"Outlier Lower Bound (Mean - {threshold_multiplier}*STD): {lower_bound:.2f} °C")
print("-" * 40)

# 3. Identify any readings where "TemperatureCelsius" is more than 1.5 standard deviations away from the mean
# This means temperatures are either above the upper_bound OR below the lower_bound.
outliers_df = df[
    (df['TemperatureCelsius'] > upper_bound) |
    (df['TemperatureCelsius'] < lower_bound)
]

# 4. For each such outlier, print the "SensorID", "Timestamp", and the anomalous "TemperatureCelsius" reading.
if not outliers_df.empty:
    print("Identified Temperature Outliers:")
    print("-" * 40)
    for index, row in outliers_df.iterrows():
        print(f"SensorID: {row['SensorID']}, "
              f"Timestamp: {row['Timestamp'].strftime('%Y-%m-%d %H:%M')}, "
              f"Temperature: {row['TemperatureCelsius']:.1f} °C (Outlier)")
else:
    print("No temperature outliers found based on the 1.5 standard deviation rule.")

```