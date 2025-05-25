```python
import pandas as pd

# Define the data for the DataFrame as specified in the task
data = {
    "Timestamp": ["2025-05-24 10:00", "2025-05-24 10:05", "2025-05-24 10:10", "2025-05-24 10:15", "2025-05-24 10:00", "2025-05-24 10:05"],
    "SensorID": ["S1", "S1", "S1", "S1", "S2", "S2"],
    "TemperatureCelsius": [22.5, 22.7, 22.6, 22.8, 21.9, 22.0],
    "HumidityPercent": [55, 56, 55, 57, 60, 59],
    "AirQualityIndex": [30, 32, 31, 33, 25, 26]
}

# Create the pandas DataFrame from the defined data
df = pd.DataFrame(data)

# --- Data Analysis ---

# 1. Find the maximum "TemperatureCelsius" recorded across all sensors.
# Access the 'TemperatureCelsius' column and use the .max() method.
max_temperature = df["TemperatureCelsius"].max()

# Print the maximum temperature.
print(f"Maximum TemperatureCelsius recorded: {max_temperature}°C")

# 2. Find the minimum "HumidityPercent" recorded across all sensors.
# Access the 'HumidityPercent' column and use the .min() method.
min_humidity = df["HumidityPercent"].min()

# Print the minimum humidity.
print(f"Minimum HumidityPercent recorded: {min_humidity}%")

# Edge case consideration: If the DataFrame or specific columns were empty,
# .max() and .min() would return NaN or raise an error depending on the pandas version
# and specific data types. For this specific task, the DataFrame is pre-defined
# with valid numerical data, so direct application is safe.
```