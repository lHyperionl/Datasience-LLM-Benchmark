```python
import pandas as pd

# 1. Define the pandas DataFrame
# Define the data as specified in the task prompt
data = {
    "Timestamp": ["2025-05-24 10:00", "2025-05-24 10:05", "2025-05-24 10:10", "2025-05-24 10:15", "2025-05-24 10:00", "2025-05-24 10:05"],
    "SensorID": ["S1", "S1", "S1", "S1", "S2", "S2"],
    "TemperatureCelsius": [22.5, 22.7, 22.6, 22.8, 21.9, 22.0],
    "HumidityPercent": [55, 56, 55, 57, 60, 59],
    "AirQualityIndex": [30, 32, 31, 33, 25, 26]
}

# Create the DataFrame from the defined data
df = pd.DataFrame(data)

# Display the created DataFrame (optional, for verification)
print("Original DataFrame:")
print(df)
print("-" * 30)

# 2. Calculate and print the average "AirQualityIndex" for each unique "SensorID"

# Group the DataFrame by 'SensorID'
# Then select the 'AirQualityIndex' column from the grouped object
# Finally, calculate the mean for each group
average_aqi_per_sensor = df.groupby('SensorID')['AirQualityIndex'].mean()

# Print the results clearly
print("Average AirQualityIndex per SensorID:")
# Iterate through the resulting Series to print each sensor's average clearly
for sensor_id, avg_aqi in average_aqi_per_sensor.items():
    print(f"  Sensor {sensor_id}: {avg_aqi:.2f}") # Format to two decimal places for clarity

# Example of handling a potential edge case (though not strictly required by prompt, good practice):
# If 'AirQualityIndex' column might contain non-numeric values or NaNs,
# .mean() would typically ignore NaNs by default.
# If a SensorID had no 'AirQualityIndex' data, it wouldn't appear in the groupby result.
# For this specific dataset, all values are numeric and present.
```