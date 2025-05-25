```python
import pandas as pd

# Define the data for the DataFrame
data = {
    "UserID": ["User1", "User2", "User3", "User4", "User5", "User6"],
    "SessionDurationMinutes": [15, 25, 10, 30, 22, 18],
    "PagesViewed": [5, 8, 3, 10, 7, 6],
    "DeviceType": ["Desktop", "Mobile", "Desktop", "Mobile", "Desktop", "Mobile"],
    "Converted": [True, False, False, True, True, False]
}

# Create the pandas DataFrame
# pd.DataFrame automatically infers data types from the provided data.
df = pd.DataFrame(data)

# --- Task 1: Calculate and print the average "SessionDurationMinutes" ---

# Select the 'SessionDurationMinutes' column and calculate its mean.
# The .mean() method automatically handles numerical data and returns the average.
# If there were non-numeric values or NaNs, pandas would typically skip NaNs
# or raise an error for non-numeric, ensuring robust calculation.
average_session_duration = df["SessionDurationMinutes"].mean()

# Print the calculated average session duration.
print(f"1. Average Session Duration: {average_session_duration:.2f} minutes")
print("-" * 50) # Separator for clarity

# --- Task 2: Count and print the number of users for each "DeviceType" ---

# Count the occurrences of each unique value in the 'DeviceType' column.
# The .value_counts() method is ideal for categorical data, providing
# a Series with unique values as index and their counts as values,
# sorted in descending order by default.
# It automatically handles all unique categories present in the column.
device_type_counts = df["DeviceType"].value_counts()

# Print the counts for each device type.
print("2. Number of users per Device Type:")
print(device_type_counts)

# Example of handling a potential edge case: if 'DeviceType' had missing values (NaN),
# value_counts() by default excludes them. To include them, one would use:
# device_type_counts_with_nan = df["DeviceType"].value_counts(dropna=False)
# However, for this specific dataset, there are no NaNs in 'DeviceType'.
```