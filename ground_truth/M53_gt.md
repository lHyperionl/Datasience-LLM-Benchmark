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
df = pd.DataFrame(data)

# --- Calculate Conversion Rates ---

# 1. Calculate the conversion rate for users on "Desktop"
# Filter the DataFrame to include only 'Desktop' users
desktop_users_df = df[df['DeviceType'] == 'Desktop']

# Calculate the conversion rate: (number of converted users / total users) * 100
# The .mean() method on a boolean Series directly calculates the proportion of True values.
# Multiplying by 100 converts it to a percentage.
if not desktop_users_df.empty:
    desktop_conversion_rate = desktop_users_df['Converted'].mean() * 100
else:
    # Handle the edge case where there are no desktop users to avoid NaN or errors
    desktop_conversion_rate = 0.0
    print("Warning: No 'Desktop' users found in the dataset.")


# 2. Calculate the conversion rate for users on "Mobile"
# Filter the DataFrame to include only 'Mobile' users
mobile_users_df = df[df['DeviceType'] == 'Mobile']

# Calculate the conversion rate for mobile users
if not mobile_users_df.empty:
    mobile_conversion_rate = mobile_users_df['Converted'].mean() * 100
else:
    # Handle the edge case where there are no mobile users
    mobile_conversion_rate = 0.0
    print("Warning: No 'Mobile' users found in the dataset.")

# --- Print Results ---

# Print both conversion rates, clearly labeled
print(f"Conversion Rate for Desktop users: {desktop_conversion_rate:.2f}%")
print(f"Conversion Rate for Mobile users: {mobile_conversion_rate:.2f}%")
```