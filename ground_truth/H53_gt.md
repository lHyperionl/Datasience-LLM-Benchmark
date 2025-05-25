```python
import pandas as pd
import numpy as np

# Define the initial DataFrame as specified in the task
data = {
    "UserID": ["User1", "User2", "User3", "User4", "User5", "User6"],
    "SessionDurationMinutes": [15, 25, 10, 30, 22, 18],
    "PagesViewed": [5, 8, 3, 10, 7, 6],
    "DeviceType": ["Desktop", "Mobile", "Desktop", "Mobile", "Desktop", "Mobile"],
    "Converted": [True, False, False, True, True, False]
}
df = pd.DataFrame(data)

# --- Data Preparation for Classification Model ---

# 1) Create a new numerical feature "DeviceType_Numeric"
# Map "Desktop" to 0 and "Mobile" to 1.
# Using a dictionary for mapping ensures explicit control over values.
device_type_mapping = {"Desktop": 0, "Mobile": 1}
df["DeviceType_Numeric"] = df["DeviceType"].map(device_type_mapping)

# 2) Create another new feature "PagesPerMinute"
# Calculated as "PagesViewed" / "SessionDurationMinutes".
# Handle any potential division by zero by replacing resulting NaNs or Infs with 0.

# Perform the division. This might result in np.inf (for x/0 where x>0) or np.nan (for 0/0).
df["PagesPerMinute"] = df["PagesViewed"] / df["SessionDurationMinutes"]

# Replace positive and negative infinities with NaN first.
# This ensures that all problematic division results (inf, -inf, nan from 0/0) are NaNs.
df["PagesPerMinute"].replace([np.inf, -np.inf], np.nan, inplace=True)

# Fill any remaining NaNs (which now include original NaNs and converted infinities) with 0.
df["PagesPerMinute"].fillna(0, inplace=True)

# 3) Select the specified features for the model
# "SessionDurationMinutes", "PagesViewed", "DeviceType_Numeric", and "PagesPerMinute"
feature_columns = [
    "SessionDurationMinutes",
    "PagesViewed",
    "DeviceType_Numeric",
    "PagesPerMinute"
]
X = df[feature_columns]

# 4) Print the head of this new feature DataFrame
print("Head of the prepared feature DataFrame (X):")
print(X.head())
```