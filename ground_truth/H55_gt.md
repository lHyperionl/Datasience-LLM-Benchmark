```python
import pandas as pd

# Define the pandas DataFrame representing social media post engagement
# This dictionary holds the raw data for the DataFrame.
data = {
    "PostID": ["SM01", "SM02", "SM03", "SM04", "SM05", "SM06", "SM07", "SM08", "SM09"],
    "Platform": ["Instagram", "Twitter", "Facebook", "Instagram", "Twitter", "Instagram", "Facebook", "Twitter", "Facebook"],
    "PostType": ["Image", "Text", "Video", "Image", "Text", "Video", "Image", "Video", "Text"],
    "Likes": [150, 80, 300, 200, 90, 450, 120, 250, 70],
    "Shares": [20, 15, 50, 30, 10, 60, 18, 40, 12],
    "Comments": [10, 5, 25, 15, 8, 30, 9, 20, 6]
}

# Create the DataFrame from the defined data
df = pd.DataFrame(data)

# --- Task 1: Create a new feature "EngagementScore" ---
# EngagementScore is calculated as: Likes + (Shares * 2) + (Comments * 3)
df['EngagementScore'] = df['Likes'] + (df['Shares'] * 2) + (df['Comments'] * 3)

# --- Task 2: Determine which "PostType" has the highest average "EngagementScore" for each unique "Platform" ---

# Calculate the average EngagementScore for each unique combination of 'Platform' and 'PostType'.
# .groupby(['Platform', 'PostType']) creates groups based on these two columns.
# ['EngagementScore'].mean() calculates the mean of 'EngagementScore' for each group.
# .reset_index() converts the grouped result (which has a MultiIndex) back into a DataFrame
# with 'Platform', 'PostType', and 'EngagementScore' as regular columns.
avg_engagement_by_platform_posttype = df.groupby(['Platform', 'PostType'])['EngagementScore'].mean().reset_index()

# Find the PostType with the highest average EngagementScore for each Platform.
# 1. Sort the DataFrame:
#    - First by 'Platform' in ascending order to group platforms together.
#    - Then by 'EngagementScore' in descending order, so the highest score for each platform comes first.
# 2. Drop duplicates:
#    - .drop_duplicates(subset='Platform', keep='first') keeps only the first row for each unique 'Platform'.
#      Since we sorted by 'EngagementScore' in descending order, this first row will correspond to the
#      PostType with the highest average EngagementScore for that specific platform.
highest_avg_engagement_per_platform = avg_engagement_by_platform_posttype.sort_values(
    by=['Platform', 'EngagementScore'], ascending=[True, False]
).drop_duplicates(subset='Platform', keep='first')

# --- Task 3: Print the results ---
# Iterate through the DataFrame containing the highest average engagement scores per platform.
print("Highest Average EngagementScore by PostType for each Platform:")
for index, row in highest_avg_engagement_per_platform.iterrows():
    # Print the Platform, the best PostType, and its corresponding highest average EngagementScore.
    # Using an f-string for clear and formatted output.
    # {:.2f} formats the EngagementScore to two decimal places.
    print(
        f"Platform: {row['Platform']}, "
        f"Best PostType: {row['PostType']}, "
        f"Highest Average EngagementScore: {row['EngagementScore']:.2f}"
    )
```