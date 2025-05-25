```python
import pandas as pd

# 1. Define the pandas DataFrame representing social media post engagement
# Define the data as a dictionary
data = {
    "PostID": ["SM01", "SM02", "SM03", "SM04", "SM05"],
    "Platform": ["Instagram", "Twitter", "Facebook", "Instagram", "Twitter"],
    "PostType": ["Image", "Text", "Video", "Image", "Text"],
    "Likes": [150, 80, 300, 200, 90],
    "Shares": [20, 15, 50, 30, 10],
    "Comments": [10, 5, 25, 15, 8]
}

# Create the DataFrame from the dictionary
df = pd.DataFrame(data)

# Print the DataFrame to verify its creation (optional, but good for debugging)
# print("Original DataFrame:")
# print(df)
# print("-" * 30)

# 2. Calculate and print the total number of "Likes" across all posts
# Sum the 'Likes' column to get the total
total_likes = df["Likes"].sum()

# Print the result
print(f"Total number of Likes across all posts: {total_likes}")

# 3. Find and print the "PostType" that has the highest average number of "Shares"
# Group the DataFrame by 'PostType' and calculate the mean of 'Shares' for each group
average_shares_by_post_type = df.groupby("PostType")["Shares"].mean()

# Find the PostType with the maximum average shares
# idxmax() returns the index (PostType in this case) of the first occurrence of the maximum value
post_type_highest_avg_shares = average_shares_by_post_type.idxmax()

# Print the result
print(f"PostType with the highest average number of Shares: {post_type_highest_avg_shares}")

# Optional: Print the average shares for all post types to verify
# print("\nAverage Shares by PostType:")
# print(average_shares_by_post_type)
```