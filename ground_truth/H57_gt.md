```python
import pandas as pd

# 1. Define the pandas DataFrame representing movie ratings
# ---
# Data dictionary containing movie information
data = {
    "MovieID": ["M01", "M02", "M03", "M04", "M05", "M06", "M07", "M08"],
    "Title": ["Inception", "The Dark Knight", "Pulp Fiction", "Forrest Gump", "The Matrix", "Interstellar", "The Lion King", "Fight Club"],
    "Genre": ["Sci-Fi", "Action", "Crime", "Drama", "Sci-Fi", "Sci-Fi", "Animation", "Drama"],
    "Year": [2010, 2008, 1994, 1994, 1999, 2014, 1994, 1999],
    "Rating": [8.8, 9.0, 8.9, 8.8, 8.7, 8.6, 8.5, 8.8],
    "Votes": [2200000, 2500000, 1900000, 1800000, 1700000, 1600000, 900000, 2000000]
}

# Create the DataFrame from the data dictionary
df = pd.DataFrame(data)

# Print the initial DataFrame for verification (optional)
# print("Initial DataFrame:")
# print(df)
# print("\n" + "="*50 + "\n")

# 2. Calculate "WeightedRating" for each movie
# ---
# Calculate the sum of all votes. This will be the denominator for the WeightedRating.
# Handle potential division by zero if total_votes could be 0 (though not in this dataset).
total_votes = df["Votes"].sum()

# Check if total_votes is zero to prevent division by zero error
if total_votes == 0:
    print("Error: Total votes are zero, cannot calculate WeightedRating.")
    # Assign NaN or 0 to WeightedRating if total_votes is zero
    df["WeightedRating"] = 0.0
else:
    # Calculate WeightedRating using the formula: ("Rating" * "Votes") / (sum of all "Votes")
    # This operation is vectorized for efficiency.
    df["WeightedRating"] = (df["Rating"] * df["Votes"]) / total_votes

# 3. Add "WeightedRating" as a new column to the DataFrame
# ---
# This step is already completed by the assignment in the previous step:
# df["WeightedRating"] = ...

# Print the DataFrame with the new 'WeightedRating' column (optional)
# print("DataFrame with WeightedRating:")
# print(df)
# print("\n" + "="*50 + "\n")

# 4. Print the "Title" and "WeightedRating" of the movie with the highest "WeightedRating"
# ---
# Find the index of the row with the maximum 'WeightedRating'
# idxmax() returns the first index of the maximum value in case of ties.
idx_highest_weighted = df["WeightedRating"].idxmax()

# Retrieve the row corresponding to the highest weighted rating
movie_highest_weighted = df.loc[idx_highest_weighted]

# Extract the Title and WeightedRating from the identified movie
title_highest = movie_highest_weighted["Title"]
weighted_rating_highest = movie_highest_weighted["WeightedRating"]

# Print the results
print(f"Movie with the highest WeightedRating:")
print(f"Title: {title_highest}")
print(f"WeightedRating: {weighted_rating_highest:.6f}") # Format for better readability
```