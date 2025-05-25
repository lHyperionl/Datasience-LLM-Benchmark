```python
import pandas as pd

# Define the movie ratings DataFrame
# This dictionary holds the data for each column.
data = {
    "MovieID": ["M01", "M02", "M03", "M04", "M05", "M06"],
    "Title": ["Inception", "The Dark Knight", "Pulp Fiction", "Forrest Gump", "The Matrix", "Interstellar"],
    "Genre": ["Sci-Fi", "Action", "Crime", "Drama", "Sci-Fi", "Sci-Fi"],
    "Year": [2010, 2008, 1994, 1994, 1999, 2014],
    "Rating": [8.8, 9.0, 8.9, 8.8, 8.7, 8.6],
    "Votes": [2200000, 2500000, 1900000, 1800000, 1700000, 1600000]
}

# Create the pandas DataFrame from the dictionary
df = pd.DataFrame(data)

# --- Task Operations ---

# 1) Filter the DataFrame to include only movies with a "Genre" of "Sci-Fi".
# We use boolean indexing: df['Genre'] == 'Sci-Fi' creates a boolean Series,
# which is then used to select rows from the DataFrame.
sci_fi_movies_df = df[df['Genre'] == 'Sci-Fi']

# Handle edge case: If no Sci-Fi movies are found, the DataFrame will be empty.
# In this specific dataset, Sci-Fi movies exist, so this check is for robustness.
if sci_fi_movies_df.empty:
    print("No Sci-Fi movies found in the DataFrame.")
    average_sci_fi_rating = 0.0 # Or pd.NA, depending on desired behavior for no data
else:
    # 2) For these Sci-Fi movies, calculate the average "Rating".
    # We select the 'Rating' column from the filtered DataFrame and apply the .mean() method.
    average_sci_fi_rating = sci_fi_movies_df['Rating'].mean()

# Print the calculated average rating for Sci-Fi movies.
# Using an f-string for clear and formatted output.
print(f"The average rating for Sci-Fi movies is: {average_sci_fi_rating:.2f}")

# Optional: Display the filtered Sci-Fi movies DataFrame for verification
# print("\nFiltered Sci-Fi Movies DataFrame:")
# print(sci_fi_movies_df)
```