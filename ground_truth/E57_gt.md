```python
import pandas as pd

# Define the DataFrame as specified in the task prompt
# This dictionary holds the data for movie attributes.
data = {
    "MovieID": ["M01", "M02", "M03", "M04", "M05"],
    "Title": ["Inception", "The Dark Knight", "Pulp Fiction", "Forrest Gump", "The Matrix"],
    "Genre": ["Sci-Fi", "Action", "Crime", "Drama", "Sci-Fi"],
    "Year": [2010, 2008, 1994, 1994, 1999],
    "Rating": [8.8, 9.0, 8.9, 8.8, 8.7],
    "Votes": [2200000, 2500000, 1900000, 1800000, 1700000]
}

# Create the pandas DataFrame from the defined data.
df = pd.DataFrame(data)

# --- Task 1: Print the "Title" of the movie with the highest "Rating" ---

# Find the maximum rating value in the 'Rating' column.
max_rating = df['Rating'].max()

# Filter the DataFrame to get rows where the 'Rating' is equal to the maximum rating.
# This handles potential edge cases where multiple movies might have the same highest rating.
highest_rated_movies = df[df['Rating'] == max_rating]

# If there are multiple movies with the same highest rating, we'll just take the title
# of the first one found. If the requirement was to list all, a loop would be used.
# For this task, "the 'Title' of the movie" implies one.
if not highest_rated_movies.empty:
    # Get the title of the first movie in the filtered DataFrame.
    title_highest_rating = highest_rated_movies['Title'].iloc[0]
    print(f"The title of the movie with the highest rating is: '{title_highest_rating}'")
else:
    # This case should ideally not happen with a non-empty DataFrame,
    # but it's good practice for robustness.
    print("No movies found in the DataFrame to determine the highest rating.")

# --- Task 2: Print the average "Year" of all movies in the list ---

# Calculate the average (mean) of the 'Year' column.
# .mean() automatically handles numerical columns and returns the average.
average_year = df['Year'].mean()

# Print the calculated average year, formatted to two decimal places for clarity.
print(f"The average year of all movies is: {average_year:.2f}")

```