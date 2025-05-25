```python
import pandas as pd

# Define the path to the dataset
DATASET_PATH = 'imdb_top_250.csv'

try:
    # 1. Load the imdb_top_250.csv dataset into a pandas DataFrame
    df = pd.read_csv(DATASET_PATH)

    # 2. Filter the dataset to include only movies with a 'Rating' greater than or equal to 9.0
    # This creates a boolean mask and applies it to the DataFrame
    high_rated_movies = df[df['Rating'] >= 9.0]

    # 3. From this filtered dataset, select the 'Title' and 'Rating' columns
    # This creates a new DataFrame containing only these two columns
    selected_columns = high_rated_movies[['Title', 'Rating']]

    # 4. Sort the resulting data by 'Rating' in descending order
    # The 'ascending=False' argument ensures descending order
    # .copy() is used to avoid SettingWithCopyWarning in case of chained operations later,
    # though not strictly necessary for this final display step.
    sorted_movies = selected_columns.sort_values(by='Rating', ascending=False).copy()

    # Display the final DataFrame
    print(sorted_movies)

except FileNotFoundError:
    # Handle the case where the dataset file is not found
    print(f"Error: The file '{DATASET_PATH}' was not found.")
    print("Please ensure the CSV file is in the same directory as the script.")
except KeyError as e:
    # Handle cases where expected columns are missing
    print(f"Error: A required column was not found in the dataset. Missing column: {e}")
    print("Please check if 'Rating' and 'Title' columns exist in the CSV file.")
except Exception as e:
    # Catch any other unexpected errors during processing
    print(f"An unexpected error occurred: {e}")
```