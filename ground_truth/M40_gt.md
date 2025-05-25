```python
import pandas as pd
import matplotlib.pyplot as plt
import re

# Define the path to the dataset
DATASET_PATH = 'imdb_top_250.csv'

def analyze_imdb_data(dataset_path: str):
    """
    Loads the imdb_top_250.csv dataset, groups by year, calculates average
    rating and runtime, and plots the trend of average rating over years.

    Args:
        dataset_path (str): The path to the imdb_top_250.csv file.
    """
    try:
        # 1. Load the imdb_top_250.csv dataset using pandas
        df = pd.read_csv(dataset_path)

        # --- Data Cleaning and Preprocessing ---
        # Ensure 'Year' column is numeric. If it contains non-numeric characters
        # (e.g., release year in parentheses), extract the year.
        # Assuming 'Year' is already clean or can be directly converted.
        # If 'Year' column contains non-numeric data like "(2000)", clean it.
        if df['Year'].dtype == 'object':
            df['Year'] = df['Year'].astype(str).str.extract(r'(\d{4})').astype(int)

        # Clean the 'Runtime' column: remove ' min' and convert to integer.
        # Handle potential missing values or non-standard formats gracefully.
        # Use a regex to extract numbers, then convert to numeric.
        df['Runtime'] = df['Runtime'].astype(str).apply(
            lambda x: int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else None
        )
        # Drop rows where Runtime could not be converted (e.g., NaN after conversion)
        df.dropna(subset=['Runtime'], inplace=True)

        # Ensure 'Rating' is numeric.
        df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce')
        # Drop rows where 'Rating' is NaN after coercion
        df.dropna(subset=['Rating'], inplace=True)

        # 2. Group the data by the `Year` column
        # 3. For each year, calculate the average `Rating` and the average `Runtime`
        yearly_stats = df.groupby('Year').agg(
            Average_Rating=('Rating', 'mean'),
            Average_Runtime=('Runtime', 'mean')
        ).reset_index() # reset_index to make 'Year' a column again for plotting

        # Sort by Year to ensure correct plotting order
        yearly_stats = yearly_stats.sort_values('Year')

        # 4. Create a line plot showing the trend of average `Rating` over the `Year`
        plt.figure(figsize=(12, 7)) # Set a larger figure size for better readability
        plt.plot(yearly_stats['Year'], yearly_stats['Average_Rating'], marker='o', linestyle='-')

        # Ensure the plot has a title and labeled axes
        plt.title('Average IMDb Rating Trend Over Years (Top 250 Movies)', fontsize=16)
        plt.xlabel('Year', fontsize=12)
        plt.ylabel('Average Rating', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7) # Add a grid for better readability
        plt.xticks(rotation=45) # Rotate x-axis labels for better visibility if years are dense
        plt.tight_layout() # Adjust layout to prevent labels from overlapping
        plt.show()

        # Optional: Print the calculated yearly statistics
        # print("\nYearly Average Rating and Runtime:")
        # print(yearly_stats.head())

    except FileNotFoundError:
        print(f"Error: The file '{dataset_path}' was not found. Please ensure it's in the current directory.")
    except KeyError as e:
        print(f"Error: Missing expected column in the dataset: {e}. Please check the CSV file structure.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# Execute the analysis function
if __name__ == "__main__":
    analyze_imdb_data(DATASET_PATH)
```