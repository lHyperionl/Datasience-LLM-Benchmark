```python
import pandas as pd
import numpy as np

# Define the path to the dataset
DATASET_PATH = 'video_games_sales.csv'

# Define the current year for 'Years_Since_Release' calculation
CURRENT_YEAR = 2023

try:
    # 1) Load the dataset
    df = pd.read_csv(DATASET_PATH)
    print(f"Original dataset shape: {df.shape}")

    # Convert 'Year' column to numeric, coercing errors to NaN
    # This handles cases where 'Year' might contain non-numeric strings
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')

    # 2) Handle missing values in the 'Year' column by filling them with the median year
    # Calculate the median of the 'Year' column
    median_year = df['Year'].median()
    df['Year'].fillna(median_year, inplace=True)
    # Convert 'Year' column to integer type after filling NaNs
    df['Year'] = df['Year'].astype(int)
    print(f"Shape after handling missing 'Year' values: {df.shape}")

    # 3) Remove games with missing Publisher information
    df.dropna(subset=['Publisher'], inplace=True)
    print(f"Shape after removing games with missing 'Publisher': {df.shape}")

    # 4) Create a new column 'Years_Since_Release'
    # Calculate how many years have passed since each game's release year
    df['Years_Since_Release'] = CURRENT_YEAR - df['Year']
    # Ensure 'Years_Since_Release' is non-negative, setting negative values to 0 if any
    df['Years_Since_Release'] = df['Years_Since_Release'].apply(lambda x: max(0, x))
    print(f"Shape after creating 'Years_Since_Release' column: {df.shape}")

    # 5) Remove duplicate games (based on Name and Platform)
    # Keeping the first occurrence
    df.drop_duplicates(subset=['Name', 'Platform'], inplace=True)
    print(f"Shape after removing duplicate games: {df.shape}")

    # Convert 'Global_Sales' to numeric, coercing errors to NaN, before filtering
    df['Global_Sales'] = pd.to_numeric(df['Global_Sales'], errors='coerce')
    # Drop rows where 'Global_Sales' became NaN after conversion
    df.dropna(subset=['Global_Sales'], inplace=True)

    # 6) Filter out games with Global_Sales less than 0.1 million
    df = df[df['Global_Sales'] >= 0.1]
    print(f"Shape after filtering out low-selling games: {df.shape}")

    # 7) Display the cleaned dataset's shape and first 5 rows
    print("\nCleaned dataset shape:")
    print(df.shape)
    print("\nFirst 5 rows of the cleaned dataset:")
    print(df.head())

except FileNotFoundError:
    print(f"Error: The file '{DATASET_PATH}' was not found. Please ensure it's in the current directory.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

```