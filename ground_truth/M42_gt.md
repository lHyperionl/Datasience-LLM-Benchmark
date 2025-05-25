```python
import pandas as pd

# Define the path to the dataset
DATASET_PATH = 'imdb_top_250.csv'

def perform_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs feature engineering on the DataFrame, specifically creating a 'Decade' column.

    Args:
        df (pd.DataFrame): The input DataFrame containing 'Year' and 'Title' columns.

    Returns:
        pd.DataFrame: The DataFrame with the new 'Decade' column added.
    """
    # Ensure 'Year' column is numeric, coercing errors to NaN
    # This handles potential non-numeric entries gracefully, though for this dataset, it's likely clean.
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')

    # Drop rows where 'Year' might have become NaN after coercion
    df.dropna(subset=['Year'], inplace=True)

    # Convert 'Year' to integer type for floor division
    df['Year'] = df['Year'].astype(int)

    # Create the 'Decade' column
    # Calculate the start year of the decade (e.g., 1995 -> 1990)
    # Convert to string and append 's'
    df['Decade'] = ((df['Year'] // 10) * 10).astype(str) + 's'

    return df

def main():
    """
    Main function to load data, perform feature engineering, and display results.
    """
    try:
        # 1. Load the dataset
        df = pd.read_csv(DATASET_PATH)
        print(f"Dataset '{DATASET_PATH}' loaded successfully. Shape: {df.shape}\n")

        # Perform feature engineering
        df_engineered = perform_feature_engineering(df.copy()) # Use .copy() to avoid modifying original df if needed elsewhere

        # 3. Display the 'Title', 'Year', and the new 'Decade' column for the first 10 movies
        print("First 10 movies with 'Title', 'Year', and 'Decade' columns:\n")
        display_columns = ['Title', 'Year', 'Decade']
        print(df_engineered[display_columns].head(10).to_string(index=False))

    except FileNotFoundError:
        print(f"Error: The file '{DATASET_PATH}' was not found.")
        print("Please ensure the CSV file is in the same directory as the script.")
    except KeyError as e:
        print(f"Error: Missing expected column in the dataset: {e}")
        print("Please check if 'Year' and 'Title' columns exist in the CSV.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
```