```python
import pandas as pd

# Define the path to the dataset
DATASET_PATH = 'youtube_trendings.csv'

try:
    # 1. Load the youtube_trendings.csv dataset
    df = pd.read_csv(DATASET_PATH)

    # 2. For 'Video views' and 'Likes' columns:
    #    a. Remove any commas
    #    b. Convert them to numeric types (e.g., float)
    #       If conversion fails for any value, treat that value as NaN
    #    c. Fill all resulting NaNs in these columns with 0

    # Process 'Video views' column
    if 'Video views' in df.columns:
        df['Video views'] = df['Video views'].astype(str).str.replace(',', '', regex=False)
        df['Video views'] = pd.to_numeric(df['Video views'], errors='coerce')
        df['Video views'] = df['Video views'].fillna(0)
    else:
        print(f"Warning: 'Video views' column not found in {DATASET_PATH}. Skipping processing for this column.")

    # Process 'Likes' column
    if 'Likes' in df.columns:
        df['Likes'] = df['Likes'].astype(str).str.replace(',', '', regex=False)
        df['Likes'] = pd.to_numeric(df['Likes'], errors='coerce')
        df['Likes'] = df['Likes'].fillna(0)
    else:
        print(f"Warning: 'Likes' column not found in {DATASET_PATH}. Skipping processing for this column.")

    # 3. Fill any missing values in the 'Category' column with the string 'Unknown'
    if 'Category' in df.columns:
        df['Category'] = df['Category'].fillna('Unknown')
    else:
        print(f"Warning: 'Category' column not found in {DATASET_PATH}. Skipping processing for this column.")

    # Ensure the columns exist before grouping and aggregating
    columns_to_aggregate = []
    if 'Video views' in df.columns:
        columns_to_aggregate.append('Video views')
    if 'Likes' in df.columns:
        columns_to_aggregate.append('Likes')

    if 'Category' in df.columns and columns_to_aggregate:
        # 4. Group the data by the 'Category' column
        # 5. For each category, calculate and display the sum of cleaned 'Video views' and the sum of cleaned 'Likes'
        category_summary = df.groupby('Category')[columns_to_aggregate].sum()

        # Display the results
        print("Sum of Video Views and Likes per Category:")
        print(category_summary)
    elif not columns_to_aggregate:
        print("Error: Neither 'Video views' nor 'Likes' columns were found. Cannot perform aggregation.")
    else:
        print("Error: 'Category' column not found. Cannot perform aggregation.")

except FileNotFoundError:
    print(f"Error: The file '{DATASET_PATH}' was not found. Please ensure it is in the current directory.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
```