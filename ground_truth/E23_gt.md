```python
import pandas as pd

# --- Configuration ---
# Define the path to the dataset file
DATASET_FILE = 'clash_royale_cards.csv'
# Define the column names for filtering and selection
ELIXIR_COST_COL = 'elixirCost'
CARD_NAME_COL = 'Card'
WIN_RATE_COL = 'Win Rate'
# Define the maximum elixir cost for filtering
MAX_ELIXIR_COST = 3

# --- Data Loading ---
try:
    # Load the dataset from the specified CSV file into a pandas DataFrame
    df = pd.read_csv(DATASET_FILE)
except FileNotFoundError:
    # Handle the case where the dataset file is not found
    print(f"Error: The file '{DATASET_FILE}' was not found. Please ensure it's in the current directory.")
    exit()
except Exception as e:
    # Handle other potential errors during file loading
    print(f"An error occurred while loading the dataset: {e}")
    exit()

# --- Data Filtering ---
# Filter the DataFrame to include only cards with 'elixirCost' less than or equal to MAX_ELIXIR_COST
# This creates a new DataFrame containing only the low-cost cards.
filtered_df = df[df[ELIXIR_COST_COL] <= MAX_ELIXIR_COST]

# --- Column Selection ---
# Select only the 'Card' name and 'Win Rate' columns from the filtered DataFrame.
# This creates a new DataFrame with only the desired columns.
# Ensure the columns exist before attempting to select them to prevent KeyError.
required_columns = [CARD_NAME_COL, WIN_RATE_COL]
if not all(col in filtered_df.columns for col in required_columns):
    print(f"Error: One or more required columns ({required_columns}) not found in the dataset.")
    exit()

selected_columns_df = filtered_df[required_columns]

# --- Data Sorting ---
# Sort the resulting DataFrame by 'Win Rate' in descending order.
# 'ascending=False' ensures the highest win rates appear first.
sorted_df = selected_columns_df.sort_values(by=WIN_RATE_COL, ascending=False)

# --- Display Results ---
# Display the final sorted DataFrame.
# This will print the 'Card' name and 'Win Rate' for cards with elixirCost <= 3,
# sorted by 'Win Rate' from highest to lowest.
print(sorted_df)
```