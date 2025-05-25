```python
import pandas as pd

# Define the path to the dataset file.
# Assumes 'clash_royale_cards.csv' is located in the current working directory.
file_path = 'clash_royale_cards.csv'

# Load the dataset into a pandas DataFrame.
# This step reads the CSV file, parsing its contents into a tabular data structure
# that can be easily manipulated with pandas.
df = pd.read_csv(file_path)

# Group the DataFrame by the 'rarity' column.
# For each unique rarity, calculate the average (mean) of the specified columns:
# 'elixirCost', 'Win Rate', and 'Usage'.
# The .mean() method is applied to the selected columns after grouping.
average_stats_by_rarity = df.groupby('rarity')[['elixirCost', 'Win Rate', 'Usage']].mean()

# Display the calculated average values for each card rarity.
# The resulting DataFrame will show 'rarity' as the index and the average
# 'elixirCost', 'Win Rate', and 'Usage' as columns for each rarity.
print(average_stats_by_rarity)
```