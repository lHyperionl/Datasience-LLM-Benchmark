```python
import pandas as pd

# 1. Define the pandas DataFrame representing Pokemon stats
# The data dictionary contains the raw data for the DataFrame.
data = {
    "PokemonID": [1, 2, 3, 4, 5, 6, 7, 8, 9],
    "Name": ["Bulbasaur", "Ivysaur", "Venusaur", "Charmander", "Charmeleon", "Charizard", "Squirtle", "Wartortle", "Blastoise"],
    "Type1": ["Grass", "Grass", "Grass", "Fire", "Fire", "Fire", "Water", "Water", "Water"],
    "Type2": ["Poison", "Poison", "Poison", None, None, "Flying", None, None, None],
    "HP": [45, 60, 80, 39, 58, 78, 44, 59, 79],
    "Attack": [49, 62, 82, 52, 64, 84, 48, 63, 83],
    "Defense": [49, 63, 83, 43, 58, 78, 65, 80, 100],
    "Speed": [45, 60, 80, 65, 80, 100, 43, 58, 78]
}

# Create the DataFrame from the dictionary
df = pd.DataFrame(data)

print("Original DataFrame:")
print(df)
print("-" * 30)

# 2. Add a new column "TotalStats" which is the sum of "HP", "Attack", "Defense", and "Speed".
# We select the relevant columns and sum them row-wise (axis=1).
df["TotalStats"] = df["HP"] + df["Attack"] + df["Defense"] + df["Speed"]

print("\nDataFrame after adding 'TotalStats' column:")
print(df)
print("-" * 30)

# 3. Identify and print the "Name" of the Pokemon with the highest "TotalStats".
# First, find the maximum value in the "TotalStats" column.
max_total_stats = df["TotalStats"].max()

# Then, find the row(s) where "TotalStats" equals this maximum.
# Using idxmax() returns the index of the first occurrence of the maximum value.
# We then use .loc to get the 'Name' at that specific index.
pokemon_with_highest_total_stats = df.loc[df["TotalStats"].idxmax(), "Name"]

print(f"\nPokemon with the highest 'TotalStats': {pokemon_with_highest_total_stats}")
print(f"Highest 'TotalStats' value: {max_total_stats}")
print("-" * 30)

# 4. For each "Type1", calculate and print the average "Speed".
# Group the DataFrame by the "Type1" column.
# Then, select the "Speed" column and calculate the mean for each group.
average_speed_by_type1 = df.groupby("Type1")["Speed"].mean()

print("\nAverage 'Speed' for each 'Type1':")
print(average_speed_by_type1)
print("-" * 30)
```