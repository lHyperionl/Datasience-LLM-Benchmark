```python
import pandas as pd

# --- Step 1: Load the dataset ---
# Define the path to the dataset.
# Assume 'penguins.csv' is in the current working directory.
try:
    df = pd.read_csv('penguins.csv')
    print("Dataset 'penguins.csv' loaded successfully.")
except FileNotFoundError:
    print("Error: 'penguins.csv' not found. Please ensure the file is in the current directory.")
    exit() # Exit if the file is not found, as subsequent steps depend on it.
except Exception as e:
    print(f"An error occurred while loading the dataset: {e}")
    exit()

# --- Step 2: Filter the data to include only penguins from the 'Torgersen' island ---
# Check if 'island' column exists to prevent KeyError.
if 'island' not in df.columns:
    print("Error: 'island' column not found in the dataset. Cannot filter by island.")
    exit()

# Filter the DataFrame where the 'island' column is 'Torgersen'.
# .copy() is used to avoid SettingWithCopyWarning in future operations,
# though not strictly necessary for this specific task.
torgersen_island_df = df[df['island'] == 'Torgersen'].copy()

# Handle the edge case where no penguins are found on 'Torgersen' island.
if torgersen_island_df.empty:
    print("\nNo penguins found on 'Torgersen' island in the dataset.")
else:
    print(f"\nFiltered data for 'Torgersen' island. Found {len(torgersen_island_df)} penguins.")

    # --- Step 3: Count and print the number of penguins belonging to each species ---
    # Check if 'species' column exists to prevent KeyError.
    if 'species' not in torgersen_island_df.columns:
        print("Error: 'species' column not found in the filtered dataset. Cannot count species.")
    else:
        # Use .value_counts() to count occurrences of each unique value in the 'species' column.
        species_counts = torgersen_island_df['species'].value_counts()

        # Print the results in a clear format.
        print("\n--- Species count for penguins on 'Torgersen' island ---")
        print(species_counts)
        print("-----------------------------------------------------")

```