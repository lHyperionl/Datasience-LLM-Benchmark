```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
DATASET_PATH = 'penguins.csv'
TARGET_COLUMN = 'body_mass_g'
GROUPING_COLUMN = 'species'

# --- 1. Load the dataset ---
try:
    df = pd.read_csv(DATASET_PATH)
    print(f"Dataset '{DATASET_PATH}' loaded successfully. Initial shape: {df.shape}")
except FileNotFoundError:
    print(f"Error: The file '{DATASET_PATH}' was not found.")
    print("Please ensure 'penguins.csv' is in the same directory as the script.")
    exit() # Exit if the dataset cannot be loaded

# --- 2. Drop rows where 'body_mass_g' or 'species' is NA ---
# Identify columns critical for this analysis
columns_to_check_na = [TARGET_COLUMN, GROUPING_COLUMN]

# Get initial NA counts for relevant columns
initial_na_counts = df[columns_to_check_na].isnull().sum()
print("\nInitial NA counts for critical columns:")
print(initial_na_counts)

# Drop rows where any of the specified columns have NA values
df_cleaned = df.dropna(subset=columns_to_check_na)

print(f"\nRows dropped due to NA in '{TARGET_COLUMN}' or '{GROUPING_COLUMN}'.")
print(f"New dataset shape after dropping NAs: {df_cleaned.shape}")

# Verify NA counts after cleaning
final_na_counts = df_cleaned[columns_to_check_na].isnull().sum()
print("\nFinal NA counts for critical columns after cleaning:")
print(final_na_counts)

# --- 3. For each species, calculate the average 'body_mass_g' ---
# Group by 'species' and calculate the mean of 'body_mass_g'
# .reset_index() converts the 'species' index back into a column
average_body_mass_per_species = df_cleaned.groupby(GROUPING_COLUMN)[TARGET_COLUMN].mean().reset_index()

print(f"\nAverage '{TARGET_COLUMN}' per '{GROUPING_COLUMN}':")
print(average_body_mass_per_species)

# --- 4. Create a bar plot showing the average 'body_mass_g' for each species ---
plt.figure(figsize=(10, 6)) # Set the figure size for better readability

# Create the bar plot using seaborn
sns.barplot(
    x=GROUPING_COLUMN,
    y=TARGET_COLUMN,
    data=average_body_mass_per_species,
    palette='viridis' # Choose a color palette
)

# Add plot title and labels
plt.title(f'Average {TARGET_COLUMN.replace("_", " ").title()} per {GROUPING_COLUMN.title()}', fontsize=16)
plt.xlabel(GROUPING_COLUMN.title(), fontsize=12)
plt.ylabel(f'Average {TARGET_COLUMN.replace("_", " ").title()}', fontsize=12)

# Rotate x-axis labels for better readability if species names are long
plt.xticks(rotation=45, ha='right')

# Add grid for better readability of values
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Ensure all elements fit within the figure area
plt.tight_layout()

# Display the plot
plt.show()
```