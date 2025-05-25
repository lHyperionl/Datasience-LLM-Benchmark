```python
import pandas as pd
import matplotlib.pyplot as plt

# --- Configuration ---
# Define the path to the dataset
DATASET_PATH = 'spotify_2023.csv'

# Define column names for plotting
BPM_COLUMN = 'bpm'
DANCEABILITY_COLUMN = 'danceability_%'
ENERGY_COLUMN = 'energy_%'

# --- Data Loading ---
try:
    # Load the dataset into a pandas DataFrame
    df = pd.read_csv(DATASET_PATH)
    print(f"Successfully loaded '{DATASET_PATH}'. Shape: {df.shape}")
except FileNotFoundError:
    print(f"Error: The file '{DATASET_PATH}' was not found.")
    print("Please ensure the CSV file is in the same directory as the script.")
    exit()
except Exception as e:
    print(f"An error occurred while loading the dataset: {e}")
    exit()

# --- Data Visualization ---

# 1. Generate a histogram for the 'bpm' column
# Create a new figure for the histogram to ensure it's a separate plot
plt.figure(figsize=(10, 6))
try:
    # Plot the histogram of the 'bpm' column
    # Using bins='auto' lets matplotlib determine an appropriate number of bins
    plt.hist(df[BPM_COLUMN], bins='auto', edgecolor='black', alpha=0.7)

    # Add title and labels for clarity
    plt.title(f'Distribution of {BPM_COLUMN.upper()} in Spotify 2023 Tracks')
    plt.xlabel(f'{BPM_COLUMN.upper()} (Beats Per Minute)')
    plt.ylabel('Number of Tracks')
    plt.grid(axis='y', alpha=0.75) # Add a grid for better readability
except KeyError:
    print(f"Error: Column '{BPM_COLUMN}' not found in the dataset. Please check the column name.")
except Exception as e:
    print(f"An error occurred while plotting the {BPM_COLUMN} histogram: {e}")

# 2. Create a scatter plot for 'danceability_%' vs 'energy_%'
# Create another new figure for the scatter plot
plt.figure(figsize=(10, 6))
try:
    # Plot the scatter plot
    # x-axis: danceability_%, y-axis: energy_%
    plt.scatter(df[DANCEABILITY_COLUMN], df[ENERGY_COLUMN], alpha=0.6, s=50) # s is marker size

    # Add title and labels for clarity
    plt.title(f'Relationship between {DANCEABILITY_COLUMN} and {ENERGY_COLUMN}')
    plt.xlabel(f'{DANCEABILITY_COLUMN} (Percentage)')
    plt.ylabel(f'{ENERGY_COLUMN} (Percentage)')
    plt.grid(True, linestyle='--', alpha=0.6) # Add a grid for better readability
except KeyError:
    print(f"Error: One or both columns ('{DANCEABILITY_COLUMN}', '{ENERGY_COLUMN}') not found in the dataset. Please check column names.")
except Exception as e:
    print(f"An error occurred while plotting the scatter plot: {e}")

# Adjust layout to prevent labels from overlapping
plt.tight_layout()

# Display all generated plots
plt.show()
```