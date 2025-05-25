```python
import pandas as pd
import matplotlib.pyplot as plt

# --- Configuration ---
DATASET_PATH = 'youtube_trendings.csv'
VIDEO_VIEWS_COL = 'Video views'
LIKES_COL = 'Likes'
PLOT_TITLE = 'Video Views vs. Likes'
X_LABEL = 'Video Views (cleaned)'
Y_LABEL = 'Likes (cleaned)'

# --- 1. Load the dataset ---
try:
    df = pd.read_csv(DATASET_PATH)
    print(f"Dataset '{DATASET_PATH}' loaded successfully.")
except FileNotFoundError:
    print(f"Error: The file '{DATASET_PATH}' was not found. Please ensure it's in the current directory.")
    exit()
except Exception as e:
    print(f"An error occurred while loading the dataset: {e}")
    exit()

# --- 2. Clean 'Likes' and 'Video views' columns ---

# Define a list of columns to clean
columns_to_clean = [LIKES_COL, VIDEO_VIEWS_COL]

for col in columns_to_clean:
    if col in df.columns:
        print(f"Cleaning column: '{col}'...")
        # Remove commas from the string representation of the column
        # .astype(str) ensures that even if there are numeric types, they are treated as strings for replacement
        df[col] = df[col].astype(str).str.replace(',', '', regex=False)

        # Convert to numeric type. 'errors='coerce'' will turn any non-convertible values into NaN.
        df[col] = pd.to_numeric(df[col], errors='coerce')

        # Fill any resulting NaN values with 0
        df[col] = df[col].fillna(0)
        print(f"Column '{col}' cleaned and NaNs filled with 0.")
    else:
        print(f"Warning: Column '{col}' not found in the dataset. Skipping cleaning for this column.")

# --- 3. Create a scatter plot ---

# Ensure the cleaned columns exist before plotting
if VIDEO_VIEWS_COL in df.columns and LIKES_COL in df.columns:
    plt.figure(figsize=(10, 6)) # Set figure size for better readability
    plt.scatter(df[VIDEO_VIEWS_COL], df[LIKES_COL], alpha=0.6, s=10) # s is marker size, alpha for transparency

    # Add plot title and labels
    plt.title(PLOT_TITLE)
    plt.xlabel(X_LABEL)
    plt.ylabel(Y_LABEL)

    # Add a grid for better readability
    plt.grid(True, linestyle='--', alpha=0.7)

    # Display the plot
    plt.tight_layout() # Adjust layout to prevent labels from overlapping
    plt.show()
    print("Scatter plot generated successfully.")
else:
    print("Cannot generate plot: One or both of the required columns ('Video views', 'Likes') were not found or processed correctly.")
```