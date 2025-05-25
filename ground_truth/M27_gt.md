```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
# Define the path to the dataset. Assumes 'iris.csv' is in the current directory.
DATASET_PATH = 'iris.csv'

# Define the numerical features for which box plots will be generated.
NUMERICAL_FEATURES = [
    'SepalLengthCm',
    'SepalWidthCm',
    'PetalLengthCm',
    'PetalWidthCm'
]

# Define the column to group the box plots by.
GROUPING_COLUMN = 'Species'

# --- Data Loading ---
try:
    # Load the iris dataset from the specified CSV file.
    df = pd.read_csv(DATASET_PATH)
except FileNotFoundError:
    print(f"Error: The file '{DATASET_PATH}' was not found.")
    print("Please ensure 'iris.csv' is in the same directory as the script.")
    exit()
except Exception as e:
    print(f"An error occurred while loading the dataset: {e}")
    exit()

# --- Box Plot Generation ---
# Iterate through each numerical feature to create a separate box plot.
for feature in NUMERICAL_FEATURES:
    # Create a new figure for each plot to ensure they are displayed independently.
    # Set a specific figure size for better readability.
    plt.figure(figsize=(8, 6))

    # Generate the box plot using Seaborn.
    # 'x' axis represents the grouping column (Species).
    # 'y' axis represents the current numerical feature.
    # 'data' specifies the DataFrame to use.
    sns.boxplot(x=GROUPING_COLUMN, y=feature, data=df)

    # Set the title of the plot to clearly indicate the feature being displayed.
    plt.title(f'Distribution of {feature} by {GROUPING_COLUMN}', fontsize=14)

    # Add labels for the x and y axes for clarity.
    plt.xlabel(GROUPING_COLUMN, fontsize=12)
    plt.ylabel(feature, fontsize=12)

    # Improve plot aesthetics:
    # Add a grid to the y-axis for easier value comparison.
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    # Adjust plot layout to prevent labels from overlapping.
    plt.tight_layout()

    # Display the generated plot.
    plt.show()

# End of script. All plots have been generated and displayed.
```