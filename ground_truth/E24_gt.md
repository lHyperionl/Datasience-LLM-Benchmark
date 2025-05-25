```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define the path to the dataset file
# Assumes 'clash_royale_cards.csv' is in the same directory as the script
file_path = 'clash_royale_cards.csv'

# --- 1. Load the clash_royale_cards.csv dataset using pandas ---
try:
    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv(file_path)
    print(f"Successfully loaded '{file_path}'.")
except FileNotFoundError:
    # Handle the case where the file does not exist
    print(f"Error: The file '{file_path}' was not found.")
    print("Please ensure the CSV file is in the current directory.")
    exit() # Exit the script if the file is not found
except Exception as e:
    # Handle any other potential errors during file loading
    print(f"An error occurred while loading the CSV file: {e}")
    exit() # Exit the script on other loading errors

# --- 2. Generate a histogram to visualize the distribution of the `elixirCost` column ---
# Ensure the 'elixirCost' column exists in the DataFrame
if 'elixirCost' not in df.columns:
    print("Error: The 'elixirCost' column was not found in the dataset.")
    print(f"Available columns are: {df.columns.tolist()}")
    exit() # Exit if the required column is missing

# Set up the matplotlib figure and axes for plotting
plt.figure(figsize=(10, 6)) # Set the figure size for better readability

# Use seaborn's histplot for a visually appealing histogram
# 'data' specifies the DataFrame, 'x' specifies the column for the x-axis
# 'bins' are set to cover typical elixir costs (1 to 10), ensuring each integer cost has its own bin
# 'kde=False' turns off the Kernel Density Estimate line, as elixir cost is discrete
# 'color' and 'edgecolor' enhance the plot's appearance
sns.histplot(data=df, x='elixirCost', bins=range(1, 12), kde=False, color='skyblue', edgecolor='black')

# --- 3. Ensure the plot has a title and labeled axes ---
# Add a descriptive title to the histogram
plt.title('Distribution of Elixir Cost for Clash Royale Cards', fontsize=16)

# Label the x-axis
plt.xlabel('Elixir Cost', fontsize=14)

# Label the y-axis
plt.ylabel('Number of Cards', fontsize=14)

# Set x-axis ticks to be integers, as elixir costs are discrete values
# This ensures clear labeling for each elixir cost
plt.xticks(range(1, 11), fontsize=12)
plt.yticks(fontsize=12)

# Add a grid to the plot for easier reading of values
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Display the plot
plt.show()
```