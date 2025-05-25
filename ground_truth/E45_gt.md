```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_penguins_culmen_data(file_path='penguins.csv'):
    """
    Loads the penguins dataset, filters out rows with missing culmen or species data,
    and generates a scatter plot of culmen length vs. culmen depth, colored by species.

    Args:
        file_path (str): The path to the penguins.csv dataset file.
    """
    try:
        # 1. Load the penguins.csv dataset using pandas
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found. Please ensure it's in the current directory.")
        return
    except Exception as e:
        print(f"An error occurred while loading the CSV file: {e}")
        return

    # Define the columns required for plotting and filtering
    x_col = 'culmen_length_mm'
    y_col = 'culmen_depth_mm'
    hue_col = 'species'

    # 3. Filter out data points where culmen_length_mm, culmen_depth_mm, OR species are missing
    # Create a list of columns to check for NA
    cols_to_check_na = [x_col, y_col, hue_col]
    
    # Drop rows where any of the specified columns have NA values
    df_filtered = df.dropna(subset=cols_to_check_na)

    # Check if there's any data left after filtering
    if df_filtered.empty:
        print("No data remaining after filtering out missing values for culmen length, culmen depth, or species.")
        return

    # Initialize the matplotlib figure and axes for plotting
    plt.figure(figsize=(10, 7))

    # 1. & 2. Create a scatter plot of culmen_length_mm (x-axis) vs culmen_depth_mm (y-axis)
    # Color the points according to the species column using seaborn
    sns.scatterplot(
        data=df_filtered,
        x=x_col,
        y=y_col,
        hue=hue_col,  # Color points by species
        s=70,         # Set marker size for better visibility
        alpha=0.8     # Set transparency for overlapping points
    )

    # 4. Ensure the plot has a title and labels for the x-axis and y-axis
    plt.title('Penguin Culmen Length vs. Culmen Depth by Species', fontsize=16)
    plt.xlabel('Culmen Length (mm)', fontsize=12)
    plt.ylabel('Culmen Depth (mm)', fontsize=12)

    # Add a legend to distinguish species colors
    plt.legend(title='Species', bbox_to_anchor=(1.05, 1), loc='upper left')

    # Adjust layout to prevent labels/legend from being cut off
    plt.tight_layout()

    # Display the plot
    plt.show()

if __name__ == '__main__':
    # This block ensures the function is called only when the script is executed directly
    # Assume 'penguins.csv' is in the same directory as this script
    plot_penguins_culmen_data('penguins.csv')
```