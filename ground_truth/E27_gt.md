```python
import pandas as pd
import matplotlib.pyplot as plt

# Define the path to the dataset file
DATASET_FILE = 'iris.csv'

def main():
    """
    Main function to load the Iris dataset, create a scatter plot,
    and display it.
    """
    try:
        # 1. Load the iris.csv dataset using pandas
        # Assumes the CSV file is in the current directory.
        df = pd.read_csv(DATASET_FILE)

        # Ensure required columns exist
        required_columns = ['SepalLengthCm', 'SepalWidthCm', 'Species']
        if not all(col in df.columns for col in required_columns):
            raise KeyError(f"One or more required columns are missing. Expected: {required_columns}")

        # Set up the plot figure size for better readability
        plt.figure(figsize=(10, 7))

        # Get unique species names to iterate and color points
        species_names = df['Species'].unique()

        # 2. Create a scatter plot visualizing the relationship between
        #    SepalLengthCm (x-axis) and SepalWidthCm (y-axis).
        # 3. Color the points in the scatter plot according to the Species column.
        # Iterate through each species to plot them with different colors and create a legend
        for species in species_names:
            # Filter the DataFrame for the current species
            subset = df[df['Species'] == species]
            plt.scatter(
                subset['SepalLengthCm'],  # X-axis data
                subset['SepalWidthCm'],   # Y-axis data
                label=species,            # Label for the legend
                alpha=0.8,                # Transparency for better visibility of overlapping points
                edgecolors='w',           # White edges for points for better separation
                linewidth=0.5             # Line width for edges
            )

        # 4. Ensure the plot has a title, and labels for the x-axis and y-axis.
        plt.title('Sepal Length vs. Sepal Width by Species (Iris Dataset)')
        plt.xlabel('SepalLengthCm')
        plt.ylabel('SepalWidthCm')

        # Add a legend to distinguish the species
        plt.legend(title='Species')

        # Add a grid for better readability
        plt.grid(True, linestyle='--', alpha=0.6)

        # Display the plot
        plt.show()

    except FileNotFoundError:
        print(f"Error: The file '{DATASET_FILE}' was not found.")
        print("Please ensure 'iris.csv' is in the same directory as the script.")
    except KeyError as e:
        print(f"Error: A required column was not found in the dataset. {e}")
        print("Please check the column names in 'iris.csv'.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
```