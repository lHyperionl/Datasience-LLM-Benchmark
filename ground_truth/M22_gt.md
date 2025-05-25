```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Define the path to the dataset
DATASET_PATH = 'clash_royale_cards.csv'

# Define the numerical columns to be analyzed
NUMERICAL_COLUMNS = [
    'elixirCost',
    'Win Rate',
    'Usage',
    'Win Rate Change',
    'Usage Change'
]

def analyze_clash_royale_data(dataset_path: str, numerical_cols: list):
    """
    Loads the Clash Royale cards dataset, selects specified numerical columns,
    calculates their pairwise correlation matrix, and visualizes it as a heatmap.

    Args:
        dataset_path (str): The path to the CSV dataset file.
        numerical_cols (list): A list of column names to select for correlation analysis.
    """
    try:
        # 1. Load the clash_royale_cards.csv dataset using pandas
        df = pd.read_csv(dataset_path)
        print(f"Dataset '{dataset_path}' loaded successfully. Shape: {df.shape}")

        # 2. Select the numerical columns
        # Check if all specified columns exist in the DataFrame
        missing_cols = [col for col in numerical_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"The following specified columns are missing from the dataset: {missing_cols}")

        df_numerical = df[numerical_cols]
        print(f"\nSelected numerical columns: {numerical_cols}")
        print("First 5 rows of selected data:")
        print(df_numerical.head())

        # 3. Calculate the pairwise correlation matrix for these selected columns
        correlation_matrix = df_numerical.corr()
        print("\nCalculated Correlation Matrix:")
        print(correlation_matrix)

        # 4. Visualize this correlation matrix using a heatmap (e.g., with seaborn)
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            correlation_matrix,
            annot=True,       # Show the correlation values on the heatmap
            cmap='coolwarm',  # Choose a diverging colormap
            fmt=".2f",        # Format annotation values to two decimal places
            linewidths=.5     # Add lines between cells
        )
        plt.title('Pairwise Correlation Matrix of Numerical Clash Royale Card Attributes')
        plt.show()

    except FileNotFoundError:
        print(f"Error: The file '{dataset_path}' was not found. Please ensure it's in the current directory.")
    except KeyError as e:
        print(f"Error: A specified column was not found in the dataset. {e}")
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    # Execute the analysis function
    analyze_clash_royale_data(DATASET_PATH, NUMERICAL_COLUMNS)
```