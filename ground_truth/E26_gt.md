```python
import pandas as pd

# Define the path to the dataset file.
# Assumes 'iris.csv' is in the same directory as the script.
DATA_FILE = 'iris.csv'

try:
    # 1. Load the iris.csv dataset using pandas.
    # This reads the CSV file into a pandas DataFrame.
    df = pd.read_csv(DATA_FILE)

    # 2. Filter the dataset to include only records where the `Species` column is 'Iris-setosa'.
    # Boolean indexing is used here: df[condition] returns rows where the condition is True.
    filtered_setosa_df = df[df['Species'] == 'Iris-setosa']

    # 3. From this filtered dataset, select and display only the `SepalLengthCm` and `SepalWidthCm` columns.
    # We use a list of column names to select multiple specific columns from the filtered DataFrame.
    # The result is then printed to the console.
    selected_sepal_columns_df = filtered_setosa_df[['SepalLengthCm', 'SepalWidthCm']]

    # Display the final DataFrame containing only the desired columns for 'Iris-setosa' species.
    print(selected_sepal_columns_df)

except FileNotFoundError:
    # Handle the case where the 'iris.csv' file is not found.
    # This provides a user-friendly error message.
    print(f"Error: The file '{DATA_FILE}' was not found.")
    print("Please ensure 'iris.csv' is in the current directory or provide the correct path.")
except KeyError as e:
    # Handle cases where expected columns ('Species', 'SepalLengthCm', 'SepalWidthCm') are missing.
    # This indicates an issue with the dataset's structure.
    print(f"Error: A required column was not found in the dataset. Details: {e}")
    print("Please ensure the dataset contains 'Species', 'SepalLengthCm', and 'SepalWidthCm' columns.")
except Exception as e:
    # Catch any other unexpected errors that might occur during execution.
    print(f"An unexpected error occurred: {e}")
```