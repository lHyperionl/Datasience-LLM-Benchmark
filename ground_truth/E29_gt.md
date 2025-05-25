```python
import pandas as pd

def analyze_heart_data(file_path: str = 'heart.csv'):
    """
    Loads the heart.csv dataset, filters it based on specific criteria,
    and displays selected columns from the filtered data.

    Args:
        file_path (str): The path to the heart.csv dataset file.
                         Defaults to 'heart.csv' assuming it's in the current directory.
    """
    try:
        # 1. Load the heart.csv dataset using pandas
        # It's good practice to specify the encoding if there are issues, though utf-8 is common.
        df = pd.read_csv(file_path)

        # 2. Filter the dataset to include only patients with Age > 50 AND HeartDisease == 1
        # Use boolean indexing with multiple conditions combined by '&' (logical AND)
        # Parentheses are important for clarity and correct evaluation of conditions.
        filtered_df = df[(df['Age'] > 50) & (df['HeartDisease'] == 1)]

        # 3. From this filtered dataset, select and display the specified columns
        # Select columns using a list of column names.
        selected_columns = ['Age', 'Sex', 'ChestPainType', 'Cholesterol']
        
        # Check if all required columns exist in the DataFrame to handle potential KeyError
        missing_columns = [col for col in selected_columns if col not in filtered_df.columns]
        if missing_columns:
            print(f"Error: The following required columns are missing from the dataset: {missing_columns}")
            return

        display_df = filtered_df[selected_columns]

        # Display the resulting DataFrame
        print("Filtered Data (Age > 50 AND HeartDisease == 1) with selected columns:")
        print(display_df)

        # Optionally, display some basic info about the displayed data
        print(f"\nNumber of records matching criteria: {len(display_df)}")
        print("\nData types of displayed columns:")
        print(display_df.dtypes)

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found. Please ensure it's in the correct directory.")
    except KeyError as e:
        print(f"Error: A required column was not found in the dataset. Details: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# --- Main execution ---
if __name__ == "__main__":
    # Call the function to perform the data analysis
    analyze_heart_data()
```