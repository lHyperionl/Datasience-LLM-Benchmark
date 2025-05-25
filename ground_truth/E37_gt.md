```python
import pandas as pd
import numpy as np

# --- Configuration ---
FILE_PATH = 'mobiles_2025.csv'

# --- Main Script ---

def analyze_mobile_data(file_path: str):
    """
    Loads, cleans, and analyzes the mobile dataset.

    Args:
        file_path (str): The path to the CSV dataset.
    """
    print(f"--- Loading Data from {file_path} ---")
    try:
        df = pd.read_csv(file_path)
        print("Data loaded successfully.")
        print(f"Initial shape: {df.shape}")
        print("\nInitial column data types:")
        print(df.dtypes)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found. Please ensure it's in the current directory.")
        return
    except Exception as e:
        print(f"An error occurred while loading the CSV: {e}")
        return

    print("\n--- Cleaning Data ---")

    # 1) Clean 'Mobile Weight' column
    # Remove 'g' and convert to numeric, coercing errors to NaN
    print("Cleaning 'Mobile Weight' column...")
    df['Mobile Weight'] = df['Mobile Weight'].astype(str).str.replace('g', '', regex=False).str.strip()
    df['Mobile Weight'] = pd.to_numeric(df['Mobile Weight'], errors='coerce')
    print("Cleaned 'Mobile Weight'.")

    # 2) Clean 'RAM' column
    # Remove 'GB' and convert to numeric, coercing errors to NaN
    print("Cleaning 'RAM' column...")
    df['RAM'] = df['RAM'].astype(str).str.replace('GB', '', regex=False).str.strip()
    df['RAM'] = pd.to_numeric(df['RAM'], errors='coerce')
    print("Cleaned 'RAM'.")

    # 3) Clean 'Battery Capacity' column
    # Remove 'mAh' and convert to numeric, coercing errors to NaN
    print("Cleaning 'Battery Capacity' column...")
    df['Battery Capacity'] = df['Battery Capacity'].astype(str).str.replace('mAh', '', regex=False).str.strip()
    df['Battery Capacity'] = pd.to_numeric(df['Battery Capacity'], errors='coerce')
    print("Cleaned 'Battery Capacity'.")

    # 4) Clean 'Launched Price (USA)' column
    # Remove 'USD ', remove commas, and convert to numeric, coercing errors to NaN
    print("Cleaning 'Launched Price (USA)' column...")
    df['Launched Price (USA)'] = df['Launched Price (USA)'].astype(str).str.replace('USD ', '', regex=False)
    df['Launched Price (USA)'] = df['Launched Price (USA)'].str.replace(',', '', regex=False).str.strip()
    df['Launched Price (USA)'] = pd.to_numeric(df['Launched Price (USA)'], errors='coerce')
    print("Cleaned 'Launched Price (USA)'.")

    print("\n--- Data Cleaning Complete ---")

    # 5) List all column names and their inferred data types after cleaning
    print("\n--- Column Names and Inferred Data Types After Cleaning ---")
    print(df.dtypes)

    # 6) Calculate and report basic descriptive statistics for specified cleaned columns
    print("\n--- Basic Descriptive Statistics for Cleaned Numeric Columns ---")
    numeric_cols_for_stats = ['Mobile Weight', 'RAM', 'Battery Capacity', 'Launched Price (USA)']
    
    # Filter out columns that might not exist or are not numeric after cleaning
    existing_numeric_cols = [col for col in numeric_cols_for_stats if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]

    if existing_numeric_cols:
        # Using .describe() for a concise summary
        print(df[existing_numeric_cols].describe().loc[['mean', '50%', 'std', 'min', 'max']])
    else:
        print("No valid numeric columns found for statistics among:", numeric_cols_for_stats)

    # 7) Report the count of unique values in the 'Company Name' column
    print("\n--- Unique Company Names Count ---")
    if 'Company Name' in df.columns:
        unique_company_count = df['Company Name'].nunique()
        print(f"Number of unique companies: {unique_company_count}")
    else:
        print("Column 'Company Name' not found in the dataset.")

    print("\n--- Analysis Complete ---")

if __name__ == "__main__":
    analyze_mobile_data(FILE_PATH)
```