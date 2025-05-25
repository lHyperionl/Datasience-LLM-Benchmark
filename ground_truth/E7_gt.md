```python
import pandas as pd

# Define the dataset file name
DATASET_FILE = 'water_potability.csv'

# --- 1. Load the dataset and display the first 10 rows ---
try:
    df = pd.read_csv(DATASET_FILE)
    print("--- 1. First 10 rows of the dataset ---")
    print(df.head(10))
    print("\n" + "="*50 + "\n")
except FileNotFoundError:
    print(f"Error: The file '{DATASET_FILE}' was not found. Please ensure it's in the current directory.")
    exit()
except Exception as e:
    print(f"An error occurred while loading the dataset: {e}")
    exit()

# --- 2. Show the shape of the dataset (rows and columns) ---
print("--- 2. Shape of the dataset (rows, columns) ---")
print(df.shape)
print("\n" + "="*50 + "\n")

# --- 3. List all column names and their data types ---
print("--- 3. Column names and their data types ---")
# Using .info() provides a comprehensive summary including non-null counts and dtypes
df.info()
print("\n" + "="*50 + "\n")

# --- 4. Calculate the percentage of potable vs non-potable water samples ---
# Assuming 'Potability' is the column indicating potability (1 for potable, 0 for non-potable)
if 'Potability' in df.columns:
    print("--- 4. Percentage of Potable vs Non-Potable water samples ---")
    potability_counts = df['Potability'].value_counts()
    total_samples = potability_counts.sum()

    if total_samples > 0:
        potability_percentage = (potability_counts / total_samples) * 100
        print(f"Total samples: {total_samples}")
        print("Potability Distribution:")
        print(potability_percentage.round(2))
        print("\nInterpretation: 1 = Potable, 0 = Non-Potable")
    else:
        print("No samples found in 'Potability' column to calculate percentages.")
else:
    print("--- 4. 'Potability' column not found. Cannot calculate percentages. ---")
print("\n" + "="*50 + "\n")

# --- 5. Display the count of missing values for each column ---
print("--- 5. Count of missing values for each column ---")
missing_values = df.isnull().sum()
print(missing_values[missing_values > 0]) # Display only columns with missing values
if missing_values.sum() == 0:
    print("No missing values found in the dataset.")
print("\n" + "="*50 + "\n")

# --- 6. Show basic summary statistics for all numerical columns ---
print("--- 6. Basic summary statistics for numerical columns ---")
# .describe() automatically selects numerical columns
print(df.describe())
print("\n" + "="*50 + "\n")
```