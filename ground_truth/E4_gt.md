```python
import pandas as pd

# --- 1. Load the dataset and display the first 5 rows ---
try:
    # Load the titanic.csv dataset into a pandas DataFrame
    df = pd.read_csv('titanic.csv')
    print("1. First 5 rows of the dataset:")
    # Display the first 5 rows of the DataFrame
    print(df.head())
except FileNotFoundError:
    print("Error: 'titanic.csv' not found. Please ensure the file is in the current directory.")
    # Exit the script if the file is not found, as subsequent operations will fail
    exit()

# --- 2. Show the shape of the dataset (rows and columns) ---
print("\n2. Shape of the dataset (rows, columns):")
# The .shape attribute returns a tuple representing the dimensionality of the DataFrame (rows, columns)
print(df.shape)

# --- 3. List all column names and their data types ---
print("\n3. Column names and their data types:")
# The .info() method prints a concise summary of a DataFrame,
# including the index dtype and column dtypes, non-null values, and memory usage.
df.info()

# --- 4. Calculate the survival rate (percentage of passengers who survived) ---
# The 'Survived' column contains 1 for passengers who survived and 0 for those who did not.
# Summing this column gives the total number of survivors.
total_passengers = df.shape[0]
survived_passengers = df['Survived'].sum()

# Calculate the survival rate as a percentage
survival_rate = (survived_passengers / total_passengers) * 100

print(f"\n4. Survival Rate: {survival_rate:.2f}%")

# --- 5. Show the count of missing values for each column ---
print("\n5. Count of missing values for each column:")
# .isnull() returns a DataFrame of boolean values indicating missing data (True for NaN).
# .sum() then counts the number of True values (missing values) for each column.
print(df.isnull().sum())
```