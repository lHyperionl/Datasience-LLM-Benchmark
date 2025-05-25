```python
import pandas as pd

# --- Task 1: Load the dataset and display the first 10 rows ---
try:
    df = pd.read_csv('world-happiness-report-2021.csv')
    print("1. First 10 rows of the dataset:")
    print(df.head(10))
    print("\n" + "="*70 + "\n") # Separator for clarity

except FileNotFoundError:
    print("Error: 'world-happiness-report-2021.csv' not found. Please ensure the file is in the current directory.")
    exit() # Exit if the file is not found, as subsequent operations will fail.

# --- Task 2: Show the shape of the dataset (rows and columns) ---
print("2. Shape of the dataset (rows, columns):")
print(df.shape)
print("\n" + "="*70 + "\n")

# --- Task 3: List all column names and their data types ---
print("3. Column names and their data types:")
# Using .info() provides a concise summary including non-null counts and dtypes
df.info()
print("\n" + "="*70 + "\n")

# --- Task 4: Display the unique regional indicators and count of countries per region ---
print("4. Unique regional indicators and count of countries per region:")
if 'Regional indicator' in df.columns:
    print(df['Regional indicator'].value_counts())
else:
    print("Column 'Regional indicator' not found in the dataset.")
print("\n" + "="*70 + "\n")

# --- Task 5: Show the top 10 happiest countries based on Ladder score ---
print("5. Top 10 happiest countries based on Ladder score:")
if 'Ladder score' in df.columns and 'Country name' in df.columns:
    top_10_happiest = df.sort_values(by='Ladder score', ascending=False).head(10)
    print(top_10_happiest[['Country name', 'Ladder score']])
else:
    print("Required columns ('Ladder score' or 'Country name') not found for this analysis.")
print("\n" + "="*70 + "\n")

# --- Task 6: Calculate basic summary statistics for key happiness factors ---
print("6. Basic summary statistics for key happiness factors:")
key_happiness_factors = [
    'Ladder score',
    'Logged GDP per capita',
    'Social support',
    'Healthy life expectancy',
    'Freedom to make life choices'
]

# Filter for columns that actually exist in the DataFrame
existing_factors = [col for col in key_happiness_factors if col in df.columns]

if existing_factors:
    print(df[existing_factors].describe())
else:
    print("None of the specified key happiness factors were found in the dataset.")
print("\n" + "="*70 + "\n")

# --- Task 7: Identify any missing values in the dataset ---
print("7. Missing values in the dataset (count per column):")
print(df.isnull().sum())
print("\n" + "="*70 + "\n")
```