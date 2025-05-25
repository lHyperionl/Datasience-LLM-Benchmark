```python
import pandas as pd
import numpy as np

# --- 1. Load the dataset and parse the 'date' column as datetime objects ---
# Assume the CSV file is in the current directory
try:
    df = pd.read_csv('synthetic_covid19_data.csv')
except FileNotFoundError:
    print("Error: 'synthetic_covid19_data.csv' not found. Please ensure the file is in the current directory.")
    exit()

# Convert 'date' column to datetime objects
df['date'] = pd.to_datetime(df['date'])

# --- 2. Handle missing values ---
# Fill missing values in 'hosp_patients' and 'icu_patients' with 0
df['hosp_patients'].fillna(0, inplace=True)
df['icu_patients'].fillna(0, inplace=True)

# Fill missing values in 'positive_rate' and 'reproduction_rate' with the mean of their respective column
# Ensure columns are numeric before calculating mean, though pandas usually infers this correctly.
df['positive_rate'] = pd.to_numeric(df['positive_rate'], errors='coerce')
df['reproduction_rate'] = pd.to_numeric(df['reproduction_rate'], errors='coerce')

df['positive_rate'].fillna(df['positive_rate'].mean(), inplace=True)
df['reproduction_rate'].fillna(df['reproduction_rate'].mean(), inplace=True)

# --- 3. Create a new column 'mortality_rate' ---
# Calculate mortality_rate as ('total_deaths' / 'total_cases') * 100
# Handle potential division by zero by replacing resulting NaNs or infs with 0.
# Convert columns to numeric to ensure calculations are correct
df['total_deaths'] = pd.to_numeric(df['total_deaths'], errors='coerce')
df['total_cases'] = pd.to_numeric(df['total_cases'], errors='coerce')

# Calculate mortality rate
df['mortality_rate'] = (df['total_deaths'] / df['total_cases']) * 100

# Replace infinite values (e.g., division by zero where numerator is non-zero) with 0
df['mortality_rate'].replace([np.inf, -np.inf], 0, inplace=True)
# Replace NaN values (e.g., 0/0 or NaN in original columns) with 0
df['mortality_rate'].fillna(0, inplace=True)

# --- 4. Create a new column 'vaccination_rate_per_capita' ---
# Calculate vaccination_rate_per_capita as ('people_fully_vaccinated' / 'population') * 100
# Handle potential division by zero.
# Convert columns to numeric
df['people_fully_vaccinated'] = pd.to_numeric(df['people_fully_vaccinated'], errors='coerce')
df['population'] = pd.to_numeric(df['population'], errors='coerce')

# Calculate vaccination rate per capita
df['vaccination_rate_per_capita'] = (df['people_fully_vaccinated'] / df['population']) * 100

# Replace infinite values with 0
df['vaccination_rate_per_capita'].replace([np.inf, -np.inf], 0, inplace=True)
# Replace NaN values with 0
df['vaccination_rate_per_capita'].fillna(0, inplace=True)

# --- 5. Filter the dataset to include only records from the year 2021 ---
df_2021 = df[df['date'].dt.year == 2021].copy() # Use .copy() to avoid SettingWithCopyWarning

# --- 6. Display the shape of the cleaned dataset and the first 5 rows with the new columns ---
print("Shape of the cleaned dataset (records from 2021):", df_2021.shape)
print("\nFirst 5 rows of the cleaned dataset (records from 2021) with new columns:")
print(df_2021.head())
```