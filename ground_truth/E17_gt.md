```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

# --- 1. Load the dataset ---
# Assume the CSV file is in the current directory
try:
    df = pd.read_csv('world-happiness-report-2021.csv')
    print("Dataset loaded successfully. Initial shape:", df.shape)
except FileNotFoundError:
    print("Error: 'world-happiness-report-2021.csv' not found. Please ensure the file is in the current directory.")
    exit()

# --- 2. Handle missing values ---
# Identify numerical and categorical columns
numerical_cols = df.select_dtypes(include=np.number).columns
categorical_cols = df.select_dtypes(include='object').columns

# Fill missing numerical values with the mean
for col in numerical_cols:
    if df[col].isnull().any():
        df[col].fillna(df[col].mean(), inplace=True)
        # print(f"Filled missing values in numerical column '{col}' with mean.")

# Fill missing categorical values with the mode
for col in categorical_cols:
    if df[col].isnull().any():
        # .mode()[0] handles cases where there might be multiple modes
        df[col].fillna(df[col].mode()[0], inplace=True)
        # print(f"Filled missing values in categorical column '{col}' with mode.")

# --- 3. Create new derived features ---

# 'Happiness_Category' (High: >6.5, Medium: 4-6.5, Low: <4)
conditions_happiness = [
    df['Ladder score'] > 6.5,
    (df['Ladder score'] >= 4) & (df['Ladder score'] <= 6.5),
    df['Ladder score'] < 4
]
choices_happiness = ['High', 'Medium', 'Low']
df['Happiness_Category'] = np.select(conditions_happiness, choices_happiness, default='Unknown')

# 'GDP_Category' (based on quartiles of 'Logged GDP per capita')
# pd.qcut automatically handles binning into quartiles
df['GDP_Category'] = pd.qcut(df['Logged GDP per capita'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])

# 'Well_being_Score' (average of Social support, Healthy life expectancy, and Freedom)
df['Well_being_Score'] = df[['Social support', 'Healthy life expectancy', 'Freedom to make life choices']].mean(axis=1)

# --- 4. Remove outliers using the IQR method for 'Ladder score' ---
Q1 = df['Ladder score'].quantile(0.25)
Q3 = df['Ladder score'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filter out the outliers
initial_rows = df.shape[0]
df = df[(df['Ladder score'] >= lower_bound) & (df['Ladder score'] <= upper_bound)]
# print(f"Removed {initial_rows - df.shape[0]} outliers from 'Ladder score'. New shape: {df.shape}")

# --- 5. Standardize numerical features using StandardScaler ---
# Identify numerical columns to scale. Exclude 'Country name' as it's an identifier.
# New categorical features ('Happiness_Category', 'GDP_Category') are not numerical.
# 'Well_being_Score' is numerical and should be scaled.
numerical_features_to_scale = df.select_dtypes(include=np.number).columns.tolist()

# Remove any columns that should not be scaled, e.g., identifiers or already encoded categorical features
# 'Regional indicator' will be encoded separately, so it's not in numerical_features_to_scale yet.
# All current numerical columns are suitable for scaling.

scaler = StandardScaler()
df[numerical_features_to_scale] = scaler.fit_transform(df[numerical_features_to_scale])
# print("Numerical features standardized.")

# --- 6. Encode the 'Regional indicator' using label encoding ---
le = LabelEncoder()
df['Regional indicator'] = le.fit_transform(df['Regional indicator'])
# print("'Regional indicator' column encoded.")

# --- 7. Display the cleaned dataset's shape and first 5 rows ---
print("\n--- Cleaned Dataset Information ---")
print("Shape of the cleaned dataset:", df.shape)
print("\nFirst 5 rows of the cleaned dataset:")
print(df.head())
```