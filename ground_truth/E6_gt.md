```python
import pandas as pd
import re

# --- 1. Load the dataset and identify columns with missing values ---

# Load the titanic.csv dataset into a pandas DataFrame
try:
    df = pd.read_csv('titanic.csv')
except FileNotFoundError:
    print("Error: 'titanic.csv' not found. Please ensure the file is in the current directory.")
    exit()

# Display columns with missing values and their counts
# print("Missing values before cleaning:")
# print(df.isnull().sum()[df.isnull().sum() > 0])

# --- 2. Fill missing Age values with the median age ---

# Calculate the median age
median_age = df['Age'].median()

# Fill missing 'Age' values with the calculated median
df['Age'].fillna(median_age, inplace=True)

# --- 3. Fill missing Embarked values with the most frequent port ---

# Find the most frequent 'Embarked' port (mode)
# .mode()[0] is used because .mode() can return multiple values if there's a tie,
# and we just need one to fill.
most_frequent_embarked = df['Embarked'].mode()[0]

# Fill missing 'Embarked' values with the most frequent port
df['Embarked'].fillna(most_frequent_embarked, inplace=True)

# --- 4. Drop the Cabin column due to excessive missing values ---

# Drop the 'Cabin' column from the DataFrame
df.drop('Cabin', axis=1, inplace=True)

# --- 5. Create a new column 'Title' extracted from the Name column ---

# Define a function to extract title using regex
def get_title(name):
    # Search for a pattern like ' Mr.', ' Mrs.', ' Miss.', etc.
    # The pattern looks for a space, then one or more letters, then a period.
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it
    if title_search:
        return title_search.group(1)
    return ""

# Apply the function to the 'Name' column to create the 'Title' column
df['Title'] = df['Name'].apply(get_title)

# Standardize less common titles to more general categories
# This step is good practice for feature engineering but not strictly required by the prompt.
# However, it makes the 'Title' column more useful.
df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
df['Title'] = df['Title'].replace('Mlle', 'Miss')
df['Title'] = df['Title'].replace('Ms', 'Miss')
df['Title'] = df['Title'].replace('Mme', 'Mrs')


# --- 6. Display the cleaned dataset's shape and the first 5 rows ---

# Display the shape of the cleaned DataFrame (rows, columns)
print("Shape of the cleaned dataset:", df.shape)

# Display the first 5 rows of the cleaned DataFrame
print("\nFirst 5 rows of the cleaned dataset:")
print(df.head())

# Optional: Verify no more missing values in processed columns
# print("\nMissing values after cleaning:")
# print(df.isnull().sum()[df.isnull().sum() > 0])
```