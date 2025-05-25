```python
import pandas as pd

# Define the data for the DataFrame as a dictionary
# Each key represents a column name, and its value is a list of data for that column
data = {
    "EmployeeID": ["E001", "E002", "E003", "E004", "E005", "E006", "E007", "E008"],
    "Department": ["Sales", "HR", "Tech", "Sales", "Tech", "HR", "Tech", "Sales"],
    "PerformanceScore": [4, 3, 5, 2, 4, 3, 5, 4],
    "YearsExperience": [5, 2, 8, 3, 6, 2, 10, 4],
    "ProjectsCompleted": [10, 5, 15, 4, 12, 6, 20, 8],
    "TrainingHours": [20, 40, 10, 50, 20, 30, 15, 25]
}

# Create the pandas DataFrame from the defined data dictionary
df = pd.DataFrame(data)

# --- Task 1: Print the shape of the DataFrame ---
# The .shape attribute returns a tuple representing the dimensionality of the DataFrame (rows, columns).
print("Shape of the DataFrame:")
print(df.shape)
print("-" * 30) # Separator for better readability

# --- Task 2: List the data types of each column ---
# The .dtypes attribute returns a Series with the data type of each column.
print("Data types of each column:")
print(df.dtypes)

# Edge case handling:
# The provided data is clean and consistent, so no explicit edge case handling
# (e.g., missing values, incorrect data types) is required for DataFrame creation
# or for the requested operations (.shape, .dtypes).
# pandas automatically infers appropriate data types (e.g., object for strings, int64 for integers).
```