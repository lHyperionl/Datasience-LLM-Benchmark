```python
import pandas as pd

# Define the DataFrame as specified in the task
# This dictionary holds the data for each column.
# 'AdmissionDate' and 'DischargeDate' are converted to datetime objects
# using pd.to_datetime for proper date handling.
data = {
    "PatientID": ["P001", "P002", "P003", "P004", "P005", "P006"],
    "AdmissionDate": pd.to_datetime(["2025-01-10", "2025-01-12", "2025-01-15", "2025-01-18", "2025-01-20", "2025-01-22"]),
    "DischargeDate": pd.to_datetime(["2025-01-15", "2025-01-16", "2025-01-22", "2025-01-25", "2025-01-24", "2025-01-28"]),
    "Diagnosis": ["Flu", "Cold", "Flu", "Injury", "Cold", "Flu"],
    "Age": [34, 56, 23, 45, 67, 30],
    "Gender": ["Male", "Female", "Female", "Male", "Male", "Female"]
}

# Create the pandas DataFrame from the defined data
df = pd.DataFrame(data)

# --- Data Analysis Tasks ---

# 1. Calculate and print the average "Age" of all patients.
# Access the 'Age' column and apply the .mean() method to compute the average.
average_age = df['Age'].mean()

# Print the calculated average age.
print(f"Average Age of all patients: {average_age:.2f} years")

# 2. Count and print the number of patients for each "Gender".
# Access the 'Gender' column and use the .value_counts() method.
# This method returns a Series containing counts of unique values in the column.
gender_counts = df['Gender'].value_counts()

# Print the counts for each gender.
print("\nNumber of patients by Gender:")
print(gender_counts)
```