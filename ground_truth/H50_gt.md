```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import numpy as np

# 1. Create a pandas DataFrame with student performance data
print("--- Creating DataFrame ---")
data = {
    'StudentID': ['S001', 'S002', 'S003', 'S004', 'S005', 'S006', 'S007', 'S008', 'S009', 'S010', 'S011', 'S012'],
    'Subject': ['Math', 'Science', 'Math', 'English', 'Science', 'Math', 'English', 'Science', 'Math', 'English', 'Math', 'Science'],
    'Midterm_Score': [85, 92, 78, 88, 76, 95, 82, 89, 70, 91, 65, 77],
    'Final_Score': [90, 88, 82, 92, 80, 91, 85, 93, 75, 89, 70, 81],
    'Attendance_Rate': [0.95, 0.98, 0.85, 0.92, 0.80, 0.99, 0.88, 0.96, 0.75, 0.93, 0.70, 0.82]
}
df = pd.DataFrame(data)
print("DataFrame created successfully:")
print(df.head())
print("\n")

# 2. Calculate the Pearson correlation coefficient between 'Attendance_Rate' and 'Final_Score'
print("--- Pearson Correlation Calculation ---")
# Ensure there are no missing values in the columns before calculating correlation
# For this dataset, we know there are none, but it's good practice.
correlation_coefficient, _ = pearsonr(df['Attendance_Rate'], df['Final_Score'])
print(f"Pearson correlation coefficient between 'Attendance_Rate' and 'Final_Score': {correlation_coefficient:.4f}")
print("\n")

# 3. Create and display a scatter plot of 'Attendance_Rate' vs 'Final_Score'
print("--- Generating Scatter Plot ---")
plt.figure(figsize=(10, 6)) # Set the figure size for better readability

# Use seaborn for a more aesthetically pleasing plot with automatic legend for hue
sns.scatterplot(
    data=df,
    x='Attendance_Rate',
    y='Final_Score',
    hue='Subject', # Color points based on 'Subject'
    s=100,         # Set marker size
    alpha=0.8,     # Set transparency
    edgecolor='w'  # Add white edge to markers
)

plt.title('Final Score vs. Attendance Rate by Subject', fontsize=16)
plt.xlabel('Attendance Rate', fontsize=12)
plt.ylabel('Final Score', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7) # Add a grid for better readability
plt.legend(title='Subject', bbox_to_anchor=(1.05, 1), loc='upper left') # Place legend outside the plot
plt.tight_layout() # Adjust layout to prevent labels from overlapping
plt.show()
print("Scatter plot displayed successfully.")
print("\n")

# 4. Add a new column 'Overall_Grade' based on 'Final_Score'
print("--- Assigning Overall Grades ---")
conditions = [
    (df['Final_Score'] >= 90),
    (df['Final_Score'] >= 80) & (df['Final_Score'] < 90),
    (df['Final_Score'] >= 70) & (df['Final_Score'] < 80),
    (df['Final_Score'] < 70)
]
grades = ['A', 'B', 'C', 'D']

df['Overall_Grade'] = np.select(conditions, grades, default='N/A') # Use np.select for conditional assignment
print("DataFrame with 'Overall_Grade' column:")
print(df[['StudentID', 'Final_Score', 'Overall_Grade']].head())
print("\n")

# 5. Print the count of students in each grade category
print("--- Grade Distribution ---")
grade_counts = df['Overall_Grade'].value_counts().sort_index() # Sort by grade for consistent order
print("Count of students in each grade category:")
print(grade_counts)
print("\n")

print("--- Task Completed ---")
```