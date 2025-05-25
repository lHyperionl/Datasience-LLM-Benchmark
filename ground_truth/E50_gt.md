```python
import pandas as pd

# 1. Create the pandas DataFrame with student performance data
# Define the data as a dictionary
data = {
    'StudentID': ['S001', 'S002', 'S003', 'S004', 'S005', 'S006', 'S007', 'S008'],
    'Subject': ['Math', 'Science', 'Math', 'English', 'Science', 'Math', 'English', 'Science'],
    'Midterm_Score': [85, 92, 78, 88, 76, 95, 82, 89],
    'Final_Score': [90, 88, 82, 92, 80, 91, 85, 93],
    'Attendance_Rate': [0.95, 0.98, 0.85, 0.92, 0.80, 0.99, 0.88, 0.96]
}

# Create the DataFrame
df = pd.DataFrame(data)

# Display the DataFrame (optional, for verification)
# print("Original DataFrame:")
# print(df)
# print("-" * 30)

# 2. Calculate and print the average 'Midterm_Score' for all students
# Use the .mean() method on the 'Midterm_Score' column
average_midterm_score = df['Midterm_Score'].mean()
print(f"Average Midterm Score for all students: {average_midterm_score:.2f}")

# 3. Calculate and print the average 'Final_Score' for all students
# Use the .mean() method on the 'Final_Score' column
average_final_score = df['Final_Score'].mean()
print(f"Average Final Score for all students: {average_final_score:.2f}")

# 4. Identify and print the 'Subject' with the highest average 'Final_Score'
# Group the DataFrame by 'Subject' and calculate the mean of 'Final_Score' for each subject
average_final_score_by_subject = df.groupby('Subject')['Final_Score'].mean()

# Find the subject with the maximum average final score
# idxmax() returns the index (which is the 'Subject' in this grouped series) of the first occurrence of the maximum value
subject_highest_final_score = average_final_score_by_subject.idxmax()

# Get the actual highest average score for that subject
highest_average_score_value = average_final_score_by_subject.max()

print(f"Subject with the highest average Final Score: '{subject_highest_final_score}' "
      f"(Average Score: {highest_average_score_value:.2f})")

# Handle edge case: If the DataFrame were empty or 'Final_Score' column was missing/empty
# The .mean() method on an empty Series would return NaN.
# .idxmax() on an empty Series would raise an error, so a check might be needed in a more robust system.
# For this specific problem, the DataFrame is guaranteed to be populated as per the prompt.
```