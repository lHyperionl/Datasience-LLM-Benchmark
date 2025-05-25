```python
import pandas as pd

# 1. Create a pandas DataFrame with student performance data
# Define the data for the DataFrame
data = {
    'StudentID': ['S001', 'S002', 'S003', 'S004', 'S005', 'S006', 'S007', 'S008', 'S009', 'S010'],
    'Subject': ['Math', 'Science', 'Math', 'English', 'Science', 'Math', 'English', 'Science', 'Math', 'English'],
    'Midterm_Score': [85, 92, 78, 88, 76, 95, 82, 89, 70, 91],
    'Final_Score': [90, 88, 82, 92, 80, 91, 85, 93, 75, 89],
    'Attendance_Rate': [0.95, 0.98, 0.85, 0.92, 0.80, 0.99, 0.88, 0.96, 0.75, 0.93]
}

# Create the DataFrame
df = pd.DataFrame(data)

print("Original DataFrame:")
print(df)
print("\n" + "="*50 + "\n")

# 2. Add a new column 'Score_Improvement'
# Calculate 'Score_Improvement' as 'Final_Score' - 'Midterm_Score'
df['Score_Improvement'] = df['Final_Score'] - df['Midterm_Score']

print("DataFrame after adding 'Score_Improvement' column:")
print(df)
print("\n" + "="*50 + "\n")

# 3. For each 'Subject', calculate and print the average 'Score_Improvement'
# Group the DataFrame by 'Subject' and calculate the mean of 'Score_Improvement' for each group
average_improvement_per_subject = df.groupby('Subject')['Score_Improvement'].mean()

print("Average Score Improvement per Subject:")
print(average_improvement_per_subject)
print("\n" + "="*50 + "\n")

# 4. Identify and print the 'StudentID' of the student(s) who achieved a
#    'Final_Score' greater than 90 and had an 'Attendance_Rate' of 0.95 or higher.

# Filter the DataFrame based on the specified conditions
# Condition 1: Final_Score > 90
# Condition 2: Attendance_Rate >= 0.95
high_achievers_and_attendees = df[
    (df['Final_Score'] > 90) &
    (df['Attendance_Rate'] >= 0.95)
]

# Extract the 'StudentID' from the filtered DataFrame
student_ids_meeting_criteria = high_achievers_and_attendees['StudentID']

print("Student(s) with Final_Score > 90 and Attendance_Rate >= 0.95:")
if not student_ids_meeting_criteria.empty:
    for student_id in student_ids_meeting_criteria:
        print(student_id)
else:
    print("No students met both criteria.")

print("\n" + "="*50 + "\n")
```