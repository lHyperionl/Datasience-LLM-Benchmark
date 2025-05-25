```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- Configuration ---
# Define the path to the dataset
DATASET_PATH = 'titanic.csv'

# --- Data Loading ---
try:
    df = pd.read_csv(DATASET_PATH)
    print(f"Successfully loaded {DATASET_PATH}")
except FileNotFoundError:
    print(f"Error: '{DATASET_PATH}' not found. Please ensure the CSV file is in the current directory.")
    exit() # Exit if the file is not found

# --- Visualization 1: Count of Survivors vs Non-survivors ---
plt.figure(figsize=(7, 5))
# Count occurrences of 'Survived' column
survivor_counts = df['Survived'].value_counts()
# Map numerical values to descriptive labels for better readability
survivor_labels = {0: 'Non-survivor', 1: 'Survivor'}
survivor_counts.index = survivor_counts.index.map(survivor_labels)
# Create a bar plot
survivor_counts.plot(kind='bar', color=['salmon', 'lightgreen'])
plt.title('Count of Survivors vs Non-survivors')
plt.xlabel('Survival Status')
plt.ylabel('Number of Passengers')
plt.xticks(rotation=0) # Keep x-axis labels horizontal
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout() # Adjust layout to prevent labels from overlapping
plt.show()

# --- Visualization 2: Distribution of Passenger Classes (Pclass) ---
plt.figure(figsize=(7, 5))
# Count occurrences of 'Pclass' column
pclass_counts = df['Pclass'].value_counts().sort_index()
# Create a bar plot
pclass_counts.plot(kind='bar', color='skyblue')
plt.title('Distribution of Passenger Classes')
plt.xlabel('Passenger Class (Pclass)')
plt.ylabel('Number of Passengers')
plt.xticks(rotation=0) # Keep x-axis labels horizontal
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# --- Visualization 3: Histogram of Passenger Ages (handling missing values) ---
plt.figure(figsize=(10, 6))
# Handle missing 'Age' values: fill with the median age
# Median is often preferred over mean for skewed distributions or presence of outliers
# A copy is made to avoid SettingWithCopyWarning if the original df is modified later
age_data = df['Age'].copy()
median_age = age_data.median()
age_data.fillna(median_age, inplace=True)
print(f"\nMissing 'Age' values filled with median: {median_age:.2f}")

# Create a histogram
plt.hist(age_data, bins=20, edgecolor='black', color='lightcoral')
plt.title('Distribution of Passenger Ages')
plt.xlabel('Age')
plt.ylabel('Number of Passengers')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# --- Visualization 4: Count of Passengers by Gender (Sex) ---
plt.figure(figsize=(7, 5))
# Count occurrences of 'Sex' column
gender_counts = df['Sex'].value_counts()
# Create a bar plot
gender_counts.plot(kind='bar', color=['lightsteelblue', 'plum'])
plt.title('Count of Passengers by Gender')
plt.xlabel('Gender')
plt.ylabel('Number of Passengers')
plt.xticks(rotation=0) # Keep x-axis labels horizontal
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

print("\nAll requested visualizations have been generated.")
```