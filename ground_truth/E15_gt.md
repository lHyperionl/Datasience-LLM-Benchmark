```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set a style for the plots for better aesthetics
sns.set_style("whitegrid")

# --- 1) Load the dataset ---
try:
    df = pd.read_csv('diabetes.csv')
    print("Dataset loaded successfully. First 5 rows:")
    print(df.head())
    print("\nDataset Info:")
    df.info()
    print("\nDataset Description:")
    print(df.describe())
except FileNotFoundError:
    print("Error: 'diabetes.csv' not found. Please ensure the file is in the current directory.")
    exit()

# Identify numerical columns for various plots.
# All columns except 'Outcome' are numerical features. 'Outcome' is the target variable.
numerical_cols = df.columns.drop('Outcome')

# --- 2) Create a correlation heatmap showing relationships between all numerical variables ---
plt.figure(figsize=(10, 8))
correlation_matrix = df[numerical_cols].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Heatmap of Numerical Features', fontsize=16)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# --- 3) Generate histograms for all numerical features to show their distributions ---
# Determine the number of rows and columns for the subplot grid
n_cols = 3
n_rows = int(np.ceil(len(numerical_cols) / n_cols))

plt.figure(figsize=(n_cols * 4, n_rows * 3))
for i, col in enumerate(numerical_cols):
    plt.subplot(n_rows, n_cols, i + 1)
    sns.histplot(df[col], kde=True, bins=30, color='skyblue')
    plt.title(f'Distribution of {col}', fontsize=12)
    plt.xlabel(col)
    plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# --- 4) Create box plots comparing each feature between diabetic and non-diabetic patients ---
plt.figure(figsize=(n_cols * 4, n_rows * 3))
for i, col in enumerate(numerical_cols):
    plt.subplot(n_rows, n_cols, i + 1)
    sns.boxplot(x='Outcome', y=col, data=df, palette='viridis')
    plt.title(f'{col} by Diabetes Outcome', fontsize=12)
    plt.xlabel('Diabetes Outcome (0: Non-Diabetic, 1: Diabetic)')
    plt.ylabel(col)
plt.tight_layout()
plt.show()

# --- 5) Plot a pie chart showing the distribution of diabetes outcomes ---
outcome_counts = df['Outcome'].value_counts()
labels = ['Non-Diabetic (0)', 'Diabetic (1)']
colors = ['lightcoral', 'lightskyblue']
explode = (0.05, 0)  # Explode the first slice (Non-Diabetic) for emphasis

plt.figure(figsize=(7, 7))
plt.pie(outcome_counts, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors, explode=explode, shadow=True)
plt.title('Distribution of Diabetes Outcomes', fontsize=16)
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()

# --- 6) Create scatter plots showing the relationship between BMI vs Glucose and Age vs Pregnancies, colored by diabetes outcome ---
plt.figure(figsize=(14, 6))

# Scatter plot: BMI vs Glucose
plt.subplot(1, 2, 1)
sns.scatterplot(x='BMI', y='Glucose', hue='Outcome', data=df, palette='coolwarm', s=50, alpha=0.7)
plt.title('BMI vs Glucose, Colored by Diabetes Outcome', fontsize=14)
plt.xlabel('Body Mass Index (BMI)')
plt.ylabel('Glucose Level')
plt.legend(title='Outcome', labels=['Non-Diabetic', 'Diabetic'])

# Scatter plot: Age vs Pregnancies
plt.subplot(1, 2, 2)
sns.scatterplot(x='Age', y='Pregnancies', hue='Outcome', data=df, palette='coolwarm', s=50, alpha=0.7)
plt.title('Age vs Pregnancies, Colored by Diabetes Outcome', fontsize=14)
plt.xlabel('Age')
plt.ylabel('Number of Pregnancies')
plt.legend(title='Outcome', labels=['Non-Diabetic', 'Diabetic'])

plt.tight_layout()
plt.show()

# --- 7) Generate a pair plot to visualize relationships between key features (Glucose, BMI, Age, Outcome) ---
# Select the key features for the pair plot
key_features = ['Glucose', 'BMI', 'Age', 'Outcome']

# Create the pair plot
# The 'hue' parameter colors the points based on the 'Outcome' variable
# The 'diag_kind' parameter specifies the type of plot for the diagonal subplots (histograms in this case)
# The 'palette' parameter sets the color scheme
sns.pairplot(df[key_features], hue='Outcome', diag_kind='kde', palette='viridis')
plt.suptitle('Pair Plot of Key Features by Diabetes Outcome', y=1.02, fontsize=16) # Adjust y to prevent overlap
plt.show()

print("\nAll requested visualizations have been generated.")
```