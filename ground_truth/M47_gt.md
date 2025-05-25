```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load the dataset
try:
    df = pd.read_csv('penguins.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: 'penguins.csv' not found. Please ensure the CSV file is in the current directory.")
    exit() # Exit if the file is not found

# 2. Drop rows where specified columns are NA
# Define the columns to check for NA values
columns_to_check_na = ['culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g']

# Drop rows where any of the specified columns have NA values
df_cleaned = df.dropna(subset=columns_to_check_na).copy() # .copy() to avoid SettingWithCopyWarning

print(f"\nOriginal number of rows: {len(df)}")
print(f"Number of rows after dropping NAs: {len(df_cleaned)}")

# Check if the cleaned DataFrame is empty after dropping NAs
if df_cleaned.empty:
    print("Error: No data remaining after dropping NA values. Cannot proceed with analysis.")
    exit()

# 3. Create a new feature `culmen_area`
df_cleaned['culmen_area'] = df_cleaned['culmen_length_mm'] * df_cleaned['culmen_depth_mm']
print("\n'culmen_area' feature created successfully.")

# 4. Calculate the Pearson correlation matrix
# Define the columns for correlation analysis
correlation_columns = [
    'culmen_length_mm',
    'culmen_depth_mm',
    'flipper_length_mm',
    'body_mass_g',
    'culmen_area'
]

# Ensure all correlation columns exist in the DataFrame
missing_cols = [col for col in correlation_columns if col not in df_cleaned.columns]
if missing_cols:
    print(f"Error: The following columns are missing from the DataFrame: {missing_cols}")
    exit()

# Calculate the correlation matrix
correlation_matrix = df_cleaned[correlation_columns].corr(method='pearson')
print("\nPearson Correlation Matrix calculated:")
print(correlation_matrix)

# 5. Visualize the correlation matrix as a heatmap
plt.figure(figsize=(8, 6)) # Set the figure size for better readability
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix of Penguin Measurements', fontsize=16)
plt.xticks(rotation=45, ha='right') # Rotate x-axis labels for better readability
plt.yticks(rotation=0) # Keep y-axis labels horizontal
plt.tight_layout() # Adjust layout to prevent labels from overlapping
plt.show()

print("\nCorrelation matrix heatmap displayed successfully.")
```