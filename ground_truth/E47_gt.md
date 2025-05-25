```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set a professional style for the plots using seaborn
sns.set_theme(style="whitegrid", palette="viridis")

# --- 1. Load the dataset ---
# Assume 'penguins.csv' is in the current directory
try:
    df = pd.read_csv('penguins.csv')
    print("Dataset 'penguins.csv' loaded successfully.")
    print(f"Initial dataset shape: {df.shape}")
except FileNotFoundError:
    print("Error: 'penguins.csv' not found. Please ensure the file is in the current directory.")
    # Exit the script if the file is not found, as subsequent operations will fail
    exit()
except Exception as e:
    print(f"An error occurred while loading the dataset: {e}")
    exit()

# --- 2. Drop rows where `flipper_length_mm` is NA ---
# It's good practice to check for NAs before dropping
initial_rows = df.shape[0]
na_flipper_length = df['flipper_length_mm'].isna().sum()
if na_flipper_length > 0:
    df_cleaned = df.dropna(subset=['flipper_length_mm'])
    rows_after_na_drop = df_cleaned.shape[0]
    print(f"Found {na_flipper_length} NA values in 'flipper_length_mm'.")
    print(f"Dropped {initial_rows - rows_after_na_drop} rows due to NA in 'flipper_length_mm'.")
    print(f"Dataset shape after dropping NAs: {df_cleaned.shape}")
else:
    df_cleaned = df.copy() # Create a copy to avoid SettingWithCopyWarning later
    print("No NA values found in 'flipper_length_mm'. No rows dropped.")

# --- 3. Create a histogram of the `flipper_length_mm` column ---
plt.figure(figsize=(10, 6)) # Set figure size for better readability
# Use seaborn's histplot for enhanced visualization, including KDE for distribution shape
sns.histplot(data=df_cleaned, x='flipper_length_mm', kde=True, bins=20)
plt.title('Distribution of Flipper Length (mm)', fontsize=16)
plt.xlabel('Flipper Length (mm)', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.7) # Add a horizontal grid for easier reading of counts
plt.tight_layout() # Adjust layout to prevent labels from overlapping
plt.show()

# --- 4. Create a bar chart showing the counts of penguins for each `island` ---
plt.figure(figsize=(8, 6)) # Set figure size
# Use seaborn's countplot which directly calculates and plots counts of categories
sns.countplot(data=df_cleaned, x='island', order=df_cleaned['island'].value_counts().index)
plt.title('Penguin Counts by Island', fontsize=16)
plt.xlabel('Island', fontsize=12)
plt.ylabel('Number of Penguins', fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.7) # Add a horizontal grid
plt.tight_layout()
plt.show()

print("\nAll visualizations generated successfully.")
```