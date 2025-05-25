```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Load the mobiles_2025.csv dataset ---
# Assumes 'mobiles_2025.csv' is in the current working directory.
try:
    df = pd.read_csv('mobiles_2025.csv')
except FileNotFoundError:
    print("Error: 'mobiles_2025.csv' not found. Please ensure the file is in the current directory.")
    # Exit the script or handle the error appropriately if the file is crucial.
    exit()

# --- 2. Clean the 'RAM' column ---
# Remove 'GB' and convert to numeric. Handle any conversion errors by setting to NaN.
# Convert column to string type first to ensure .str accessor works, then remove 'GB' and strip whitespace.
df['RAM_cleaned'] = df['RAM'].astype(str).str.replace('GB', '', regex=False).str.strip()
# Convert the cleaned string column to numeric, coercing any errors (non-numeric values) to NaN.
df['RAM_cleaned'] = pd.to_numeric(df['RAM_cleaned'], errors='coerce')

# --- 3. Clean the 'Battery Capacity' column ---
# Remove 'mAh' and convert to numeric. Handle any conversion errors by setting to NaN.
# Convert column to string type first, then remove 'mAh' and strip whitespace.
df['Battery Capacity_cleaned'] = df['Battery Capacity'].astype(str).str.replace('mAh', '', regex=False).str.strip()
# Convert the cleaned string column to numeric, coercing any errors to NaN.
df['Battery Capacity_cleaned'] = pd.to_numeric(df['Battery Capacity_cleaned'], errors='coerce')

# --- 4. Create a scatter plot ---
# Visualize the relationship between cleaned Battery Capacity (y-axis) and cleaned RAM (x-axis).
# Color the points in the scatter plot according to the 'Company Name' column.
plt.figure(figsize=(12, 8)) # Set a larger figure size for better readability

# Use seaborn's scatterplot for enhanced aesthetics and automatic handling of categorical 'hue'.
sns.scatterplot(
    data=df,
    x='RAM_cleaned',
    y='Battery Capacity_cleaned',
    hue='Company Name', # Color points based on 'Company Name'
    s=100, # Set marker size for better visibility
    alpha=0.7 # Add transparency to points, useful for overlapping data
)

# --- 5. Ensure the plot has a title, and labels for the x-axis and y-axis ---
plt.title('Battery Capacity vs. RAM by Company', fontsize=16)
plt.xlabel('RAM (GB)', fontsize=14)
plt.ylabel('Battery Capacity (mAh)', fontsize=14)

# Add a grid for easier reading of values
plt.grid(True, linestyle='--', alpha=0.6)

# Adjust layout to prevent labels/title from being cut off
plt.tight_layout()

# Display the plot
plt.show()
```