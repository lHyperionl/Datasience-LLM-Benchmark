```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set a consistent style for plots
sns.set_style("whitegrid")

# --- 1. Load the dataset ---
try:
    df = pd.read_csv('water_potability.csv')
except FileNotFoundError:
    print("Error: 'water_potability.csv' not found. Make sure the file is in the current directory.")
    exit()

# --- 2. Create a bar chart showing the distribution of potable vs non-potable water samples ---
plt.figure(figsize=(8, 6))
sns.countplot(x='Potability', data=df, palette='viridis')
plt.title('Distribution of Water Potability', fontsize=16)
plt.xlabel('Potability (0: Not Potable, 1: Potable)', fontsize=12)
plt.ylabel('Number of Samples', fontsize=12)
plt.xticks(ticks=[0, 1], labels=['Not Potable', 'Potable'])
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# --- 3. Create histograms for pH, Hardness, and Chloramines showing their distributions ---
# Handle missing values for plotting: seaborn's histplot automatically ignores NaNs.
# Create a figure with subplots for better organization
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Histogram for pH
sns.histplot(df['ph'].dropna(), kde=True, ax=axes[0], color='skyblue', bins=30)
axes[0].set_title('Distribution of pH Levels', fontsize=14)
axes[0].set_xlabel('pH Value', fontsize=12)
axes[0].set_ylabel('Frequency', fontsize=12)
axes[0].grid(axis='y', linestyle='--', alpha=0.7)

# Histogram for Hardness
sns.histplot(df['Hardness'].dropna(), kde=True, ax=axes[1], color='lightcoral', bins=30)
axes[1].set_title('Distribution of Hardness', fontsize=14)
axes[1].set_xlabel('Hardness (mg/L)', fontsize=12)
axes[1].set_ylabel('Frequency', fontsize=12)
axes[1].grid(axis='y', linestyle='--', alpha=0.7)

# Histogram for Chloramines
sns.histplot(df['Chloramines'].dropna(), kde=True, ax=axes[2], color='lightgreen', bins=30)
axes[2].set_title('Distribution of Chloramines', fontsize=14)
axes[2].set_xlabel('Chloramines (mg/L)', fontsize=12)
axes[2].set_ylabel('Frequency', fontsize=12)
axes[2].grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

# --- 4. Create a boxplot comparing pH levels between potable and non-potable water ---
plt.figure(figsize=(8, 6))
# seaborn's boxplot automatically handles NaNs by ignoring them
sns.boxplot(x='Potability', y='ph', data=df, palette='coolwarm')
plt.title('pH Levels by Water Potability', fontsize=16)
plt.xlabel('Potability (0: Not Potable, 1: Potable)', fontsize=12)
plt.ylabel('pH Value', fontsize=12)
plt.xticks(ticks=[0, 1], labels=['Not Potable', 'Potable'])
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# --- 5. Generate a simple correlation heatmap for all numerical features ---
# Calculate the correlation matrix. Pandas' .corr() method handles NaNs by computing pairwise correlations.
# For a heatmap, it's generally acceptable to let .corr() handle NaNs this way.
correlation_matrix = df.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Heatmap of Water Potability Features', fontsize=16)
plt.xticks(rotation=45, ha='right') # Rotate x-axis labels for better readability
plt.yticks(rotation=0) # Keep y-axis labels horizontal
plt.tight_layout()
plt.show()
```