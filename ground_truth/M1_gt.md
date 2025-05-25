```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress
import numpy as np

# --- Configuration ---
DATASET_PATH = 'winequality-red.csv'
TARGET_FEATURE = 'quality'
HEATMAP_FIGSIZE = (12, 10)
SCATTER_FIGSIZE = (8, 6)
SCATTER_DPI = 100

# --- 1. Load the dataset ---
try:
    df = pd.read_csv(DATASET_PATH)
    print(f"Dataset '{DATASET_PATH}' loaded successfully. Shape: {df.shape}")
    print("First 5 rows of the dataset:")
    print(df.head())
    print("\nDataset Info:")
    df.info()
    print("\nDescriptive Statistics:")
    print(df.describe())
except FileNotFoundError:
    print(f"Error: The file '{DATASET_PATH}' was not found.")
    print("Please ensure the CSV file is in the same directory as the script.")
    exit()
except Exception as e:
    print(f"An error occurred while loading the dataset: {e}")
    exit()

# --- 2. Calculate the correlation matrix ---
# Calculate Pearson correlation matrix for all numerical features
correlation_matrix = df.corr()

# --- 3. Determine correlations with 'quality' ---
# Extract correlations of all features with the 'quality' target
quality_correlations = correlation_matrix[TARGET_FEATURE].sort_values(ascending=False)

# Drop the correlation of 'quality' with itself
quality_correlations = quality_correlations.drop(TARGET_FEATURE)

print(f"\n--- Feature Correlations with '{TARGET_FEATURE}' ---")
print(quality_correlations)

# --- 4. List top 3 positively and top 3 negatively correlated features ---
print(f"\n--- Top 3 Positively Correlated Features with '{TARGET_FEATURE}' ---")
top_3_positive = quality_correlations.head(3)
print(top_3_positive)

print(f"\n--- Top 3 Negatively Correlated Features with '{TARGET_FEATURE}' ---")
top_3_negative = quality_correlations.tail(3)
print(top_3_negative)

# --- 5. Create a correlation matrix heatmap ---
plt.figure(figsize=HEATMAP_FIGSIZE)
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix of Wine Quality Features', fontsize=16)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# --- 6. Create scatter plots with linear regression lines ---

# Get the single most positively and most negatively correlated features
most_positive_feature = top_3_positive.index[0]
most_negative_feature = top_3_negative.index[0]

print(f"\n--- Generating Scatter Plots with Regression Lines ---")
print(f"Most positively correlated feature: '{most_positive_feature}' (Correlation: {quality_correlations[most_positive_feature]:.2f})")
print(f"Most negatively correlated feature: '{most_negative_feature}' (Correlation: {quality_correlations[most_negative_feature]:.2f})")

# Plot for the most positively correlated feature
plt.figure(figsize=SCATTER_FIGSIZE, dpi=SCATTER_DPI)
sns.scatterplot(x=df[most_positive_feature], y=df[TARGET_FEATURE], alpha=0.6)

# Fit linear regression
slope_pos, intercept_pos, r_value_pos, p_value_pos, std_err_pos = linregress(df[most_positive_feature], df[TARGET_FEATURE])
# Create regression line
x_pos = np.array([df[most_positive_feature].min(), df[most_positive_feature].max()])
plt.plot(x_pos, intercept_pos + slope_pos * x_pos, color='red', label=f'Regression Line (R²={r_value_pos**2:.2f})')

plt.title(f'Scatter Plot: {most_positive_feature} vs. {TARGET_FEATURE}', fontsize=14)
plt.xlabel(most_positive_feature, fontsize=12)
plt.ylabel(TARGET_FEATURE, fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.show()

# Plot for the most negatively correlated feature
plt.figure(figsize=SCATTER_FIGSIZE, dpi=SCATTER_DPI)
sns.scatterplot(x=df[most_negative_feature], y=df[TARGET_FEATURE], alpha=0.6)

# Fit linear regression
slope_neg, intercept_neg, r_value_neg, p_value_neg, std_err_neg = linregress(df[most_negative_feature], df[TARGET_FEATURE])
# Create regression line
x_neg = np.array([df[most_negative_feature].min(), df[most_negative_feature].max()])
plt.plot(x_neg, intercept_neg + slope_neg * x_neg, color='red', label=f'Regression Line (R²={r_value_neg**2:.2f})')

plt.title(f'Scatter Plot: {most_negative_feature} vs. {TARGET_FEATURE}', fontsize=14)
plt.xlabel(most_negative_feature, fontsize=12)
plt.ylabel(TARGET_FEATURE, fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.show()

print("\nAnalysis complete. All plots generated.")
```