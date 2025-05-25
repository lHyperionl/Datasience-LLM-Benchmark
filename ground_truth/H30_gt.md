```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
import numpy as np

# --- 1. Load the dataset ---
# Assume 'heart.csv' is in the current directory
try:
    df = pd.read_csv('heart.csv')
    print("Dataset loaded successfully. First 5 rows:")
    print(df.head())
    print("\nDataset information:")
    df.info()
    print("\n")
except FileNotFoundError:
    print("Error: 'heart.csv' not found. Please ensure the file is in the current directory.")
    exit() # Exit if the dataset cannot be loaded

# Ensure 'Oldpeak' is numeric and 'HeartDisease' is binary (0 or 1)
# Check for missing values in relevant columns
if df[['Oldpeak', 'HeartDisease']].isnull().any().any():
    print("Warning: Missing values found in 'Oldpeak' or 'HeartDisease'. Dropping rows with NaNs in these columns.")
    df.dropna(subset=['Oldpeak', 'HeartDisease'], inplace=True)

# --- 2. Create a box plot showing the distribution of Oldpeak for patients with HeartDisease=0 and HeartDisease=1 ---
plt.figure(figsize=(8, 6))
sns.boxplot(x='HeartDisease', y='Oldpeak', data=df)
plt.title('Distribution of Oldpeak by Heart Disease Status')
plt.xlabel('Heart Disease (0: No, 1: Yes)')
plt.ylabel('Oldpeak (ST depression induced by exercise relative to rest)')
plt.xticks(ticks=[0, 1], labels=['No Heart Disease', 'Heart Disease'])
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

print("\n--- Box plot displayed: Distribution of Oldpeak by Heart Disease Status ---\n")

# --- 3. Perform a logistic regression with only Oldpeak as the independent variable to predict HeartDisease ---
# Define independent variable (X) and dependent variable (y)
X = df[['Oldpeak']] # X must be 2D for scikit-learn
y = df['HeartDisease']

# Initialize and train the logistic regression model
model = LogisticRegression()
model.fit(X, y)

# Report the coefficient for Oldpeak
oldpeak_coefficient = model.coef_[0][0]
print(f"Logistic Regression Coefficient for Oldpeak: {oldpeak_coefficient:.4f}")

# Interpret its meaning in the context of odds ratios
# The odds ratio is exp(coefficient)
odds_ratio = np.exp(oldpeak_coefficient)
print(f"Odds Ratio for Oldpeak: {odds_ratio:.4f}")

# Interpretation
if oldpeak_coefficient > 0:
    print(f"Interpretation: For every one-unit increase in Oldpeak, the odds of having Heart Disease are multiplied by a factor of {odds_ratio:.4f} (i.e., they increase by approximately {(odds_ratio - 1) * 100:.2f}%), holding other factors constant.")
elif oldpeak_coefficient < 0:
    print(f"Interpretation: For every one-unit increase in Oldpeak, the odds of having Heart Disease are multiplied by a factor of {odds_ratio:.4f} (i.e., they decrease by approximately {(1 - odds_ratio) * 100:.2f}%), holding other factors constant.")
else:
    print("Interpretation: The coefficient for Oldpeak is zero, indicating no linear relationship with the log-odds of Heart Disease.")

print("\n--- Logistic Regression analysis complete ---\n")

# --- 4. Discretize the Oldpeak column into three equally sized bins (low, medium, high) ---
# Use pd.qcut for quantile-based binning
# Handle potential issues with duplicate edges if Oldpeak has many identical values
try:
    df['Oldpeak_Bin'] = pd.qcut(df['Oldpeak'], q=3, labels=['Low', 'Medium', 'High'], duplicates='drop')
    print("Oldpeak column discretized into 3 quantile-based bins (Low, Medium, High).")
    print("Distribution of Oldpeak_Bin:")
    print(df['Oldpeak_Bin'].value_counts().sort_index())
    print("\n")
except ValueError as e:
    print(f"Error during qcut: {e}")
    print("This might happen if there are too many identical values in 'Oldpeak' preventing unique quantiles.")
    print("Falling back to fixed-width binning if quantile binning fails (less ideal for 'equally sized').")
    # Fallback to fixed-width binning if qcut fails due to duplicate edges
    min_oldpeak = df['Oldpeak'].min()
    max_oldpeak = df['Oldpeak'].max()
    bins = np.linspace(min_oldpeak, max_oldpeak, 4) # 3 bins, 4 edges
    df['Oldpeak_Bin'] = pd.cut(df['Oldpeak'], bins=bins, labels=['Low', 'Medium', 'High'], include_lowest=True)
    print("Used pd.cut (fixed-width) as a fallback for Oldpeak binning.")
    print("Distribution of Oldpeak_Bin (fallback):")
    print(df['Oldpeak_Bin'].value_counts().sort_index())
    print("\n")


# --- 5. Calculate and display the proportion of patients with HeartDisease=1 within each of these Oldpeak bins ---
# Group by the new 'Oldpeak_Bin' and calculate the mean of 'HeartDisease'
# The mean of a binary (0/1) column represents the proportion of 1s
proportion_heart_disease_by_bin = df.groupby('Oldpeak_Bin')['HeartDisease'].mean()

print("Proportion of patients with Heart Disease (1) within each Oldpeak bin:")
print(proportion_heart_disease_by_bin)

# Optional: Visualize the proportions
plt.figure(figsize=(8, 5))
proportion_heart_disease_by_bin.plot(kind='bar', color='skyblue')
plt.title('Proportion of Heart Disease by Oldpeak Bin')
plt.xlabel('Oldpeak Bin')
plt.ylabel('Proportion of Heart Disease (1)')
plt.ylim(0, 1) # Proportions are between 0 and 1
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

print("\n--- Analysis complete: Oldpeak binning and Heart Disease proportion calculated and displayed ---")
```