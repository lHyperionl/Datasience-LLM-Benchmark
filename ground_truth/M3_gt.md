```python
import pandas as pd
import numpy as np

# --- 1. Load the dataset ---
try:
    df = pd.read_csv('winequality-red.csv')
    print("Dataset 'winequality-red.csv' loaded successfully.")
    print(f"Original dataset shape: {df.shape}")
    print("\nFirst 5 rows of the original dataset:")
    print(df.head())
except FileNotFoundError:
    print("Error: 'winequality-red.csv' not found. Please ensure the file is in the current directory.")
    exit() # Exit the script if the file is not found

# Create a copy of the DataFrame for winsorization to preserve the original data
df_winsorized = df.copy()

# Define the columns for outlier treatment
columns_to_treat = ['volatile acidity', 'residual sugar']

print("\n--- 2. Outlier Identification and Reporting (using IQR method) ---")

# Dictionary to store bounds for each column
bounds = {}

for col in columns_to_treat:
    print(f"\nProcessing column: '{col}'")

    # Calculate Q1 (25th percentile) and Q3 (75th percentile)
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)

    # Calculate the Interquartile Range (IQR)
    IQR = Q3 - Q1

    # Calculate the lower and upper bounds for outlier detection
    # Outliers are typically defined as values outside Q1 - 1.5*IQR and Q3 + 1.5*IQR
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Store bounds for later use in winsorization
    bounds[col] = {'lower': lower_bound, 'upper': upper_bound}

    print(f"  Q1 (25th percentile): {Q1:.4f}")
    print(f"  Q3 (75th percentile): {Q3:.4f}")
    print(f"  IQR: {IQR:.4f}")
    print(f"  Lower Bound (Q1 - 1.5*IQR): {lower_bound:.4f}")
    print(f"  Upper Bound (Q3 + 1.5*IQR): {upper_bound:.4f}")

    # Identify potential outliers
    # Outliers are values less than the lower bound or greater than the upper bound
    outliers_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
    num_outliers = df[outliers_mask].shape[0]

    # Report the number of outliers found
    print(f"  Number of outliers identified in '{col}': {num_outliers}")

    # --- 3. Create a new dataset version where identified outliers are capped (winsorized) ---
    # Apply winsorization:
    # Values below the lower bound are set to the lower bound
    # Values above the upper bound are set to the upper bound
    df_winsorized[col] = np.where(df_winsorized[col] < lower_bound, lower_bound, df_winsorized[col])
    df_winsorized[col] = np.where(df_winsorized[col] > upper_bound, upper_bound, df_winsorized[col])
    print(f"  Outliers in '{col}' have been capped (winsorized) in 'df_winsorized'.")

print("\n--- 4. Compare Mean and Standard Deviation Before and After Outlier Treatment ---")

for col in columns_to_treat:
    print(f"\nComparison for column: '{col}'")

    # Original data statistics
    original_mean = df[col].mean()
    original_std = df[col].std()

    # Winsorized data statistics
    winsorized_mean = df_winsorized[col].mean()
    winsorized_std = df_winsorized[col].std()

    print(f"  Original Data:")
    print(f"    Mean: {original_mean:.4f}")
    print(f"    Standard Deviation: {original_std:.4f}")

    print(f"  Winsorized Data:")
    print(f"    Mean: {winsorized_mean:.4f}")
    print(f"    Standard Deviation: {winsorized_std:.4f}")

    # Calculate percentage change for better comparison
    mean_change_percent = ((winsorized_mean - original_mean) / original_mean) * 100 if original_mean != 0 else 0
    std_change_percent = ((winsorized_std - original_std) / original_std) * 100 if original_std != 0 else 0

    print(f"  Change in Mean: {mean_change_percent:.2f}%")
    print(f"  Change in Standard Deviation: {std_change_percent:.2f}%")

print("\n--- Outlier Treatment Complete ---")
print("\nFirst 5 rows of the winsorized dataset:")
print(df_winsorized[columns_to_treat].head())
```