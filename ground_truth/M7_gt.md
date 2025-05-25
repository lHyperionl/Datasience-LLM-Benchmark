```python
import pandas as pd
import numpy as np
from scipy import stats

# --- Configuration ---
ALPHA = 0.05 # Significance level for all hypothesis tests

# --- 1. Load Data and Handle Missing Values ---
try:
    df = pd.read_csv('water_potability.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: 'water_potability.csv' not found. Please ensure the file is in the current directory.")
    exit()

# Impute missing values with the mean of their respective columns.
# This is a common and simple approach for numerical data in statistical analysis.
print("\n--- Handling Missing Values ---")
initial_missing_counts = df.isnull().sum()
print("Missing values before imputation:\n", initial_missing_counts[initial_missing_counts > 0])

# Identify numerical columns for imputation, excluding 'Potability' as it's a target/categorical.
numerical_cols_for_imputation = df.select_dtypes(include=np.number).columns.drop('Potability', errors='ignore')
for col in numerical_cols_for_imputation:
    if df[col].isnull().any():
        df[col].fillna(df[col].mean(), inplace=True)
        print(f"Missing values in '{col}' imputed with mean.")

final_missing_counts = df.isnull().sum()
if final_missing_counts.sum() == 0:
    print("All missing values handled.")
else:
    print("Missing values after imputation:\n", final_missing_counts[final_missing_counts > 0])

# Separate data into potable and non-potable groups based on the 'Potability' column.
potable_df = df[df['Potability'] == 1]
non_potable_df = df[df['Potability'] == 0]

# Identify numerical features for analysis (excluding 'Potability' itself).
numerical_features = df.columns.drop('Potability').tolist()

# --- 2. Normality Tests (Shapiro-Wilk) ---
print("\n--- 2. Normality Tests (Shapiro-Wilk) ---")
print(f"Significance level (α) for normality tests: {ALPHA}")

for feature in numerical_features:
    # Shapiro-Wilk test is suitable for sample sizes up to 5000.
    # For larger datasets, other tests like Kolmogorov-Smirnov or Anderson-Darling might be more appropriate,
    # or one might sample a subset.
    if len(df[feature]) > 5000:
        print(f"\nSkipping Shapiro-Wilk for '{feature}' due to large sample size ({len(df[feature])}).")
        print("  Consider alternative tests like Kolmogorov-Smirnov or Anderson-Darling.")
        continue
    
    stat, p_value = stats.shapiro(df[feature])
    print(f"\nFeature: {feature}")
    print(f"  Shapiro-Wilk Statistic: {stat:.4f}")
    print(f"  P-value: {p_value:.4f}")
    if p_value < ALPHA:
        print(f"  Interpretation: P-value < {ALPHA}. Reject the null hypothesis.")
        print(f"  The data for '{feature}' is NOT normally distributed.")
    else:
        print(f"  Interpretation: P-value >= {ALPHA}. Fail to reject the null hypothesis.")
        print(f"  The data for '{feature}' IS normally distributed (or not enough evidence to say otherwise).")

# --- 3. T-tests to Compare Mean Values & Calculate Cohen's d ---
print("\n--- 3. T-tests (Independent Samples) and Cohen's d ---")
print(f"Significance level (α) for t-tests: {ALPHA}")

for feature in numerical_features:
    # Extract data for the current feature for both potable and non-potable groups.
    # .dropna() is used here to ensure no NaNs are passed to the t-test, though imputation should have handled most.
    data_potable = potable_df[feature].dropna()
    data_non_potable = non_potable_df[feature].dropna()

    # Ensure there are enough samples in each group to perform a t-test.
    if len(data_potable) < 2 or len(data_non_potable) < 2:
        print(f"\nSkipping t-test for '{feature}': Not enough samples in one or both groups ({len(data_potable)} vs {len(data_non_potable)}).")
        continue

    # Perform independent samples t-test (Welch's t-test).
    # Welch's t-test (equal_var=False) is more robust when population variances are unequal,
    # which is often a safer assumption in real-world data.
    t_stat, p_value = stats.ttest_ind(data_potable, data_non_potable, equal_var=False)

    print(f"\n--- Comparing '{feature}' between Potable and Non-Potable Water ---")
    print(f"  Mean Potable: {data_potable.mean():.4f}")
    print(f"  Mean Non-Potable: {data_non_potable.mean():.4f}")
    print(f"  T-statistic: {t_stat:.4f}")
    print(f"  P-value: {p_value:.4f}")

    # Interpretation of t-test results
    if p_value < ALPHA:
        print(f"  Interpretation: P-value < {ALPHA}. Reject the null hypothesis.")
        print(f"  There is a statistically significant difference in '{feature}' between potable and non-potable water samples.")
        
        # Calculate Cohen's d for effect size.
        # Cohen's d = (Mean1 - Mean2) / Pooled Standard Deviation
        n1, n2 = len(data_potable), len(data_non_potable)
        s1, s2 = data_potable.std(), data_non_potable.std()
        
        # Calculate pooled standard deviation, handling edge cases for division by zero.
        if n1 + n2 - 2 <= 0: # Avoid division by zero if sample sizes are too small
            pooled_std = np.nan
            cohens_d = np.nan
        elif s1 == 0 and s2 == 0: # If both standard deviations are zero, means are identical, so d=0
            pooled_std = 0.0
            cohens_d = 0.0
        else:
            pooled_std = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
            if pooled_std == 0: # If pooled_std is zero (e.g., all values are identical), Cohen's d is 0
                cohens_d = 0.0
            else:
                cohens_d = (data_potable.mean() - data_non_potable.mean()) / pooled_std
        
        print(f"  Cohen's d (Effect Size): {cohens_d:.4f}")
        # Interpret Cohen's d based on common guidelines
        if abs(cohens_d) < 0.2:
            print("  Effect Size Interpretation: Trivial effect.")
        elif abs(cohens_d) < 0.5:
            print("  Effect Size Interpretation: Small effect.")
        elif abs(cohens_d) < 0.8:
            print("  Effect Size Interpretation: Medium effect.")
        else:
            print("  Effect Size Interpretation: Large effect.")
            
    else:
        print(f"  Interpretation: P-value >= {ALPHA}. Fail to reject the null hypothesis.")
        print(f"  There is NO statistically significant difference in '{feature}' between potable and non-potable water samples.")
        print("  Cohen's d is not typically interpreted when the difference is not statistically significant.")

# --- 4. Chi-square Goodness-of-Fit Test for Potability Distribution ---
print("\n--- 4. Chi-square Goodness-of-Fit Test for Potability Distribution ---")
print(f"Significance level (α) for chi-square test: {ALPHA}")

# Get observed frequencies of the 'Potability' column.
# .sort_index() ensures consistent order (0 then 1).
observed_counts = df['Potability'].value_counts().sort_index()
total_samples = len(df)

# Define expected frequencies. The prompt asks to check if potability follows an "expected distribution".
# Without a specified expected distribution, a common default for a binary variable is a uniform distribution (50/50).
expected_counts = [total_samples / 2, total_samples / 2]

print(f"Observed Potability Counts (0=Non-Potable, 1=Potable):\n{observed_counts}")
print(f"Expected Potability Counts (assuming 50/50 distribution):\n{expected_counts}")

# Perform Chi-square Goodness-of-Fit test.
# f_obs: observed frequencies, f_exp: expected frequencies.
chi2_stat, p_value = stats.chisquare(f_obs=observed_counts, f_exp=expected_counts)

print(f"  Chi-square Statistic: {chi2_stat:.4f}")
print(f"  P-value: {p_value:.4f}")

# Interpretation of Chi-square test results.
if p_value < ALPHA:
    print(f"  Interpretation: P-value < {ALPHA}. Reject the null hypothesis.")
    print(f"  The observed distribution of 'Potability' is significantly different from the expected uniform (50/50) distribution.")
else:
    print(f"  Interpretation: P-value >= {ALPHA}. Fail to reject the null hypothesis.")
    print(f"  The observed distribution of 'Potability' is NOT significantly different from the expected uniform (50/50) distribution.")

print("\n--- Statistical Analysis Complete ---")
```