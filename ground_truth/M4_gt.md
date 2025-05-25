```python
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency, ttest_ind, f_oneway
import math

# Define the significance level for all hypothesis tests
alpha = 0.05

# --- 1. Load the dataset ---
try:
    df = pd.read_csv('titanic.csv')
    print("Dataset 'titanic.csv' loaded successfully.")
except FileNotFoundError:
    print("Error: 'titanic.csv' not found. Please ensure the file is in the current directory.")
    # Exit the script if the file is not found
    exit()

print("\n--- Statistical Analysis and Hypothesis Testing on Titanic Dataset ---")
print(f"Significance Level (alpha): {alpha}\n")

# --- 2. Chi-square test: Association between Passenger Class (Pclass) and Survival ---
# Hypothesis:
# H0: There is no significant association between Pclass and Survival.
# Ha: There is a significant association between Pclass and Survival.
print("--- Chi-square Test: Association between Passenger Class (Pclass) and Survival ---")

# Create a contingency table of Pclass and Survived
# This table shows the frequency of each combination of Pclass and Survived.
contingency_table = pd.crosstab(df['Pclass'], df['Survived'])
print("Contingency Table (Pclass vs. Survived):\n", contingency_table)

# Perform the Chi-square test for independence
# Returns: chi2 statistic, p-value, degrees of freedom, expected frequencies
chi2_stat, p_value_chi2, dof, expected_freq = chi2_contingency(contingency_table)

# Calculate Cramer's V as the effect size for the Chi-square test
# Cramer's V measures the strength of association between two nominal variables.
# Formula: sqrt(chi2 / (n * min(rows-1, cols-1)))
n = contingency_table.sum().sum() # Total number of observations
min_dim = min(contingency_table.shape) - 1 # Minimum of (rows - 1) or (columns - 1)
cramers_v = np.sqrt(chi2_stat / (n * min_dim)) if (n * min_dim) > 0 else 0

print(f"\nChi-square Statistic: {chi2_stat:.3f}")
print(f"P-value: {p_value_chi2:.3f}")
print(f"Degrees of Freedom: {dof}")
print(f"Cramer's V (Effect Size): {cramers_v:.3f}")

# Interpret the result based on the p-value and alpha
if p_value_chi2 < alpha:
    print(f"Conclusion: Since p-value ({p_value_chi2:.3f}) < alpha ({alpha}), we reject the null hypothesis.")
    print("There is a statistically significant association between Passenger Class (Pclass) and Survival.")
else:
    print(f"Conclusion: Since p-value ({p_value_chi2:.3f}) >= alpha ({alpha}), we fail to reject the null hypothesis.")
    print("There is no statistically significant association between Passenger Class (Pclass) and Survival.")
print("-" * 70 + "\n")

# --- 3. T-test: Compare ages of survivors vs non-survivors ---
# Hypothesis:
# H0: The mean age of survivors is equal to the mean age of non-survivors.
# Ha: The mean age of survivors is different from the mean age of non-survivors.
print("--- Independent Samples T-test: Ages of Survivors vs. Non-survivors ---")

# Drop rows where 'Age' is missing, as it's crucial for this analysis
df_age_cleaned = df.dropna(subset=['Age'])

# Separate ages into two groups based on 'Survived' status
survivor_ages = df_age_cleaned[df_age_cleaned['Survived'] == 1]['Age']
non_survivor_ages = df_age_cleaned[df_age_cleaned['Survived'] == 0]['Age']

# Check if either group is empty after cleaning
if survivor_ages.empty or non_survivor_ages.empty:
    print("Error: One or both age groups are empty after handling missing values. Cannot perform t-test.")
else:
    print(f"Mean Age of Survivors: {survivor_ages.mean():.2f} (N={len(survivor_ages)})")
    print(f"Mean Age of Non-Survivors: {non_survivor_ages.mean():.2f} (N={len(non_survivor_ages)})")

    # Perform Levene's test for equality of variances (assumption for t-test)
    # H0: Variances are equal
    # Ha: Variances are not equal
    stat_levene, p_value_levene = stats.levene(survivor_ages, non_survivor_ages)
    print(f"\nLevene's Test for Equality of Variances: Statistic={stat_levene:.3f}, P-value={p_value_levene:.3f}")

    # Decide whether to assume equal variances for the t-test based on Levene's test p-value
    # If p_value_levene < alpha, variances are significantly different, so set equal_var=False (Welch's t-test)
    equal_var_assumption = p_value_levene >= alpha
    print(f"Assuming equal variances for t-test: {equal_var_assumption} (based on Levene's p-value >= alpha)")

    # Perform the independent samples t-test
    t_statistic, p_value_ttest = ttest_ind(survivor_ages, non_survivor_ages, equal_var=equal_var_assumption)

    # Calculate Cohen's d (Effect Size for T-test)
    # Cohen's d measures the standardized difference between two means.
    # Formula: (mean1 - mean2) / pooled_standard_deviation
    n1, n2 = len(survivor_ages), len(non_survivor_ages)
    s1, s2 = survivor_ages.std(), non_survivor_ages.std()

    # Calculate pooled standard deviation
    if (n1 + n2 - 2) > 0: # Avoid division by zero if groups are too small
        pooled_std = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
        cohens_d = (survivor_ages.mean() - non_survivor_ages.mean()) / pooled_std
    else:
        cohens_d = np.nan # Cannot calculate if not enough data

    print(f"\nT-statistic: {t_statistic:.3f}")
    print(f"P-value: {p_value_ttest:.3f}")
    print(f"Cohen's d (Effect Size): {cohens_d:.3f}")

    # Interpret the result
    if p_value_ttest < alpha:
        print(f"Conclusion: Since p-value ({p_value_ttest:.3f}) < alpha ({alpha}), we reject the null hypothesis.")
        print("There is a statistically significant difference in the mean ages between survivors and non-survivors.")
    else:
        print(f"Conclusion: Since p-value ({p_value_ttest:.3f}) >= alpha ({alpha}), we fail to reject the null hypothesis.")
        print("There is no statistically significant difference in the mean ages between survivors and non-survivors.")
print("-" * 70 + "\n")

# --- 4. ANOVA: Fare prices across the three passenger classes ---
# Hypothesis:
# H0: The mean fare prices are equal across all three passenger classes (Pclass 1, 2, and 3).
# Ha: At least one mean fare price is different across the passenger classes.
print("--- One-Way ANOVA: Fare Prices across Passenger Classes (Pclass) ---")

# Drop rows where 'Fare' might be missing, though it's usually complete
df_fare_cleaned = df.dropna(subset=['Fare'])

# Separate fare prices into three groups based on Pclass
fare_pclass1 = df_fare_cleaned[df_fare_cleaned['Pclass'] == 1]['Fare']
fare_pclass2 = df_fare_cleaned[df_fare_cleaned['Pclass'] == 2]['Fare']
fare_pclass3 = df_fare_cleaned[df_fare_cleaned['Pclass'] == 3]['Fare']

# Check if any group is empty
if fare_pclass1.empty or fare_pclass2.empty or fare_pclass3.empty:
    print("Error: One or more Pclass fare groups are empty. Cannot perform ANOVA.")
else:
    print(f"Mean Fare Pclass 1: {fare_pclass1.mean():.2f} (N={len(fare_pclass1)})")
    print(f"Mean Fare Pclass 2: {fare_pclass2.mean():.2f} (N={len(fare_pclass2)})")
    print(f"Mean Fare Pclass 3: {fare_pclass3.mean():.2f} (N={len(fare_pclass3)})")

    # Perform Levene's test for equality of variances (ANOVA assumption check)
    # If p-value < alpha, the assumption of homogeneity of variances is violated.
    stat_levene_fare, p_value_levene_fare = stats.levene(fare_pclass1, fare_pclass2, fare_pclass3)
    print(f"\nLevene's Test for Equality of Variances (Fare): Statistic={stat_levene_fare:.3f}, P-value={p_value_levene_fare:.3f}")
    if p_value_levene_fare < alpha:
        print("Warning: Levene's test suggests unequal variances. ANOVA results might be less reliable or require robust methods.")

    # Perform One-Way ANOVA
    # Returns: F-statistic, p-value
    f_statistic, p_value_anova = f_oneway(fare_pclass1, fare_pclass2, fare_pclass3)

    # Calculate Eta-squared (Effect Size for ANOVA)
    # Eta-squared measures the proportion of variance in the dependent variable
    # that is explained by the independent variable.
    # Formula: SS_between / SS_total
    all_fares = pd.concat([fare_pclass1, fare_pclass2, fare_pclass3])
    grand_mean_fare = all_fares.mean()

    # Calculate Sum of Squares Total (SST)
    ss_total = np.sum((all_fares - grand_mean_fare)**2)

    # Calculate Sum of Squares Between (SSB)
    ss_between = (len(fare_pclass1) * (fare_pclass1.mean() - grand_mean_fare)**2 +
                  len(fare_pclass2) * (fare_pclass2.mean() - grand_mean_fare)**2 +
                  len(fare_pclass3) * (fare_pclass3.mean() - grand_mean_fare)**2)

    # Calculate Eta-squared
    eta_squared = ss_between / ss_total if ss_total > 0 else 0

    print(f"\nF-statistic: {f_statistic:.3f}")
    print(f"P-value: {p_value_anova:.3f}")
    print(f"Eta-squared (Effect Size): {eta_squared:.3f}")

    # Interpret the result
    if p_value_anova < alpha:
        print(f"Conclusion: Since p-value ({p_value_anova:.3f}) < alpha ({alpha}), we reject the null hypothesis.")
        print("There are statistically significant differences in mean fare prices across the three passenger classes.")
        print("Note: Further post-hoc tests (e.g., Tukey's HSD) would be needed to identify which specific class means differ.")
    else:
        print(f"Conclusion: Since p-value ({p_value_anova:.3f}) >= alpha ({alpha}), we fail to reject the null hypothesis.")
        print("There are no statistically significant differences in mean fare prices across the three passenger classes.")
print("-" * 70 + "\n")

print("--- Statistical Analysis Complete ---")
```