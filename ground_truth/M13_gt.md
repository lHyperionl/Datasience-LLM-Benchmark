```python
import pandas as pd
import numpy as np
from scipy import stats

# Define significance level
ALPHA = 0.05

# --- 1. Load the dataset and clean zero values ---
try:
    df = pd.read_csv('diabetes.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: 'diabetes.csv' not found. Please ensure the file is in the current directory.")
    exit()

# Columns where 0 values are biologically implausible and should be treated as NaN
# Pregnancies and Outcome can legitimately be 0.
cols_to_impute = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

# Replace 0s with NaN in specified columns
for col in cols_to_impute:
    df[col] = df[col].replace(0, np.nan)

# Perform median imputation for NaN values
for col in cols_to_impute:
    if df[col].isnull().any():
        median_val = df[col].median()
        df[col].fillna(median_val, inplace=True)
        print(f"Imputed missing values in '{col}' with median: {median_val}")

print("\nData cleaning and imputation complete.")
print("First 5 rows after cleaning:")
print(df.head())
print("\nDescriptive statistics after cleaning:")
print(df.describe())

# --- 2. Test whether diabetic patients have significantly higher glucose levels compared to non-diabetic patients (t-test) ---
print("\n--- Hypothesis Test 1: Glucose Levels (Diabetic vs. Non-Diabetic) ---")
# Separate glucose levels for diabetic (Outcome=1) and non-diabetic (Outcome=0) patients
diabetic_glucose = df[df['Outcome'] == 1]['Glucose']
nondiabetic_glucose = df[df['Outcome'] == 0]['Glucose']

# Perform independent samples t-test
# Using equal_var=False (Welch's t-test) as it does not assume equal population variances,
# which is often a safer choice when assumptions are not explicitly checked.
t_stat, p_value_glucose_ttest = stats.ttest_ind(diabetic_glucose, nondiabetic_glucose, equal_var=False)

print(f"T-test for Glucose Levels:")
print(f"  Diabetic Glucose Mean: {diabetic_glucose.mean():.2f}")
print(f"  Non-Diabetic Glucose Mean: {nondiabetic_glucose.mean():.2f}")
print(f"  T-statistic: {t_stat:.3f}")
print(f"  P-value: {p_value_glucose_ttest:.4f}")

if p_value_glucose_ttest < ALPHA:
    print(f"  Interpretation: Since p-value ({p_value_glucose_ttest:.4f}) < α ({ALPHA}), we reject the null hypothesis.")
    print("  Conclusion: There is a statistically significant difference in glucose levels between diabetic and non-diabetic patients.")
    print("  Specifically, diabetic patients have significantly higher glucose levels.")
else:
    print(f"  Interpretation: Since p-value ({p_value_glucose_ttest:.4f}) >= α ({ALPHA}), we fail to reject the null hypothesis.")
    print("  Conclusion: There is no statistically significant difference in glucose levels between diabetic and non-diabetic patients.")

# --- 3. Test if there's a significant difference in BMI between diabetic and non-diabetic groups (Mann-Whitney U test) ---
print("\n--- Hypothesis Test 2: BMI (Diabetic vs. Non-Diabetic) ---")
# Separate BMI for diabetic (Outcome=1) and non-diabetic (Outcome=0) patients
diabetic_bmi = df[df['Outcome'] == 1]['BMI']
nondiabetic_bmi = df[df['Outcome'] == 0]['BMI']

# Perform Mann-Whitney U test (non-parametric test for comparing two independent groups)
u_stat, p_value_bmi_mw = stats.mannwhitneyu(diabetic_bmi, nondiabetic_bmi)

print(f"Mann-Whitney U test for BMI:")
print(f"  Diabetic BMI Median: {diabetic_bmi.median():.2f}")
print(f"  Non-Diabetic BMI Median: {nondiabetic_bmi.median():.2f}")
print(f"  U-statistic: {u_stat:.3f}")
print(f"  P-value: {p_value_bmi_mw:.4f}")

if p_value_bmi_mw < ALPHA:
    print(f"  Interpretation: Since p-value ({p_value_bmi_mw:.4f}) < α ({ALPHA}), we reject the null hypothesis.")
    print("  Conclusion: There is a statistically significant difference in BMI between diabetic and non-diabetic groups.")
else:
    print(f"  Interpretation: Since p-value ({p_value_bmi_mw:.4f}) >= α ({ALPHA}), we fail to reject the null hypothesis.")
    print("  Conclusion: There is no statistically significant difference in BMI between diabetic and non-diabetic groups.")

# --- 4. Perform a chi-square test to determine if age groups are associated with diabetes outcome ---
print("\n--- Hypothesis Test 3: Age Groups and Diabetes Outcome (Chi-square test) ---")
# Create age groups
# Young: <30, Middle: 30-50, Older: >50
df['AgeGroup'] = pd.cut(df['Age'],
                        bins=[0, 29, 50, np.inf],
                        labels=['Young', 'Middle', 'Older'],
                        right=True) # right=True means (a, b]

# Create a contingency table
contingency_table_age_outcome = pd.crosstab(df['AgeGroup'], df['Outcome'])
print("Contingency Table (AgeGroup vs. Outcome):")
print(contingency_table_age_outcome)

# Perform Chi-square test of independence
chi2_stat, p_value_chi2, dof, expected_freq = stats.chi2_contingency(contingency_table_age_outcome)

print(f"Chi-square test for Age Group vs. Outcome:")
print(f"  Chi-square statistic: {chi2_stat:.3f}")
print(f"  P-value: {p_value_chi2:.4f}")
print(f"  Degrees of Freedom: {dof}")

if p_value_chi2 < ALPHA:
    print(f"  Interpretation: Since p-value ({p_value_chi2:.4f}) < α ({ALPHA}), we reject the null hypothesis.")
    print("  Conclusion: There is a statistically significant association between age groups and diabetes outcome.")
else:
    print(f"  Interpretation: Since p-value ({p_value_chi2:.4f}) >= α ({ALPHA}), we fail to reject the null hypothesis.")
    print("  Conclusion: There is no statistically significant association between age groups and diabetes outcome.")

# --- 5. Test correlation between pregnancies and age using Pearson correlation coefficient with significance testing ---
print("\n--- Hypothesis Test 4: Correlation between Pregnancies and Age (Pearson Correlation) ---")
# Perform Pearson correlation test
corr_coeff_pearson, p_value_pearson = stats.pearsonr(df['Pregnancies'], df['Age'])

print(f"Pearson Correlation between Pregnancies and Age:")
print(f"  Correlation Coefficient: {corr_coeff_pearson:.3f}")
print(f"  P-value: {p_value_pearson:.4f}")

if p_value_pearson < ALPHA:
    print(f"  Interpretation: Since p-value ({p_value_pearson:.4f}) < α ({ALPHA}), we reject the null hypothesis.")
    print(f"  Conclusion: There is a statistically significant linear correlation between pregnancies and age.")
    if corr_coeff_pearson > 0:
        print("  The correlation is positive, suggesting that older patients tend to have more pregnancies.")
    else:
        print("  The correlation is negative, suggesting that older patients tend to have fewer pregnancies.")
else:
    print(f"  Interpretation: Since p-value ({p_value_pearson:.4f}) >= α ({ALPHA}), we fail to reject the null hypothesis.")
    print("  Conclusion: There is no statistically significant linear correlation between pregnancies and age.")

# --- 6. Perform ANOVA to test if glucose levels differ significantly across BMI categories ---
print("\n--- Hypothesis Test 5: Glucose Levels across BMI Categories (ANOVA) ---")
# Create BMI categories
# Underweight: <18.5, Normal: 18.5-24.9, Overweight: 25-29.9, Obese: >=30
df['BMICategory'] = pd.cut(df['BMI'],
                           bins=[0, 18.5, 25, 30, np.inf],
                           labels=['Underweight', 'Normal', 'Overweight', 'Obese'],
                           right=False) # right=False means [a, b)

# Get glucose levels for each BMI category
glucose_underweight = df[df['BMICategory'] == 'Underweight']['Glucose']
glucose_normal = df[df['BMICategory'] == 'Normal']['Glucose']
glucose_overweight = df[df['BMICategory'] == 'Overweight']['Glucose']
glucose_obese = df[df['BMICategory'] == 'Obese']['Glucose']

# Filter out empty categories if any (though unlikely with this dataset)
glucose_groups = [group for group in [glucose_underweight, glucose_normal, glucose_overweight, glucose_obese] if not group.empty]
group_labels = [label for label, group in zip(['Underweight', 'Normal', 'Overweight', 'Obese'], [glucose_underweight, glucose_normal, glucose_overweight, glucose_obese]) if not group.empty]

if len(glucose_groups) < 2:
    print("  Not enough BMI categories with data to perform ANOVA.")
else:
    # Perform one-way ANOVA
    f_stat_anova, p_value_anova = stats.f_oneway(*glucose_groups)

    print(f"ANOVA for Glucose Levels across BMI Categories:")
    for label, group in zip(group_labels, glucose_groups):
        print(f"  {label} Glucose Mean: {group.mean():.2f}")
    print(f"  F-statistic: {f_stat_anova:.3f}")
    print(f"  P-value: {p_value_anova:.4f}")

    if p_value_anova < ALPHA:
        print(f"  Interpretation: Since p-value ({p_value_anova:.4f}) < α ({ALPHA}), we reject the null hypothesis.")
        print("  Conclusion: There is a statistically significant difference in glucose levels across different BMI categories.")
    else:
        print(f"  Interpretation: Since p-value ({p_value_anova:.4f}) >= α ({ALPHA}), we fail to reject the null hypothesis.")
        print("  Conclusion: There is no statistically significant difference in glucose levels across different BMI categories.")

print("\n--- Statistical Analysis Complete ---")
```