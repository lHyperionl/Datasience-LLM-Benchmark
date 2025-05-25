```python
import pandas as pd
import numpy as np
from scipy import stats

# Define significance level for all tests
alpha = 0.05

# --- 1. Load the dataset and handle any missing values ---
try:
    df = pd.read_csv('world-happiness-report-2021.csv')
except FileNotFoundError:
    print("Error: 'world-happiness-report-2021.csv' not found. Please ensure the file is in the current directory.")
    exit()

# Identify columns critical for the analysis
critical_columns = [
    'Ladder score',
    'Regional indicator',
    'Logged GDP per capita',
    'Social support',
    'Freedom to make life choices',
    'Generosity'
]

# Handle missing values by dropping rows where any critical column has a NaN.
# This ensures all subsequent statistical tests have complete data.
initial_rows = df.shape[0]
df.dropna(subset=critical_columns, inplace=True)
rows_after_na = df.shape[0]

print(f"Data Loading and Preprocessing:")
print(f"Initial rows: {initial_rows}")
print(f"Rows after dropping NaNs in critical columns: {rows_after_na} (Dropped {initial_rows - rows_after_na} rows)")
print(f"Significance level (alpha): {alpha}\n")

# --- 2. Test whether Western European countries have significantly higher happiness scores compared to all other regions combined (use t-test) ---
print("--- Test 1: Western Europe vs. Other Regions Happiness (Independent Samples t-test) ---")

# Separate happiness scores for Western Europe and all other regions
western_europe_happiness = df[df['Regional indicator'] == 'Western Europe']['Ladder score']
other_regions_happiness = df[df['Regional indicator'] != 'Western Europe']['Ladder score']

# Perform independent samples t-test. Using equal_var=False for Welch's t-test,
# which is more robust when variances are unequal, a common scenario in real-world data.
t_stat_we, p_val_we = stats.ttest_ind(western_europe_happiness, other_regions_happiness, equal_var=False)

print(f"Mean Happiness (Western Europe): {western_europe_happiness.mean():.3f}")
print(f"Mean Happiness (Other Regions): {other_regions_happiness.mean():.3f}")
print(f"T-statistic: {t_stat_we:.3f}")
print(f"P-value: {p_val_we:.3f}")

if p_val_we < alpha:
    print(f"Result: Reject the null hypothesis. (p < {alpha})")
    if western_europe_happiness.mean() > other_regions_happiness.mean():
        print("Interpretation: Western European countries have significantly HIGHER happiness scores compared to other regions.")
    else:
        print("Interpretation: Western European countries have significantly LOWER happiness scores compared to other regions.")
else:
    print(f"Result: Fail to reject the null hypothesis. (p >= {alpha})")
    print("Interpretation: There is no statistically significant difference in happiness scores between Western European countries and other regions.")

print("-" * 80 + "\n")

# --- 3. Test if there's a significant difference in GDP per capita between the top 25% happiest countries and the bottom 25% (use Mann-Whitney U test) ---
print("--- Test 2: Top 25% vs. Bottom 25% Happiest GDP per Capita (Mann-Whitney U test) ---")

# Sort the DataFrame by 'Ladder score' to identify happiest and least happy countries
df_sorted_by_happiness = df.sort_values(by='Ladder score').reset_index(drop=True)

# Determine the number of countries for top/bottom 25%
n_countries = len(df_sorted_by_happiness)
num_25_percent = int(0.25 * n_countries)

# Extract 'Logged GDP per capita' for the top 25% happiest countries
top_25_gdp = df_sorted_by_happiness.tail(num_25_percent)['Logged GDP per capita']
# Extract 'Logged GDP per capita' for the bottom 25% happiest countries
bottom_25_gdp = df_sorted_by_happiness.head(num_25_percent)['Logged GDP per capita']

# Perform Mann-Whitney U test (non-parametric test, suitable for non-normally distributed data or ordinal data)
u_stat_gdp, p_val_gdp = stats.mannwhitneyu(top_25_gdp, bottom_25_gdp, alternative='two-sided')

print(f"Median GDP per Capita (Top 25% Happiest): {top_25_gdp.median():.3f}")
print(f"Median GDP per Capita (Bottom 25% Happiest): {bottom_25_gdp.median():.3f}")
print(f"U-statistic: {u_stat_gdp:.3f}")
print(f"P-value: {p_val_gdp:.3f}")

if p_val_gdp < alpha:
    print(f"Result: Reject the null hypothesis. (p < {alpha})")
    if top_25_gdp.median() > bottom_25_gdp.median():
        print("Interpretation: The top 25% happiest countries have significantly HIGHER GDP per capita compared to the bottom 25%.")
    else:
        print("Interpretation: The top 25% happiest countries have significantly LOWER GDP per capita compared to the bottom 25%.")
else:
    print(f"Result: Fail to reject the null hypothesis. (p >= {alpha})")
    print("Interpretation: There is no statistically significant difference in GDP per capita between the top 25% happiest and bottom 25% happiest countries.")

print("-" * 80 + "\n")

# --- 4. Perform ANOVA to test if happiness scores differ significantly across all regional indicators ---
print("--- Test 3: Regional Happiness Scores (One-Way ANOVA) ---")

# Prepare data for ANOVA: create a list of happiness scores for each unique region
regional_groups = [df['Ladder score'][df['Regional indicator'] == region] for region in df['Regional indicator'].unique()]

# Perform one-way ANOVA
f_stat_anova, p_val_anova = stats.f_oneway(*regional_groups)

print(f"F-statistic: {f_stat_anova:.3f}")
print(f"P-value: {p_val_anova:.3f}")

if p_val_anova < alpha:
    print(f"Result: Reject the null hypothesis. (p < {alpha})")
    print("Interpretation: There is a statistically significant difference in happiness scores across regional indicators. At least one regional indicator has a significantly different mean happiness score compared to others.")
    print("Note: Post-hoc tests (e.g., Tukey HSD) would be needed to identify which specific regional pairs differ.")
else:
    print(f"Result: Fail to reject the null hypothesis. (p >= {alpha})")
    print("Interpretation: There is no statistically significant difference in happiness scores across regional indicators.")

print("-" * 80 + "\n")

# --- 5. Test the correlation between GDP per capita and Social support with statistical significance ---
print("--- Test 4: Correlation between GDP per Capita and Social Support (Pearson Correlation) ---")

gdp_per_capita = df['Logged GDP per capita']
social_support = df['Social support']

# Perform Pearson correlation test
corr_coeff, p_val_corr = stats.pearsonr(gdp_per_capita, social_support)

print(f"Pearson Correlation Coefficient: {corr_coeff:.3f}")
print(f"P-value: {p_val_corr:.3f}")

if p_val_corr < alpha:
    print(f"Result: Reject the null hypothesis. (p < {alpha})")
    if corr_coeff > 0:
        print("Interpretation: There is a statistically significant POSITIVE correlation between GDP per capita and Social support.")
    else:
        print("Interpretation: There is a statistically significant NEGATIVE correlation between GDP per capita and Social support.")
else:
    print(f"Result: Fail to reject the null hypothesis. (p >= {alpha})")
    print("Interpretation: There is no statistically significant correlation between GDP per capita and Social support.")

print("-" * 80 + "\n")

# --- 6. Perform a chi-square test to determine if happiness categories (High/Medium/Low) are associated with regional indicators ---
print("--- Test 5: Happiness Categories vs. Regional Indicators (Chi-square Test of Independence) ---")

# Create happiness categories based on quantiles to ensure roughly equal group sizes
low_threshold = df['Ladder score'].quantile(1/3)
high_threshold = df['Ladder score'].quantile(2/3)

def categorize_happiness(score):
    if score <= low_threshold:
        return 'Low'
    elif score <= high_threshold:
        return 'Medium'
    else:
        return 'High'

df['Happiness Category'] = df['Ladder score'].apply(categorize_happiness)

# Create a contingency table (cross-tabulation) of the two categorical variables
contingency_table = pd.crosstab(df['Happiness Category'], df['Regional indicator'])

# Perform Chi-square test of independence
chi2_stat_chi, p_val_chi, dof_chi, expected_freq_chi = stats.chi2_contingency(contingency_table)

print("Contingency Table:")
print(contingency_table)
print(f"\nChi-square Statistic: {chi2_stat_chi:.3f}")
print(f"P-value: {p_val_chi:.3f}")
print(f"Degrees of Freedom: {dof_chi}")

if p_val_chi < alpha:
    print(f"Result: Reject the null hypothesis. (p < {alpha})")
    print("Interpretation: There is a statistically significant association between happiness categories and regional indicators. Happiness levels are not independent of the regional indicator.")
else:
    print(f"Result: Fail to reject the null hypothesis. (p >= {alpha})")
    print("Interpretation: There is no statistically significant association between happiness categories and regional indicators. Happiness levels are independent of the regional indicator.")

print("-" * 80 + "\n")

# --- 7. Test if countries with above-median freedom scores have significantly different generosity levels (use t-test) ---
print("--- Test 6: Above-Median Freedom vs. Generosity (Independent Samples t-test) ---")

# Calculate the median freedom score
median_freedom = df['Freedom to make life choices'].median()

# Create two groups based on whether freedom score is above or below the median
above_median_freedom_generosity = df[df['Freedom to make life choices'] >= median_freedom]['Generosity']
below_median_freedom_generosity = df[df['Freedom to make life choices'] < median_freedom]['Generosity']

# Perform independent samples t-test (Welch's t-test)
t_stat_gen, p_val_gen = stats.ttest_ind(above_median_freedom_generosity, below_median_freedom_generosity, equal_var=False)

print(f"Mean Generosity (Above Median Freedom): {above_median_freedom_generosity.mean():.3f}")
print(f"Mean Generosity (Below Median Freedom): {below_median_freedom_generosity.mean():.3f}")
print(f"T-statistic: {t_stat_gen:.3f}")
print(f"P-value: {p_val_gen:.3f}")

if p_val_gen < alpha:
    print(f"Result: Reject the null hypothesis. (p < {alpha})")
    if above_median_freedom_generosity.mean() > below_median_freedom_generosity.mean():
        print("Interpretation: Countries with above-median freedom scores have significantly HIGHER generosity levels.")
    else:
        print("Interpretation: Countries with above-median freedom scores have significantly LOWER generosity levels.")
else:
    print(f"Result: Fail to reject the null hypothesis. (p >= {alpha})")
    print("Interpretation: There is no statistically significant difference in generosity levels between countries with above-median freedom scores and those with below-median freedom scores.")

print("-" * 80 + "\n")
```