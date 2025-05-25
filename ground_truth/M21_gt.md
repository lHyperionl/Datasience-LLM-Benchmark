```python
import pandas as pd
from scipy import stats
import numpy as np

# --- Configuration ---
ALPHA = 0.05  # Significance level
DATA_PATH = 'synthetic_covid19_data.csv'

# --- 1. Load Data and Handle Missing Values ---

# Load the dataset
try:
    df = pd.read_csv(DATA_PATH)
    print(f"Dataset '{DATA_PATH}' loaded successfully.")
except FileNotFoundError:
    print(f"Error: Dataset '{DATA_PATH}' not found. Please ensure the file is in the correct directory.")
    exit() # Exit if the dataset is not found

# Columns to be imputed with their median
impute_cols = [
    'reproduction_rate',
    'stringency_index',
    'median_age',
    'total_deaths',
    'population',
    'new_cases'
]

# Perform median imputation for specified columns
for col in impute_cols:
    if col in df.columns:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            # print(f"Imputed missing values in '{col}' with median: {median_val:.2f}")
    else:
        print(f"Warning: Column '{col}' not found in the dataset. Skipping imputation for this column.")

# Convert 'date' column to datetime objects for proper sorting
df['date'] = pd.to_datetime(df['date'])

print("\n--- Data Loading and Imputation Complete ---")
print(f"Missing values after imputation for relevant columns:\n{df[impute_cols].isnull().sum()}")

# --- 2. ANOVA Test: Mean 'reproduction_rate' across Continents ---

print("\n--- ANOVA Test: Mean 'reproduction_rate' across Continents (Asia, Europe, North America) ---")
print("Null Hypothesis (H0): The mean 'reproduction_rate' is the same across Asia, Europe, and North America.")

# Define the continents of interest
continents_of_interest = ['Asia', 'Europe', 'North America']

# Filter the DataFrame to include only data from the specified continents
df_anova = df[df['continent'].isin(continents_of_interest)].copy()

# Prepare lists of 'reproduction_rate' for each continent
reproduction_rates_by_continent = []
for continent in continents_of_interest:
    # Extract non-null reproduction rates for the current continent
    rates = df_anova[df_anova['continent'] == continent]['reproduction_rate'].dropna()
    if not rates.empty:
        reproduction_rates_by_continent.append(rates)
    else:
        print(f"Warning: No valid 'reproduction_rate' data found for {continent}. This continent will be excluded from ANOVA.")

# Perform ANOVA if there is data for at least two groups
if len(reproduction_rates_by_continent) >= 2:
    f_stat, p_value_anova = stats.f_oneway(*reproduction_rates_by_continent)
    print(f"ANOVA p-value: {p_value_anova:.4f}")

    # Conclude based on the p-value and significance level (alpha)
    if p_value_anova < ALPHA:
        print(f"Conclusion: Reject the null hypothesis. There is a significant difference in mean 'reproduction_rate' across the selected continents (p < {ALPHA}).")
    else:
        print(f"Conclusion: Fail to reject the null hypothesis. There is no significant difference in mean 'reproduction_rate' across the selected continents (p >= {ALPHA}).")
else:
    print("ANOVA could not be performed due to insufficient data (less than two groups with data).")


# --- 3. Pearson Correlation Test: 'stringency_index' and 'new_cases' ---

print("\n--- Pearson Correlation Test: 'stringency_index' vs 'new_cases' ---")
print("Null Hypothesis (H0): There is no significant linear correlation between 'stringency_index' and 'new_cases'.")

# Select the columns for correlation and drop any remaining NaNs (though imputation should have handled most)
df_corr = df[['stringency_index', 'new_cases']].dropna()

# Ensure there's enough data to perform correlation
if len(df_corr) > 1:
    correlation, p_value_pearson = stats.pearsonr(df_corr['stringency_index'], df_corr['new_cases'])
    print(f"Pearson correlation coefficient: {correlation:.4f}")
    print(f"Pearson correlation p-value: {p_value_pearson:.4f}")

    # Conclude based on the p-value and significance level (alpha)
    if p_value_pearson < ALPHA:
        print(f"Conclusion: Reject the null hypothesis. There is a significant linear correlation between 'stringency_index' and 'new_cases' (p < {ALPHA}).")
    else:
        print(f"Conclusion: Fail to reject the null hypothesis. There is no significant linear correlation between 'stringency_index' and 'new_cases' (p >= {ALPHA}).")
else:
    print("Pearson correlation could not be performed due to insufficient data (less than 2 data points).")


# --- 4. T-test: 'median_age' between 'high_death_rate_country' and others ---

print("\n--- T-test: 'median_age' between 'high_death_rate_country' and others ---")
print("Null Hypothesis (H0): The mean 'median_age' is the same for 'high_death_rate_country' and 'other' countries.")

# 4.1. Derive 'high_death_rate_country' category
# Sort by date and group by country to get the latest entry for each country
latest_country_data = df.sort_values(by=['country', 'date']).groupby('country').last().reset_index()

# Calculate the death rate ratio (total_deaths / population)
# Handle potential division by zero or NaN populations by replacing inf/-inf with NaN
latest_country_data['death_rate_ratio'] = latest_country_data['total_deaths'] / latest_country_data['population']
latest_country_data['death_rate_ratio'].replace([np.inf, -np.inf], np.nan, inplace=True)

# Drop rows where death_rate_ratio is NaN (e.g., if population was 0 or NaN)
latest_country_data.dropna(subset=['death_rate_ratio'], inplace=True)

# Calculate the 75th percentile of the death rate ratio across all countries
if not latest_country_data['death_rate_ratio'].empty:
    percentile_75 = latest_country_data['death_rate_ratio'].quantile(0.75)
    print(f"75th percentile of death rate ratio: {percentile_75:.6f}")

    # Create the binary category 'high_death_rate_country'
    latest_country_data['high_death_rate_country'] = latest_country_data['death_rate_ratio'] > percentile_75

    # Prepare data for t-test: 'median_age' for each group
    # Ensure 'median_age' is not NaN for the t-test (should be handled by initial imputation)
    latest_country_data.dropna(subset=['median_age'], inplace=True)

    group_high_death = latest_country_data[latest_country_data['high_death_rate_country']]['median_age']
    group_other_death = latest_country_data[~latest_country_data['high_death_rate_country']]['median_age']

    # Perform independent t-test (Welch's t-test, assuming unequal variances, which is robust)
    if not group_high_death.empty and not group_other_death.empty:
        t_stat, p_value_ttest = stats.ttest_ind(group_high_death, group_other_death, equal_var=False)
        print(f"T-test p-value: {p_value_ttest:.4f}")

        # Conclude based on the p-value and significance level (alpha)
        if p_value_ttest < ALPHA:
            print(f"Conclusion: Reject the null hypothesis. There is a significant difference in mean 'median_age' between 'high_death_rate_country' and 'other' countries (p < {ALPHA}).")
        else:
            print(f"Conclusion: Fail to reject the null hypothesis. There is no significant difference in mean 'median_age' between 'high_death_rate_country' and 'other' countries (p >= {ALPHA}).")
    else:
        print("T-test could not be performed due to insufficient data in one or both groups:")
        if group_high_death.empty:
            print("  - No countries classified as 'high_death_rate_country' with valid 'median_age'.")
        if group_other_death.empty:
            print("  - No countries classified as 'other' with valid 'median_age'.")
else:
    print("T-test could not be performed: Not enough data to calculate 75th percentile of death rate ratio.")
```