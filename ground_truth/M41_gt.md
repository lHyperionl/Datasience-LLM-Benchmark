```python
import pandas as pd
from scipy import stats

# --- 1. Load the dataset ---
try:
    df = pd.read_csv('imdb_top_250.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: 'imdb_top_250.csv' not found. Please ensure the file is in the current directory.")
    exit()

# Ensure 'Year' and 'Rating' columns are in appropriate numeric formats
# 'Year' might be an object if it contains non-numeric characters or is read as such.
# 'Rating' should be float.
df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce')

# Drop rows where 'Year' or 'Rating' became NaN due to conversion errors
df.dropna(subset=['Year', 'Rating'], inplace=True)

# --- 2. Create two groups of movies based on the `Year` column ---
# Group 1: Movies released before 1990
group_before_1990 = df[df['Year'] < 1990]

# Group 2: Movies released in or after 1990
group_after_1990 = df[df['Year'] >= 1990]

# Extract the 'Rating' values for each group
ratings_before_1990 = group_before_1990['Rating']
ratings_after_1990 = group_after_1990['Rating']

# Check if groups have enough data for the test
if len(ratings_before_1990) < 2 or len(ratings_after_1990) < 2:
    print("Error: One or both groups have insufficient data (less than 2 samples) to perform a t-test.")
    print(f"Movies before 1990: {len(ratings_before_1990)} samples")
    print(f"Movies after 1990: {len(ratings_after_1990)} samples")
    exit()

# --- 3. Perform an independent two-sample t-test ---

# --- 4. Clearly state the null and alternative hypotheses for this test ---
print("\n--- Hypotheses ---")
print("Null Hypothesis (H0): There is no statistically significant difference in the mean 'Rating' between movies released before 1990 and movies released in or after 1990.")
print("Alternative Hypothesis (H1): There is a statistically significant difference in the mean 'Rating' between movies released before 1990 and movies released in or after 1990.")

# Perform the independent two-sample t-test
# `equal_var=True` assumes equal population variances. If this assumption is violated,
# `equal_var=False` (Welch's t-test) would be more appropriate.
t_statistic, p_value = stats.ttest_ind(ratings_before_1990, ratings_after_1990, equal_var=True)

# --- 5. Print the calculated t-statistic and the p-value ---
print("\n--- T-Test Results ---")
print(f"T-statistic: {t_statistic:.4f}")
print(f"P-value: {p_value:.4f}")

# --- 6. Based on a significance level of alpha = 0.05, interpret the p-value ---
alpha = 0.05
print(f"Significance Level (alpha): {alpha}")

print("\n--- Interpretation ---")
if p_value < alpha:
    print(f"Since the p-value ({p_value:.4f}) is less than the significance level ({alpha}), we reject the null hypothesis.")
    print("Conclusion: There is a statistically significant difference in the mean 'Rating' between movies released before 1990 and movies released in or after 1990.")
else:
    print(f"Since the p-value ({p_value:.4f}) is greater than or equal to the significance level ({alpha}), we fail to reject the null hypothesis.")
    print("Conclusion: There is no statistically significant difference in the mean 'Rating' between movies released before 1990 and movies released in or after 1990.")

# Optional: Print mean ratings for context
print(f"\nMean Rating (Movies before 1990): {ratings_before_1990.mean():.2f}")
print(f"Mean Rating (Movies after 1990): {ratings_after_1990.mean():.2f}")
```