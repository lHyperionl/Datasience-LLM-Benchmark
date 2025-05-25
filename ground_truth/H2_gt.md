```python
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests

# --- Configuration ---
FILE_PATH = 'winequality-red.csv'
ALPHA = 0.05  # Significance level for all statistical tests

# --- 1. Load Data and Define Quality Groups ---
try:
    df = pd.read_csv(FILE_PATH)
except FileNotFoundError:
    print(f"Error: The file '{FILE_PATH}' was not found. Please ensure it's in the current directory.")
    exit()

# Define quality tiers based on the 'quality' score
def assign_quality_tier(score):
    if score <= 4:
        return 'low'
    elif 5 <= score <= 6:
        return 'medium'
    else:  # score >= 7
        return 'high'

df['quality_tier'] = df['quality'].apply(assign_quality_tier)

# Ensure 'quality_tier' is a categorical type with a specific order for consistent plotting
quality_order = ['low', 'medium', 'high']
df['quality_tier'] = pd.Categorical(df['quality_tier'], categories=quality_order, ordered=True)

# Identify chemical properties (all columns except 'quality' and the newly created 'quality_tier')
chemical_properties = [col for col in df.columns if col not in ['quality', 'quality_tier']]

# Prepare dictionaries to store test results and significant features
statistical_results = {}
significant_features = []

print("--- Performing Statistical Tests ---")

# --- 2. Perform Statistical Tests (ANOVA or Kruskal-Wallis) ---
for feature in chemical_properties:
    print(f"\nAnalyzing feature: {feature}")
    
    # Extract data for each quality tier, dropping NaN values
    low_quality_data = df[df['quality_tier'] == 'low'][feature].dropna()
    medium_quality_data = df[df['quality_tier'] == 'medium'][feature].dropna()
    high_quality_data = df[df['quality_tier'] == 'high'][feature].dropna()

    # Skip analysis if any group has insufficient data (e.g., less than 3 samples for Shapiro-Wilk)
    # Shapiro-Wilk requires at least 3 data points.
    if len(low_quality_data) < 3 or len(medium_quality_data) < 3 or len(high_quality_data) < 3:
        print(f"  Skipping {feature}: Not enough data in one or more quality tiers for statistical testing.")
        continue

    # Check for normality using Shapiro-Wilk test for each group
    # Null hypothesis (H0): The data is drawn from a normal distribution.
    # If p-value < ALPHA, we reject H0, meaning the data is likely not normal.
    shapiro_low_p = stats.shapiro(low_quality_data).pvalue
    shapiro_medium_p = stats.shapiro(medium_quality_data).pvalue
    shapiro_high_p = stats.shapiro(high_quality_data).pvalue

    is_normal_low = shapiro_low_p > ALPHA
    is_normal_medium = shapiro_medium_p > ALPHA
    is_normal_high = shapiro_high_p > ALPHA

    all_groups_normal = is_normal_low and is_normal_medium and is_normal_high
    
    print(f"  Normality (Shapiro-Wilk p-values): Low={shapiro_low_p:.3f}, Medium={shapiro_medium_p:.3f}, High={shapiro_high_p:.3f}")

    # Check for homogeneity of variances using Levene's test (only if all groups are considered normal)
    # Null hypothesis (H0): All input samples are from populations with equal variances.
    # If p-value < ALPHA, we reject H0, meaning variances are not equal.
    levene_p = np.nan # Initialize
    has_equal_variance = False
    if all_groups_normal:
        levene_p = stats.levene(low_quality_data, medium_quality_data, high_quality_data).pvalue
        has_equal_variance = levene_p > ALPHA
        print(f"  Homogeneity of Variance (Levene's p-value): {levene_p:.3f}")

    test_type = ""
    p_value = np.nan

    # Decide which statistical test to use based on normality and homogeneity of variance
    if all_groups_normal and has_equal_variance:
        # Use ANOVA (Parametric test) if assumptions are met
        test_type = "ANOVA"
        f_stat, p_value = stats.f_oneway(low_quality_data, medium_quality_data, high_quality_data)
        print(f"  Using ANOVA: F-statistic={f_stat:.3f}, p-value={p_value:.3f}")
    else:
        # Use Kruskal-Wallis (Non-parametric test) if assumptions for ANOVA are not met
        test_type = "Kruskal-Wallis"
        h_stat, p_value = stats.kruskal(low_quality_data, medium_quality_data, high_quality_data)
        print(f"  Using Kruskal-Wallis: H-statistic={h_stat:.3f}, p-value={p_value:.3f}")

    # Store the results of the primary test
    statistical_results[feature] = {
        'test_type': test_type,
        'p_value': p_value,
        'significant': p_value < ALPHA
    }

    if p_value < ALPHA:
        significant_features.append(feature)
        print(f"  Significant difference found for {feature} (p < {ALPHA})")
    else:
        print(f"  No significant difference found for {feature} (p >= {ALPHA})")

print("\n--- Performing Post-hoc Tests for Significant Features ---")

# --- 3. Perform Post-hoc Tests ---
posthoc_results = {}

for feature in significant_features:
    print(f"\nPost-hoc analysis for: {feature}")
    test_type = statistical_results[feature]['test_type']

    # Re-extract data for post-hoc tests to ensure consistency
    low_quality_data = df[df['quality_tier'] == 'low'][feature].dropna()
    medium_quality_data = df[df['quality_tier'] == 'medium'][feature].dropna()
    high_quality_data = df[df['quality_tier'] == 'high'][feature].dropna()

    # Check if any group has insufficient data for pairwise comparisons (e.g., less than 2 samples)
    if len(low_quality_data) < 2 or len(medium_quality_data) < 2 or len(high_quality_data) < 2:
        print(f"  Skipping post-hoc for {feature}: Not enough data in one or more quality tiers for pairwise comparison.")
        continue

    if test_type == "ANOVA":
        # Tukey's HSD for post-hoc analysis after ANOVA
        # Combine all data into a single series with corresponding group labels
        data_for_tukey = pd.concat([low_quality_data, medium_quality_data, high_quality_data])
        labels_for_tukey = ['low'] * len(low_quality_data) + \
                           ['medium'] * len(medium_quality_data) + \
                           ['high'] * len(high_quality_data)
        
        try:
            tukey_result = pairwise_tukeyhsd(endog=data_for_tukey, groups=labels_for_tukey, alpha=ALPHA)
            posthoc_results[feature] = {'test': 'Tukey HSD', 'results': str(tukey_result)}
            print(tukey_result)
        except ValueError as e:
            print(f"  Could not perform Tukey HSD for {feature}: {e}")
            posthoc_results[feature] = {'test': 'Tukey HSD', 'results': f"Error: {e}"}
    
    elif test_type == "Kruskal-Wallis":
        # Pairwise Mann-Whitney U tests with Bonferroni correction as a proxy for Dunn's test
        # This is used when Kruskal-Wallis was the primary test.
        groups_data = {
            'low': low_quality_data,
            'medium': medium_quality_data,
            'high': high_quality_data
        }
        
        comparisons = [('low', 'medium'), ('low', 'high'), ('medium', 'high')]
        p_values_mw = []
        comparison_labels = []

        print("  Performing pairwise Mann-Whitney U tests with Bonferroni correction:")
        for g1_name, g2_name in comparisons:
            # Ensure both groups have data before performing test
            if not groups_data[g1_name].empty and not groups_data[g2_name].empty:
                stat, p_val = stats.mannwhitneyu(groups_data[g1_name], groups_data[g2_name], alternative='two-sided')
                p_values_mw.append(p_val)
                comparison_labels.append(f"{g1_name} vs {g2_name}")
                print(f"    {g1_name} vs {g2_name}: U-statistic={stat:.3f}, raw p-value={p_val:.3f}")
            else:
                print(f"    Skipping {g1_name} vs {g2_name}: One or both groups are empty.")

        if p_values_mw: # Only apply correction if there were comparisons
            # Apply Bonferroni correction for multiple comparisons
            reject_bonf, p_values_corrected, _, _ = multipletests(p_values_mw, alpha=ALPHA, method='bonferroni')
            
            mw_posthoc_summary = []
            for i, (label, p_corr, reject) in enumerate(zip(comparison_labels, p_values_corrected, reject_bonf)):
                summary_line = f"    {label}: Corrected p-value={p_corr:.3f}, Significant={'Yes' if reject else 'No'}"
                mw_posthoc_summary.append(summary_line)
                print(summary_line)
            
            posthoc_results[feature] = {'test': 'Pairwise Mann-Whitney U (Bonferroni corrected)', 'results': "\n".join(mw_posthoc_summary)}
        else:
            print("  No valid pairwise comparisons to perform for this feature.")
            posthoc_results[feature] = {'test': 'Pairwise Mann-Whitney U (Bonferroni corrected)', 'results': "No valid comparisons."}

print("\n--- Creating Box Plots for Significant Features ---")

# --- 4. Create Box Plots ---
plt.style.use('seaborn-v0_8-darkgrid') # Apply a consistent and visually appealing style
fig_count = 0
for feature in significant_features:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='quality_tier', y=feature, data=df, palette='viridis', order=quality_order)
    plt.title(f'Distribution of {feature} across Wine Quality Tiers')
    plt.xlabel('Wine Quality Tier')
    plt.ylabel(feature)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()
    fig_count += 1

if fig_count == 0:
    print("No significant features found to plot box plots for.")

print("\n--- Summary of Features with Significant Differences ---")

# --- 5. Summarize Results ---
if not significant_features:
    print("No chemical properties showed a statistically significant difference across quality tiers at alpha = 0.05.")
else:
    print("The following chemical properties showed a statistically significant difference across quality tiers (alpha = 0.05):")
    for feature in chemical_properties:
        if feature in statistical_results and statistical_results[feature]['significant']:
            print(f"\nFeature: {feature}")
            print(f"  Primary Test: {statistical_results[feature]['test_type']} (p-value = {statistical_results[feature]['p_value']:.3f})")
            if feature in posthoc_results:
                print(f"  Post-hoc Test ({posthoc_results[feature]['test']}):")
                print(posthoc_results[feature]['results'])
            else:
                print("  No post-hoc results available (e.g., not enough data for post-hoc or an error occurred).")

print("\n--- Analysis Complete ---")
```