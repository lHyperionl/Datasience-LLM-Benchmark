```python
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# --- Configuration ---
# Define the path to the dataset
DATASET_PATH = 'winequality-red.csv'

# Define pH group boundaries and labels
PH_BINS = [0, 3.2, 3.4, np.inf]  # np.inf for the upper bound of the last bin
PH_LABELS = ['Low pH (< 3.2)', 'Medium pH (3.2 to 3.4)', 'High pH (> 3.4)']
# Ensure the order of categories for consistent plotting and grouping
PH_GROUP_ORDER = ['Low pH (< 3.2)', 'Medium pH (3.2 to 3.4)', 'High pH (> 3.4)']

# Confidence interval level for visualization
CONFIDENCE_LEVEL = 0.95

# --- Data Loading ---
try:
    df = pd.read_csv(DATASET_PATH)
except FileNotFoundError:
    print(f"Error: '{DATASET_PATH}' not found. Please ensure the file is in the current directory.")
    exit()

# --- 1. Categorize wines based on 'pH' values ---
# Create a new column 'pH_group' by categorizing 'pH' values
# right=False means the bins are [lower, upper)
df['pH_group'] = pd.cut(df['pH'], bins=PH_BINS, labels=PH_LABELS, right=False, include_lowest=True)

# Convert 'pH_group' to a categorical type with a specified order
# This ensures consistent ordering in groupby and plots
df['pH_group'] = pd.Categorical(df['pH_group'], categories=PH_GROUP_ORDER, ordered=True)

# --- 2. Calculate and report the average 'quality' for each pH group ---
# Group by 'pH_group' and calculate mean, standard error of the mean (SEM), and count for 'quality'
group_stats = df.groupby('pH_group')['quality'].agg(
    mean_quality='mean',
    sem_quality='sem',
    count_quality='count'
)

# Calculate 95% Confidence Intervals for the mean quality of each group
group_stats['ci_lower'] = np.nan
group_stats['ci_upper'] = np.nan

for group_name, row in group_stats.iterrows():
    mean = row['mean_quality']
    sem = row['sem_quality']
    count = row['count_quality']

    # Calculate confidence interval only if there's enough data (df > 0)
    if count > 1:
        # stats.t.interval(alpha, df, loc=mean, scale=sem)
        # alpha is the confidence level (e.g., 0.95 for 95% CI)
        # df is degrees of freedom (count - 1)
        # loc is the mean, scale is the standard error of the mean
        ci_lower, ci_upper = stats.t.interval(CONFIDENCE_LEVEL, df=count - 1, loc=mean, scale=sem)
        group_stats.loc[group_name, 'ci_lower'] = ci_lower
        group_stats.loc[group_name, 'ci_upper'] = ci_upper
    else:
        # If only one data point, CI cannot be calculated meaningfully, set to mean
        group_stats.loc[group_name, 'ci_lower'] = mean
        group_stats.loc[group_name, 'ci_upper'] = mean

# Calculate the error bar length (distance from mean to lower CI bound)
group_stats['error_bar_length'] = group_stats['mean_quality'] - group_stats['ci_lower']

print("--- Average 'quality' for each pH group with 95% Confidence Intervals ---")
print(group_stats[['mean_quality', 'ci_lower', 'ci_upper']].round(3))
print("\n")

# --- 3. Perform an ANOVA test ---
# Extract 'quality' scores for each pH group into separate arrays
quality_groups_for_anova = []
for group_label in PH_GROUP_ORDER:
    group_data = df[df['pH_group'] == group_label]['quality'].dropna()
    if not group_data.empty:
        quality_groups_for_anova.append(group_data)

print("--- ANOVA Test Results ---")
# Perform one-way ANOVA test if there are at least two groups with data
if len(quality_groups_for_anova) >= 2:
    f_statistic, p_value = stats.f_oneway(*quality_groups_for_anova)
    print(f"F-statistic: {f_statistic:.3f}")
    print(f"P-value: {p_value:.3f}")

    # Interpret the p-value
    alpha = 0.05
    if p_value < alpha:
        print(f"Conclusion: The p-value ({p_value:.3f}) is less than {alpha}, indicating a statistically significant difference in mean 'quality' among pH groups.")
    else:
        print(f"Conclusion: The p-value ({p_value:.3f}) is greater than or equal to {alpha}, indicating no statistically significant difference in mean 'quality' among pH groups.")
else:
    print("ANOVA test could not be performed: Not enough pH groups with data to compare.")
print("\n")

# --- 4. Create a visualization showing average qualities with error bars ---
plt.figure(figsize=(10, 6))

# Create a bar plot with error bars representing 95% confidence intervals
bars = plt.bar(
    group_stats.index,
    group_stats['mean_quality'],
    yerr=group_stats['error_bar_length'],  # Symmetric error bars
    capsize=5,  # Width of the error bar caps
    color='skyblue',
    edgecolor='black'
)

# Add labels and title
plt.xlabel('pH Group', fontsize=12)
plt.ylabel('Average Wine Quality', fontsize=12)
plt.title('Average Wine Quality by pH Group (with 95% CI)', fontsize=14)
plt.ylim(0, 10)  # Quality scores typically range from 0 to 10
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add text labels for average quality on top of bars
for bar in bars:
    yval = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        yval + 0.1,  # Position text slightly above the bar
        f'{yval:.2f}',  # Format to two decimal places
        ha='center',  # Horizontal alignment
        va='bottom',  # Vertical alignment
        fontsize=10
    )

plt.tight_layout()  # Adjust layout to prevent labels from overlapping
plt.show()
```