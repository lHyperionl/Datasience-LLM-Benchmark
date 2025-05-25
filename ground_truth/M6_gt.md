```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np

# --- Configuration and Data Loading ---
# Set Seaborn style for better aesthetics
sns.set_style("whitegrid")

# Configure Matplotlib for better resolution and default figure size
plt.rcParams['figure.dpi'] = 100
plt.rcParams['figure.figsize'] = (10, 6) # Default figure size for individual plots

# Load the dataset
try:
    df = pd.read_csv('titanic.csv')
except FileNotFoundError:
    print("Error: 'titanic.csv' not found. Please ensure the file is in the current directory.")
    exit()

# --- Data Preprocessing ---
# Handle missing 'Age' values by filling with the median.
# Median is preferred over mean for potentially skewed distributions like age.
df['Age'].fillna(df['Age'].median(), inplace=True)

# Handle missing 'Fare' values (though typically complete for Titanic, good practice)
df['Fare'].fillna(df['Fare'].median(), inplace=True)

# Create 'AgeGroup' bins for categorical analysis of age
bins = [0, 12, 18, 35, 60, np.inf]
labels = ['Child', 'Teenager', 'Young Adult', 'Adult', 'Senior']
df['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False, include_lowest=True)

# Convert 'Survived' to a more descriptive categorical type for plotting labels
df['Survived_Cat'] = df['Survived'].map({0: 'No', 1: 'Yes'})

# Convert 'Pclass' to categorical for better plotting labels and ordering
df['Pclass_Cat'] = df['Pclass'].map({1: '1st Class', 2: '2nd Class', 3: '3rd Class'})
# Ensure the order of classes for consistent plotting
df['Pclass_Cat'] = pd.Categorical(df['Pclass_Cat'], categories=['1st Class', '2nd Class', '3rd Class'], ordered=True)

# --- 1. Multi-panel figure: Survival rates by different demographic groups ---
# Calculate survival rates for each group
survival_by_class = df.groupby('Pclass_Cat')['Survived'].mean().reset_index()
survival_by_gender = df.groupby('Sex')['Survived'].mean().reset_index()
survival_by_agegroup = df.groupby('AgeGroup')['Survived'].mean().reset_index()

# Create a figure with 1 row and 3 columns for the subplots
fig1, axes1 = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
fig1.suptitle('Survival Rates by Key Demographic Groups', fontsize=18, y=1.02) # Overall title

# Plot 1.1: Survival Rate by Passenger Class
sns.barplot(x='Pclass_Cat', y='Survived', data=survival_by_class, ax=axes1[0], palette='viridis')
axes1[0].set_title('By Passenger Class', fontsize=14)
axes1[0].set_ylabel('Survival Rate')
axes1[0].set_xlabel('Passenger Class')
axes1[0].set_ylim(0, 1) # Set y-axis limit for consistency

# Plot 1.2: Survival Rate by Gender
sns.barplot(x='Sex', y='Survived', data=survival_by_gender, ax=axes1[1], palette='viridis')
axes1[1].set_title('By Gender', fontsize=14)
axes1[1].set_ylabel('') # Shared y-axis, no need for redundant label
axes1[1].set_xlabel('Gender')
axes1[1].set_ylim(0, 1)

# Plot 1.3: Survival Rate by Age Group
sns.barplot(x='AgeGroup', y='Survived', data=survival_by_agegroup, ax=axes1[2], palette='viridis')
axes1[2].set_title('By Age Group', fontsize=14)
axes1[2].set_ylabel('') # Shared y-axis
axes1[2].set_xlabel('Age Group')
axes1[2].set_ylim(0, 1)

plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
plt.show()

# --- 2. Heatmap: Survival rates across combinations of class and gender ---
# Create a pivot table to calculate the mean survival rate for each combination
survival_pivot = df.pivot_table(values='Survived', index='Pclass_Cat', columns='Sex', aggfunc='mean')

plt.figure(figsize=(8, 6))
sns.heatmap(survival_pivot, annot=True, fmt=".2f", cmap="YlGnBu", linewidths=.5, linecolor='black',
            cbar_kws={'label': 'Average Survival Rate'})
plt.title('Survival Rates by Passenger Class and Gender', fontsize=16)
plt.xlabel('Gender')
plt.ylabel('Passenger Class')
plt.show()

# --- 3. Violin plots: Age distributions of survivors vs non-survivors by gender ---
plt.figure(figsize=(12, 7))
# 'split=True' allows comparing distributions within the same violin for 'Survived_Cat'
sns.violinplot(x='Sex', y='Age', hue='Survived_Cat', data=df, palette='muted', split=True, inner='quartile')
plt.title('Age Distribution of Survivors vs. Non-Survivors by Gender', fontsize=16)
plt.xlabel('Gender')
plt.ylabel('Age')
plt.legend(title='Survival Status')
plt.show()

# --- 4. Interactive scatter plot: Age, Fare, and Survival with Passenger Class colors ---
# Prepare 'Survived' column as string for Plotly symbol mapping
df['Survived_Str'] = df['Survived'].map({0: 'Perished', 1: 'Survived'})

fig_interactive = px.scatter(df,
                             x='Age',
                             y='Fare',
                             color='Pclass_Cat', # Color points by passenger class
                             symbol='Survived_Str', # Use different symbols for survival status
                             hover_name='Name', # Show passenger name on hover
                             hover_data={'Age': True, 'Fare': True, 'Sex': True, 'Pclass_Cat': True, 'Survived_Str': True}, # Additional info on hover
                             title='Age vs. Fare by Passenger Class and Survival',
                             labels={'Pclass_Cat': 'Passenger Class', 'Survived_Str': 'Survival Status'},
                             color_discrete_map={'1st Class': '#1f77b4', '2nd Class': '#ff7f0e', '3rd Class': '#2ca02c'}, # Custom colors
                             category_orders={'Pclass_Cat': ['1st Class', '2nd Class', '3rd Class']} # Ensure class order
                            )

fig_interactive.update_layout(
    xaxis_title='Age',
    yaxis_title='Fare',
    legend_title='Passenger Class',
    hovermode='closest', # Show hover info for the closest point
    height=600 # Set plot height
)
fig_interactive.show()

# --- 5. Comprehensive Dashboard-style figure combining multiple visualizations ---
# Using matplotlib.gridspec for a custom and flexible layout
fig_dashboard = plt.figure(figsize=(20, 15))
# Define a 3x3 grid, with the last row being shorter for text insights
gs = fig_dashboard.add_gridspec(3, 3, height_ratios=[1, 1, 0.5])

# Plot 5.1: Survival Rate by Passenger Class (Top-Left)
ax_dash1 = fig_dashboard.add_subplot(gs[0, 0])
sns.barplot(x='Pclass_Cat', y='Survived', data=survival_by_class, ax=ax_dash1, palette='coolwarm')
ax_dash1.set_title('Survival Rate by Passenger Class', fontsize=14)
ax_dash1.set_ylabel('Survival Rate')
ax_dash1.set_xlabel('')
ax_dash1.set_ylim(0, 1)

# Plot 5.2: Survival Rate by Gender (Top-Middle)
ax_dash2 = fig_dashboard.add_subplot(gs[0, 1])
sns.barplot(x='Sex', y='Survived', data=survival_by_gender, ax=ax_dash2, palette='coolwarm')
ax_dash2.set_title('Survival Rate by Gender', fontsize=14)
ax_dash2.set_ylabel('')
ax_dash2.set_xlabel('')
ax_dash2.set_ylim(0, 1)

# Plot 5.3: Survival Rate by Age Group (Top-Right)
ax_dash3 = fig_dashboard.add_subplot(gs[0, 2])
sns.barplot(x='AgeGroup', y='Survived', data=survival_by_agegroup, ax=ax_dash3, palette='coolwarm')
ax_dash3.set_title('Survival Rate by Age Group', fontsize=14)
ax_dash3.set_ylabel('')
ax_dash3.set_xlabel('')
ax_dash3.set_ylim(0, 1)

# Plot 5.4: Age Distribution (Survivors vs Non-Survivors) - Middle Row, spans 2 columns
ax_dash4 = fig_dashboard.add_subplot(gs[1, :2]) # Spans columns 0 and 1
sns.kdeplot(data=df, x='Age', hue='Survived_Cat', fill=True, common_norm=False, palette='viridis', ax=ax_dash4, alpha=0.6)
ax_dash4.set_title('Age Distribution: Survivors vs. Non-Survivors', fontsize=14)
ax_dash4.set_xlabel('Age')
ax_dash4.set_ylabel('Density')
ax_dash4.legend(title='Survival Status')

# Plot 5.5: Fare Distribution (Survivors vs Non-Survivors) - Middle Row, rightmost column
ax_dash5 = fig_dashboard.add_subplot(gs[1, 2])
sns.histplot(data=df, x='Fare', hue='Survived_Cat', kde=True, palette='viridis', ax=ax_dash5, alpha=0.6, stat='density', common_norm=False)
ax_dash5.set_title('Fare Distribution: Survivors vs. Non-Survivors', fontsize=14)
ax_dash5.set_xlabel('Fare')
ax_dash5.set_ylabel('Density')
ax_dash5.legend(title='Survival Status')

# Plot 5.6: Insights/Summary Text (Bottom Row, spans all columns)
ax_dash6 = fig_dashboard.add_subplot(gs[2, :])
ax_dash6.set_facecolor('#f0f0f0') # Light grey background for the text box
ax_dash6.text(0.02, 0.95, "Key Insights from Titanic Survival Analysis:", fontsize=16, fontweight='bold', va='top')
ax_dash6.text(0.02, 0.85, "- Women and children (especially those in 1st/2nd class) had significantly higher survival rates.", fontsize=12, va='top')
ax_dash6.text(0.02, 0.75, "- Passengers in 1st class had the highest survival probability, indicating 'class privilege'.", fontsize=12, va='top')
ax_dash6.text(0.02, 0.65, "- Survival rates generally decreased with age, with 'Young Adults' having a slightly lower rate than 'Children' or 'Teenagers'.", fontsize=12, va='top')
ax_dash6.text(0.02, 0.55, "- Higher fare paid correlates with higher survival probability, likely due to better cabins/access.", fontsize=12, va='top')
ax_dash6.text(0.02, 0.45, "- The interactive plot (shown separately) allows exploring individual passenger data points.", fontsize=12, va='top')
ax_dash6.set_xticks([]) # Hide x-axis ticks
ax_dash6.set_yticks([]) # Hide y-axis ticks
# Hide spines (borders) for a cleaner text box appearance
ax_dash6.spines['top'].set_visible(False)
ax_dash6.spines['right'].set_visible(False)
ax_dash6.spines['bottom'].set_visible(False)
ax_dash6.spines['left'].set_visible(False)
ax_dash6.set_title('Summary and Key Insights', fontsize=16, pad=20) # Title for the insights section

plt.tight_layout(rect=[0, 0.03, 1, 0.97]) # Adjust layout to prevent title overlap
fig_dashboard.suptitle('Comprehensive Titanic Survival Analysis Dashboard', fontsize=22, y=0.99) # Overall title for the dashboard
plt.show()
```