```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
import math

# Set plot styles for better aesthetics
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['figure.figsize'] = (12, 8) # Default figure size for Matplotlib plots

# --- 1. Load the dataset ---
try:
    df = pd.read_csv('world-happiness-report-2021.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: 'world-happiness-report-2021.csv' not found. Please ensure the file is in the current directory.")
    exit() # Exit if the file is not found

# Rename columns for easier access and better readability in plots
df.rename(columns={
    'Ladder score': 'Happiness Score',
    'Logged GDP per capita': 'GDP per Capita',
    'Social support': 'Social Support',
    'Healthy life expectancy': 'Healthy Life Expectancy',
    'Freedom to make life choices': 'Freedom',
    'Generosity': 'Generosity',
    'Perceptions of corruption': 'Perceptions of Corruption'
}, inplace=True)

# --- 2. Create a world map visualization showing happiness scores by country (using Plotly) ---
print("\nGenerating World Map of Happiness Scores...")
fig_map = px.choropleth(
    df,
    locations="Country name",
    locationmode="country names", # Plotly will try to match country names to ISO codes
    color="Happiness Score",
    hover_name="Country name",
    color_continuous_scale=px.colors.sequential.Plasma, # Choose a color scale
    title="World Happiness Score Map (2021)",
    height=600
)
fig_map.update_layout(
    geo=dict(
        showframe=False,
        showcoastlines=False,
        projection_type='equirectangular' # Or 'natural earth', 'orthographic' etc.
    )
)
fig_map.show()

# --- 3. Generate a horizontal bar chart showing average happiness scores by region ---
print("Generating Horizontal Bar Chart of Average Happiness by Region...")
avg_happiness_by_region = df.groupby('Regional indicator')['Happiness Score'].mean().sort_values(ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x=avg_happiness_by_region.values, y=avg_happiness_by_region.index, palette='viridis')
plt.title('Average Happiness Score by Regional Indicator (2021)', fontsize=16)
plt.xlabel('Average Happiness Score', fontsize=12)
plt.ylabel('Regional Indicator', fontsize=12)
plt.tight_layout() # Adjust layout to prevent labels from overlapping
plt.show()

# --- 4. Create a correlation heatmap showing relationships between happiness factors ---
print("Generating Correlation Heatmap...")
# Select relevant numerical columns for correlation analysis
correlation_cols = [
    'Happiness Score', 'GDP per Capita', 'Social Support',
    'Healthy Life Expectancy', 'Freedom', 'Generosity', 'Perceptions of Corruption'
]
correlation_matrix = df[correlation_cols].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, linecolor='black')
plt.title('Correlation Heatmap of Happiness Factors (2021)', fontsize=16)
plt.xticks(rotation=45, ha='right') # Rotate x-axis labels for better readability
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# --- 5. Plot scatter plots showing the relationship between GDP per capita vs Ladder score and Social support vs Ladder score ---
print("Generating Scatter Plots...")
plt.figure(figsize=(15, 6))

# Scatter plot: GDP per Capita vs Happiness Score
plt.subplot(1, 2, 1) # 1 row, 2 columns, 1st plot
sns.scatterplot(x='GDP per Capita', y='Happiness Score', data=df, hue='Regional indicator', palette='tab10', s=100, alpha=0.7)
plt.title('Happiness Score vs. GDP per Capita (2021)', fontsize=14)
plt.xlabel('Logged GDP per Capita', fontsize=12)
plt.ylabel('Happiness Score', fontsize=12)
plt.legend(title='Region', bbox_to_anchor=(1.05, 1), loc='upper left') # Place legend outside the plot
plt.grid(True, linestyle='--', alpha=0.6)

# Scatter plot: Social Support vs Happiness Score
plt.subplot(1, 2, 2) # 1 row, 2 columns, 2nd plot
sns.scatterplot(x='Social Support', y='Happiness Score', data=df, hue='Regional indicator', palette='tab10', s=100, alpha=0.7)
plt.title('Happiness Score vs. Social Support (2021)', fontsize=14)
plt.xlabel('Social Support', fontsize=12)
plt.ylabel('Happiness Score', fontsize=12)
plt.legend(title='Region', bbox_to_anchor=(1.05, 1), loc='upper left') # Place legend outside the plot
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()

# --- 6. Generate box plots comparing happiness scores across different regions ---
print("Generating Box Plots of Happiness Scores by Region...")
plt.figure(figsize=(14, 8))
sns.boxplot(x='Regional indicator', y='Happiness Score', data=df, palette='Set3')
plt.title('Distribution of Happiness Scores by Regional Indicator (2021)', fontsize=16)
plt.xlabel('Regional Indicator', fontsize=12)
plt.ylabel('Happiness Score', fontsize=12)
plt.xticks(rotation=45, ha='right') # Rotate x-axis labels for better readability
plt.tight_layout()
plt.show()

# --- 7. Create a radar chart showing the happiness profile of the top 5 happiest countries ---
print("Generating Radar Chart for Top 5 Happiest Countries...")

# Get top 5 happiest countries based on 'Happiness Score'
top_5_countries = df.nlargest(5, 'Happiness Score')

# Define features for the radar chart (these are the factors contributing to happiness)
radar_features = [
    'GDP per Capita', 'Social Support', 'Healthy Life Expectancy',
    'Freedom', 'Generosity', 'Perceptions of Corruption'
]

# Normalize the features using MinMaxScaler across the entire dataset
# This ensures that the values are scaled relative to the global min/max for each feature,
# allowing for a fair comparison on the radar chart.
scaler = MinMaxScaler()
df_scaled = df.copy()
df_scaled[radar_features] = scaler.fit_transform(df[radar_features])

# Get the scaled data for the top 5 countries
top_5_scaled = df_scaled[df_scaled['Country name'].isin(top_5_countries['Country name'])]

# Number of variables (features) we're plotting
num_vars = len(radar_features)

# Calculate angle for each axis (equally spaced)
angles = [n / float(num_vars) * 2 * math.pi for n in range(num_vars)]
angles += angles[:1] # Complete the circle by adding the first angle at the end

plt.figure(figsize=(10, 10))
ax = plt.subplot(111, polar=True) # Create a polar plot

# Plot each country's profile on the radar chart
for index, row in top_5_scaled.iterrows():
    values = row[radar_features].tolist()
    values += values[:1] # Complete the circle for plotting
    ax.plot(angles, values, linewidth=2, linestyle='solid', label=row['Country name'])
    ax.fill(angles, values, alpha=0.25) # Fill the area under the plot

# Set the labels for each axis
ax.set_xticks(angles[:-1])
ax.set_xticklabels(radar_features, fontsize=10)

# Set y-limits and grid for normalized values (0 to 1)
ax.set_ylim(0, 1)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=8)

plt.title('Happiness Profile of Top 5 Happiest Countries (2021)', size=16, color='black', y=1.1)
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1)) # Place legend outside the plot
plt.tight_layout()
plt.show()

print("\nAll visualizations generated successfully.")
```