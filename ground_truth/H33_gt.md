```python
import pandas as pd
import matplotlib.pyplot as plt

# --- 1. Load data ---
try:
    df = pd.read_csv('spotify_2023.csv')
    print("Data loaded successfully.")
    print(f"Initial dataset shape: {df.shape}")
except FileNotFoundError:
    print("Error: 'spotify_2023.csv' not found. Please ensure the CSV file is in the same directory.")
    exit() # Exit if the file is not found

# --- 2. Convert `streams` to numeric (coerce errors to NaN) ---
# It's good practice to inspect the 'streams' column before conversion if possible,
# but the requirement is to coerce errors.
# Some values in 'streams' might be strings like '1,000,000' or non-numeric.
# We'll convert them to numeric, coercing any non-convertible values to NaN.
df['streams'] = pd.to_numeric(df['streams'], errors='coerce')
print(f"Number of NaN values in 'streams' after conversion: {df['streams'].isnull().sum()}")

# --- 3. Filter data ---
# Drop rows with NaN `streams` after conversion, as these cannot be used for numerical comparison.
df_filtered = df.dropna(subset=['streams'])
print(f"Dataset shape after dropping NaN 'streams': {df_filtered.shape}")

# Filter for songs with `streams` greater than 10,000,000
df_filtered = df_filtered[df_filtered['streams'] > 10_000_000]
print(f"Dataset shape after filtering for streams > 10M: {df_filtered.shape}")

# Filter for `released_year` between 2010 and 2023, inclusive
df_filtered = df_filtered[
    (df_filtered['released_year'] >= 2010) &
    (df_filtered['released_year'] <= 2023)
]
print(f"Dataset shape after filtering for released_year (2010-2023): {df_filtered.shape}")

# Check if there's any data left after filtering
if df_filtered.empty:
    print("No data left after filtering. Cannot perform analysis or plotting.")
    exit()

# --- 4. Group the filtered data by `released_year` ---
# --- 5. For each year, calculate the average `danceability_%`, `energy_%`, and `valence_%` ---
# Define the audio features to analyze
audio_features = ['danceability_%', 'energy_%', 'valence_%']

# Group by released_year and calculate the mean for the specified audio features
yearly_avg_features = df_filtered.groupby('released_year')[audio_features].mean()
print("\nYearly average audio features:")
print(yearly_avg_features.head()) # Display first few rows of the aggregated data

# --- 6. Create line plots for each of these three average audio features over `released_year` ---
# Set up the plot style
plt.style.use('seaborn-v0_8-darkgrid') # A nice style for plots

# Create a figure to hold all plots
plt.figure(figsize=(15, 5)) # Adjust figure size for better readability

# Plot for Danceability
plt.subplot(1, 3, 1) # 1 row, 3 columns, 1st plot
plt.plot(yearly_avg_features.index, yearly_avg_features['danceability_%'], marker='o', color='skyblue')
plt.title('Average Danceability Over Years (2010-2023)')
plt.xlabel('Released Year')
plt.ylabel('Average Danceability (%)')
plt.xticks(yearly_avg_features.index, rotation=45) # Rotate x-axis labels for better readability
plt.grid(True)

# Plot for Energy
plt.subplot(1, 3, 2) # 1 row, 3 columns, 2nd plot
plt.plot(yearly_avg_features.index, yearly_avg_features['energy_%'], marker='o', color='lightcoral')
plt.title('Average Energy Over Years (2010-2023)')
plt.xlabel('Released Year')
plt.ylabel('Average Energy (%)')
plt.xticks(yearly_avg_features.index, rotation=45)
plt.grid(True)

# Plot for Valence
plt.subplot(1, 3, 3) # 1 row, 3 columns, 3rd plot
plt.plot(yearly_avg_features.index, yearly_avg_features['valence_%'], marker='o', color='lightgreen')
plt.title('Average Valence Over Years (2010-2023)')
plt.xlabel('Released Year')
plt.ylabel('Average Valence (%)')
plt.xticks(yearly_avg_features.index, rotation=45)
plt.grid(True)

# Adjust layout to prevent titles/labels from overlapping
plt.tight_layout()

# Display all plots
plt.show()

print("\nAnalysis complete. Plots displayed.")
```