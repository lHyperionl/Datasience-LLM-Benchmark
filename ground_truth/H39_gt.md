```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set a professional style for plots
plt.style.use('seaborn-v0_8-darkgrid')

# --- 1. Load the dataset ---
# Assumes 'mobiles_2025.csv' is in the current working directory.
try:
    df = pd.read_csv('mobiles_2025.csv')
    print("Dataset 'mobiles_2025.csv' loaded successfully.")
except FileNotFoundError:
    print("Error: 'mobiles_2025.csv' not found. Please ensure the file is in the correct directory.")
    exit() # Exit if the dataset cannot be loaded

# Store initial row count for comparison after cleaning
initial_row_count = len(df)

# --- 2. Clean `RAM`, `Battery Capacity`, and `Launched Price (USA)` columns ---

# Clean 'RAM' column: remove 'GB', convert to numeric, handle errors
# Convert to string first to ensure .str accessor works, then remove 'GB' and strip whitespace
df['RAM'] = df['RAM'].astype(str).str.replace('GB', '', regex=False).str.strip()
# Convert to numeric, coercing any conversion errors to NaN
df['RAM'] = pd.to_numeric(df['RAM'], errors='coerce')
print("Cleaned 'RAM' column.")

# Clean 'Battery Capacity' column: remove 'mAh', convert to numeric, handle errors
# Convert to string, remove 'mAh', and strip whitespace
df['Battery Capacity'] = df['Battery Capacity'].astype(str).str.replace('mAh', '', regex=False).str.strip()
# Convert to numeric, coercing any conversion errors to NaN
df['Battery Capacity'] = pd.to_numeric(df['Battery Capacity'], errors='coerce')
print("Cleaned 'Battery Capacity' column.")

# Clean 'Launched Price (USA)' column: remove 'USD ', commas, convert to numeric, handle errors
# Convert to string, remove 'USD ' and commas, and strip whitespace
df['Launched Price (USA)'] = df['Launched Price (USA)'].astype(str).str.replace('USD ', '', regex=False).str.replace(',', '', regex=False).str.strip()
# Convert to numeric, coercing any conversion errors to NaN
df['Launched Price (USA)'] = pd.to_numeric(df['Launched Price (USA)'], errors='coerce')
print("Cleaned 'Launched Price (USA)' column.")

# Ensure 'Launched Year' is numeric and convert to integer
# Coerce errors to NaN, then drop rows where year is NaN
df['Launched Year'] = pd.to_numeric(df['Launched Year'], errors='coerce')
df.dropna(subset=['Launched Year'], inplace=True)
df['Launched Year'] = df['Launched Year'].astype(int)
print("Cleaned 'Launched Year' column.")

# Drop rows where any of the three cleaned columns (RAM, Battery Capacity, Launched Price (USA)) are NaN
columns_to_check_nan = ['RAM', 'Battery Capacity', 'Launched Price (USA)']
df.dropna(subset=columns_to_check_nan, inplace=True)
rows_dropped_cleaning = initial_row_count - len(df)
print(f"Dropped {rows_dropped_cleaning} rows due to NaN values in cleaned RAM, Battery Capacity, or Launched Price (USA).")

# --- 3. Filter the data to include only records where `Company Name` is 'Apple' ---
# Use .copy() to avoid SettingWithCopyWarning when modifying the filtered DataFrame later
apple_df = df[df['Company Name'] == 'Apple'].copy()
print(f"Filtered data to include {len(apple_df)} Apple products.")

# Check if there's any Apple data left after cleaning and filtering
if apple_df.empty:
    print("No Apple products found in the dataset after cleaning and filtering. Cannot perform analysis.")
else:
    # --- 4. Group the filtered Apple data by `Launched Year` ---
    # Get unique launched years for Apple products and sort them for chronological plotting
    unique_apple_years = sorted(apple_df['Launched Year'].unique())
    print(f"Found Apple product data for years: {unique_apple_years}")

    # Define the metrics to be visualized and their corresponding y-axis labels
    metrics_to_plot = {
        'RAM': 'RAM (GB)',
        'Battery Capacity': 'Battery Capacity (mAh)',
        'Launched Price (USA)': 'Launched Price (USD)'
    }

    # --- 5. For each `Launched Year`, generate box plots ---
    # Iterate through each unique year to create separate plots for each year
    for year in unique_apple_years:
        # Filter data for the current year
        year_data = apple_df[apple_df['Launched Year'] == year]

        # Skip if no data for this specific year (e.g., if all products for a year were dropped due to NaNs)
        if year_data.empty:
            print(f"No valid data for Apple products in year {year} after cleaning. Skipping plots for this year.")
            continue

        # Create a figure with three subplots (one row, three columns)
        fig, axes = plt.subplots(1, 3, figsize=(20, 7)) # Adjust figure size for better readability
        # Set a main title for the entire figure, indicating the year
        fig.suptitle(f'Apple Product Distributions for Year {year}', fontsize=18, weight='bold')

        # Iterate through each metric to plot on its respective subplot
        for i, (column_name, y_label) in enumerate(metrics_to_plot.items()):
            # Generate a box plot for the current metric
            sns.boxplot(y=year_data[column_name], ax=axes[i], color='skyblue', width=0.3)
            # Set the title for the subplot, indicating the metric
            axes[i].set_title(f'{column_name} Distribution', fontsize=14)
            # Set the y-axis label
            axes[i].set_ylabel(y_label, fontsize=12)
            # Remove the x-axis label as it's a single box plot per subplot
            axes[i].set_xlabel('')
            # Add a grid for better readability
            axes[i].grid(True, linestyle='--', alpha=0.7)

        # Adjust layout to prevent titles and labels from overlapping
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # rect=[left, bottom, right, top] adjusts space for suptitle
        # Display the plots for the current year
        plt.show()

    print("\nComparative analysis plots generated successfully for Apple products over the years.")

```