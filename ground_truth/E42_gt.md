```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define the dataset file path
csv_file_path = 'imdb_top_250.csv'

# --- Load the dataset ---
try:
    df = pd.read_csv(csv_file_path)
except FileNotFoundError:
    print(f"Error: The file '{csv_file_path}' was not found.")
    print("Please ensure the CSV file is in the same directory as the script.")
    exit()

# --- Data Preprocessing ---

# Ensure 'Rating' column exists for plotting
if 'Rating' not in df.columns:
    print("Error: 'Rating' column not found in the dataset. Cannot proceed with plotting.")
    exit()

# Prepare 'Runtime' column for plotting.
# It might be in a string format like '120 min', so we need to extract numeric values.
original_runtime_col_name = 'Runtime'
processed_runtime_col_name = 'Runtime_minutes' # New column for processed numeric runtime

if original_runtime_col_name in df.columns:
    # Extract numeric part from 'Runtime' string (e.g., '120 min' -> '120')
    # and convert to numeric, coercing errors to NaN.
    # [0] is used because .str.extract returns a DataFrame.
    df[processed_runtime_col_name] = pd.to_numeric(
        df[original_runtime_col_name].astype(str).str.extract('(\d+)')[0],
        errors='coerce'
    )
    # Drop rows where Runtime could not be converted (NaN) as they cannot be plotted
    df.dropna(subset=[processed_runtime_col_name], inplace=True)
    
    # Check if any valid runtime data remains after cleaning
    if df[processed_runtime_col_name].empty:
        print(f"Warning: No valid numeric data found in '{original_runtime_col_name}' after processing. Scatter plot will be skipped.")
        # Set to None to indicate that the column is not usable for plotting
        original_runtime_col_name = None 
    else:
        # Update the column name to use the newly processed numeric column
        original_runtime_col_name = processed_runtime_col_name 
else:
    print(f"Warning: '{original_runtime_col_name}' column not found. Scatter plot for Runtime vs Rating will be skipped.")
    # Set to None to indicate that the column is not available
    original_runtime_col_name = None 

# --- 1. Create a histogram to visualize the distribution of the `Rating` column ---

# Set figure size for better readability
plt.figure(figsize=(10, 6)) 
# Use seaborn's histplot for a nicer histogram with a Kernel Density Estimate (KDE)
sns.histplot(df['Rating'], bins=10, kde=True, color='skyblue') 
# Add a descriptive title to the plot
plt.title('Distribution of Movie Ratings (IMDb Top 250)', fontsize=16) 
# Label the x-axis
plt.xlabel('Rating', fontsize=12) 
# Label the y-axis
plt.ylabel('Number of Movies', fontsize=12) 
# Add a grid for better readability
plt.grid(axis='y', linestyle='--', alpha=0.7) 
# Adjust layout to prevent labels from overlapping
plt.tight_layout() 
# Display the plot
plt.show() 

# --- 2. Create a scatter plot to visualize the relationship between `Rating` and `Runtime` ---

# Only proceed if a valid runtime column is available after preprocessing
if original_runtime_col_name: 
    # Set figure size
    plt.figure(figsize=(10, 6)) 
    # Use seaborn's scatterplot for visualizing the relationship
    sns.scatterplot(x=df[original_runtime_col_name], y=df['Rating'], alpha=0.7, color='coral') 
    # Add a descriptive title to the plot
    plt.title('Movie Rating vs. Runtime (IMDb Top 250)', fontsize=16) 
    # Label the x-axis
    plt.xlabel('Runtime (minutes)', fontsize=12) 
    # Label the y-axis
    plt.ylabel('Rating', fontsize=12) 
    # Add a grid for better readability
    plt.grid(True, linestyle='--', alpha=0.7) 
    # Adjust layout
    plt.tight_layout() 
    # Display the plot
    plt.show()

```