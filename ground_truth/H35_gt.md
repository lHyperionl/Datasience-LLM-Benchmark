```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def analyze_youtube_trends(file_path='youtube_trendings.csv'):
    """
    Analyzes trends in YouTube trending videos from a CSV dataset.

    The analysis includes:
    1. Loading the dataset.
    2. Cleaning the 'Video views' column (removing commas, converting to numeric, filling NaNs).
    3. Extracting and converting the 'published' year to a numeric type.
    4. Grouping data by the 'published' year.
    5. Calculating average video views and total unique videos per year.
    6. Generating two line plots to visualize these trends.

    Args:
        file_path (str): The path to the youtube_trendings.csv dataset.
    """
    # --- 1. Load the dataset ---
    try:
        df = pd.read_csv(file_path)
        print(f"Dataset '{file_path}' loaded successfully. Shape: {df.shape}")
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found. Please ensure it's in the current directory.")
        return
    except Exception as e:
        print(f"An error occurred while loading the dataset: {e}")
        return

    # --- 2. For the 'Video views' column: remove commas, convert to numeric, and fill resulting NaNs with 0 ---
    # Convert to string first to handle potential mixed types before string operations
    df['Video views'] = df['Video views'].astype(str)
    # Remove commas from the 'Video views' string
    df['Video views'] = df['Video views'].str.replace(',', '', regex=False)
    # Convert the cleaned string to numeric, coercing errors to NaN
    df['Video views'] = pd.to_numeric(df['Video views'], errors='coerce')
    # Fill any resulting NaNs (from conversion errors or original NaNs) with 0
    df['Video views'] = df['Video views'].fillna(0)
    print("Cleaned 'Video views' column.")

    # --- 3. Ensure the 'published' column (year) is treated as a numeric or integer type ---
    # Assuming 'publishedAt' is the column containing the full publication date string.
    # Convert 'publishedAt' to datetime objects, coercing errors to NaT (Not a Time)
    df['published_datetime'] = pd.to_datetime(df['publishedAt'], errors='coerce')
    # Extract the year from the datetime object
    df['published_year'] = df['published_datetime'].dt.year
    # Fill any NaNs (from failed datetime conversion) with 0 and convert to integer
    # Note: Years filled with 0 might represent unparseable dates and will be grouped as '0'.
    df['published_year'] = df['published_year'].fillna(0).astype(int)
    print("Processed 'published' year column.")

    # --- 4. Group the data by the 'published' year ---
    # --- 5. For each year, calculate two metrics: ---
    # a) the average of the cleaned 'Video views'
    # b) the total number of unique videos (count of unique 'Video' titles)
    # Filter out year 0 if it's an artifact of NaN filling and not a real year
    yearly_trends = df[df['published_year'] > 0].groupby('published_year').agg(
        average_views=('Video views', 'mean'),
        unique_videos=('Video', 'nunique')
    ).reset_index()

    # Sort by year to ensure correct plotting order
    yearly_trends = yearly_trends.sort_values('published_year')
    print("Calculated yearly trends.")
    print(yearly_trends.head())

    # --- 6. Create a figure with two subplots ---
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 10))
    fig.suptitle('YouTube Trending Video Analysis by Year', fontsize=16)

    # --- First subplot: Line graph showing the average cleaned 'Video views' per year ---
    axes[0].plot(yearly_trends['published_year'], yearly_trends['average_views'], marker='o', linestyle='-', color='skyblue')
    axes[0].set_title('Average Video Views Per Year', fontsize=14)
    axes[0].set_xlabel('Year', fontsize=12)
    axes[0].set_ylabel('Average Views', fontsize=12)
    axes[0].grid(True, linestyle='--', alpha=0.7)
    axes[0].ticklabel_format(style='plain', axis='y') # Prevent scientific notation on y-axis
    # Set x-axis ticks to be integer years
    axes[0].set_xticks(yearly_trends['published_year'].unique())
    axes[0].tick_params(axis='x', rotation=45)

    # --- Second subplot: Line graph showing the total number of unique videos per year ---
    axes[1].plot(yearly_trends['published_year'], yearly_trends['unique_videos'], marker='o', linestyle='-', color='lightcoral')
    axes[1].set_title('Total Number of Unique Videos Per Year', fontsize=14)
    axes[1].set_xlabel('Year', fontsize=12)
    axes[1].set_ylabel('Number of Unique Videos', fontsize=12)
    axes[1].grid(True, linestyle='--', alpha=0.7)
    # Set x-axis ticks to be integer years
    axes[1].set_xticks(yearly_trends['published_year'].unique())
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout(rect=[0, 0.03, 1, 0.96]) # Adjust layout to prevent suptitle overlap
    plt.show()
    print("Plots generated successfully.")

if __name__ == "__main__":
    # Call the analysis function with the dataset file path
    analyze_youtube_trends('youtube_trendings.csv')
```