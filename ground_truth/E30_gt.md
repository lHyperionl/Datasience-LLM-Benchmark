```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define the path to the dataset
file_path = 'heart.csv'

try:
    # Load the dataset using pandas
    df = pd.read_csv(file_path)

    # --- 1. Generate a histogram to visualize the distribution of the `Age` column ---
    # Create a new figure for the histogram
    plt.figure(figsize=(10, 6))
    # Use seaborn's histplot for a visually appealing histogram
    # kde=True adds a Kernel Density Estimate line
    # bins='auto' lets seaborn determine an optimal number of bins
    sns.histplot(data=df, x='Age', kde=True, bins='auto', color='skyblue')
    # Set the title of the plot
    plt.title('Distribution of Age', fontsize=16)
    # Set the x-axis label
    plt.xlabel('Age (Years)', fontsize=12)
    # Set the y-axis label
    plt.ylabel('Number of Individuals', fontsize=12)
    # Add a grid for better readability
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    # Adjust layout to prevent labels from overlapping
    plt.tight_layout()
    # Display the plot
    plt.show()

    # --- 2. Generate a bar chart to show the distribution of the `Sex` column (M, F) ---
    # Create a new figure for the bar chart
    plt.figure(figsize=(7, 5))
    # Assuming 'Sex' column contains 0 for Female and 1 for Male (common in heart.csv)
    # Create a temporary column with descriptive labels for better visualization
    df['Sex_Label'] = df['Sex'].map({1: 'Male', 0: 'Female'})
    # Use seaborn's countplot to show the counts of each category in 'Sex_Label'
    sns.countplot(data=df, x='Sex_Label', palette='viridis')
    # Set the title of the plot
    plt.title('Distribution of Sex', fontsize=16)
    # Set the x-axis label
    plt.xlabel('Sex', fontsize=12)
    # Set the y-axis label
    plt.ylabel('Number of Individuals', fontsize=12)
    # Ensure x-axis labels are horizontal for better readability
    plt.xticks(rotation=0)
    # Add a grid for better readability
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    # Adjust layout to prevent labels from overlapping
    plt.tight_layout()
    # Display the plot
    plt.show()

except FileNotFoundError:
    # Handle the case where the heart.csv file is not found
    print(f"Error: The file '{file_path}' was not found.")
    print("Please ensure 'heart.csv' is in the same directory as the script.")
except KeyError as e:
    # Handle the case where expected columns are missing from the dataset
    print(f"Error: Missing expected column in the dataset: {e}.")
    print("Please check if 'Age' and 'Sex' columns exist in 'heart.csv'.")
except Exception as e:
    # Handle any other unexpected errors during execution
    print(f"An unexpected error occurred: {e}")
```