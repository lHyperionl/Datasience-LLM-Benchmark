```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def create_heart_disease_chest_pain_grouped_bar_chart(file_path='heart.csv'):
    """
    Loads the heart.csv dataset and creates a grouped bar chart showing the count
    of HeartDisease (0 and 1) for each category in the ChestPainType column.

    Args:
        file_path (str): The path to the heart.csv dataset file.
    """
    try:
        # 1. Load the dataset
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found. Please ensure it's in the current directory.")
        return
    except Exception as e:
        print(f"An error occurred while loading the dataset: {e}")
        return

    # Ensure required columns exist
    required_columns = ['ChestPainType', 'HeartDisease']
    if not all(col in df.columns for col in required_columns):
        print(f"Error: Dataset must contain '{required_columns[0]}' and '{required_columns[1]}' columns.")
        return

    # Convert 'HeartDisease' to a categorical type for better legend labels
    # Map 0 to 'No Heart Disease' and 1 to 'Heart Disease'
    df['HeartDisease_Label'] = df['HeartDisease'].map({0: 'No Heart Disease', 1: 'Heart Disease'})

    # Define the order for ChestPainType for consistent plotting
    # Based on common understanding or dataset description (TA, ATA, NAP, ASY)
    chest_pain_order = ['TA', 'ATA', 'NAP', 'ASY']
    df['ChestPainType'] = pd.Categorical(df['ChestPainType'], categories=chest_pain_order, ordered=True)

    # Set the style for the plot
    sns.set_style("whitegrid")

    # 2. Create a grouped bar chart
    # 3. The x-axis should represent ChestPainType, and for each type, there should be
    #    two bars representing the count of HeartDisease=0 and HeartDisease=1.
    plt.figure(figsize=(10, 6)) # Set figure size for better readability
    sns.countplot(
        data=df,
        x='ChestPainType',
        hue='HeartDisease_Label', # Use the new labeled column for hue
        palette='viridis' # Choose a color palette
    )

    # 4. Ensure the plot has a title, labeled axes, and a legend.
    plt.title('Count of Heart Disease by Chest Pain Type', fontsize=16)
    plt.xlabel('Chest Pain Type', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.legend(title='Heart Disease', loc='upper right', bbox_to_anchor=(1.25, 1)) # Adjust legend position
    plt.xticks(rotation=0) # Keep x-axis labels horizontal if they fit

    # Adjust layout to prevent labels from overlapping
    plt.tight_layout()

    # Display the plot
    plt.show()

if __name__ == '__main__':
    # Example usage:
    # Make sure 'heart.csv' is in the same directory as this script,
    # or provide the full path to the file.
    create_heart_disease_chest_pain_grouped_bar_chart('heart.csv')
```