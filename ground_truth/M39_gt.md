```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

# 1. Load the dataset
# Assumes 'mobiles_2025.csv' is in the current working directory.
try:
    df = pd.read_csv('mobiles_2025.csv')
except FileNotFoundError:
    print("Error: 'mobiles_2025.csv' not found. Please ensure the file is in the current directory.")
    exit()

# 2. Clean specified columns and convert to numeric
# Define columns to be cleaned and their respective cleaning patterns
columns_to_clean = {
    'RAM': {'pattern': 'GB', 'type': float},
    'Battery Capacity': {'pattern': 'mAh', 'type': float},
    'Screen Size': {'pattern': ' inches', 'type': float},
    'Launched Price (USA)': {'pattern': ['USD ', ','], 'type': float}
}

for col, details in columns_to_clean.items():
    if col in df.columns:
        # Convert column to string type first to handle potential mixed types gracefully
        df[col] = df[col].astype(str)
        
        # Remove specified patterns
        if isinstance(details['pattern'], list):
            for p in details['pattern']:
                df[col] = df[col].str.replace(p, '', regex=False)
        else:
            df[col] = df[col].str.replace(details['pattern'], '', regex=False)
        
        # Remove any leading/trailing whitespace after cleaning
        df[col] = df[col].str.strip()
        
        # Convert to numeric, coercing errors to NaN
        df[col] = pd.to_numeric(df[col], errors='coerce')
    else:
        print(f"Warning: Column '{col}' not found in the dataset. Skipping cleaning for this column.")

# Drop rows where any of the specified cleaned columns have NaN values
# This ensures that only valid numeric data is used for modeling
cleaned_numeric_cols = list(columns_to_clean.keys())
df.dropna(subset=cleaned_numeric_cols, inplace=True)

# Check if there's enough data left after cleaning
if df.empty:
    print("Error: No valid data remaining after cleaning and dropping NaN values. Cannot proceed with model training.")
    exit()

# 3. Define features X and target y
# Features are the cleaned 'RAM', 'Battery Capacity', and 'Screen Size'
X = df[['RAM', 'Battery Capacity', 'Screen Size']]
# Target is the cleaned 'Launched Price (USA)'
y = df['Launched Price (USA)']

# 4. Split the data into training (80%) and testing (20%) sets
# random_state ensures reproducibility of the split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train a Linear Regression model on the training data
model = LinearRegression()
model.fit(X_train, y_train)

# 6. Evaluate the model on the test set
# Make predictions on the test features
y_pred = model.predict(X_test)

# Calculate R-squared (R2) score
r2 = r2_score(y_test, y_pred)

# Calculate Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test, y_pred)

# Report the evaluation metrics
print(f"Model Evaluation Results:")
print(f"R-squared (R2) score: {r2:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
```