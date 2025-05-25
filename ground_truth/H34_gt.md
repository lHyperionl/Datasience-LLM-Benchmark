```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
import numpy as np

# --- 1. Load the dataset ---
try:
    df = pd.read_csv('youtube_trendings.csv')
except FileNotFoundError:
    print("Error: 'youtube_trendings.csv' not found. Please ensure the file is in the current directory.")
    exit()

# --- 2. Clean 'Likes', 'Dislikes', and 'Video views' columns ---
# List of columns to clean
numeric_cols_to_clean = ['Likes', 'Dislikes', 'Video views']

for col in numeric_cols_to_clean:
    if col in df.columns:
        # Remove commas
        df[col] = df[col].astype(str).str.replace(',', '', regex=False)
        # Convert to numeric, coercing errors to NaN
        df[col] = pd.to_numeric(df[col], errors='coerce')
        # Fill resulting NaNs with 0
        df[col] = df[col].fillna(0)
    else:
        print(f"Warning: Column '{col}' not found in the dataset.")

# Ensure the cleaned columns are of numeric type (e.g., int or float)
df['Likes'] = df['Likes'].astype(int)
df['Dislikes'] = df['Dislikes'].astype(int)
df['Video views'] = df['Video views'].astype(int)

# --- 3. Treat the 'published' column (year) as a numeric feature ---
if 'published' in df.columns:
    # Convert 'published' to datetime objects
    df['published_datetime'] = pd.to_datetime(df['published'], errors='coerce')
    # Extract the year
    df['published_year'] = df['published_datetime'].dt.year
    # Fill any NaNs in 'published_year' that might result from parsing errors with a reasonable default (e.g., mode or 0)
    # For simplicity and as per prompt, we'll fill with 0 if parsing failed, assuming it's an edge case.
    df['published_year'] = df['published_year'].fillna(0).astype(int)
else:
    print("Warning: 'published' column not found. 'published_year' feature will not be available.")
    df['published_year'] = 0 # Create a dummy column if not found

# --- 4. For the 'Category' column: fill missing values with 'Unknown', then apply one-hot encoding ---
if 'Category' in df.columns:
    # Fill missing values with 'Unknown'
    df['Category'] = df['Category'].fillna('Unknown')
    # Apply one-hot encoding
    # Use pd.get_dummies for simplicity and direct integration into DataFrame
    df_category_encoded = pd.get_dummies(df['Category'], prefix='Category', dtype=int)
else:
    print("Warning: 'Category' column not found. One-hot encoding for 'Category' will not be applied.")
    df_category_encoded = pd.DataFrame() # Create an empty DataFrame if not found

# --- 5. Define features X and target y ---
# Features: cleaned 'Dislikes', cleaned 'Video views', 'published_year', and one-hot encoded 'Category'
# Target: cleaned 'Likes'

# Combine all features into a single DataFrame X
# Start with the numeric features
X_numeric = df[['Dislikes', 'Video views', 'published_year']]

# Concatenate with one-hot encoded category features
# Use pd.concat to combine X_numeric and df_category_encoded
X = pd.concat([X_numeric, df_category_encoded], axis=1)

# Define the target variable y
y = df['Likes']

# --- 6. Split the data into training (80%) and testing (20%) sets ---
# Use random_state for reproducibility
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 7. Train a Linear Regression model on the training data ---
model = LinearRegression()
model.fit(X_train, y_train)

# --- 8. Evaluate the model on the test set and report R-squared (R2) score and Mean Absolute Error (MAE) ---
y_pred = model.predict(X_test)

# Calculate R-squared score
r2 = r2_score(y_test, y_pred)

# Calculate Mean Absolute Error
mae = mean_absolute_error(y_test, y_pred)

print(f"Model Evaluation on Test Set:")
print(f"R-squared (R2) Score: {r2:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
```