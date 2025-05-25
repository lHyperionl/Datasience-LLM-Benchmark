```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# --- 1. Load the dataset ---
try:
    df = pd.read_csv('spotify_2023.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: 'spotify_2023.csv' not found. Please ensure the CSV file is in the current directory.")
    exit() # Exit if the file is not found

# --- 2. Ensure 'streams' column is numeric and handle NaNs ---
# Coerce non-numeric values in 'streams' to NaN
df['streams'] = pd.to_numeric(df['streams'], errors='coerce')

# Define the columns that will be used for the model
# This helps in dropping NaNs only from relevant rows for the model
model_columns = ['streams', 'in_spotify_playlists', 'energy_%']

# Drop rows where 'streams' or any of the predictor columns are NaN
# This handles the requirement to drop NaNs in 'streams' or predictor columns
initial_rows = df.shape[0]
df_cleaned = df.dropna(subset=model_columns)
rows_dropped = initial_rows - df_cleaned.shape[0]
print(f"Dropped {rows_dropped} rows containing NaN values in 'streams', 'in_spotify_playlists', or 'energy_%'.")

# --- 3. Select features (X) and target (y) ---
X = df_cleaned[['in_spotify_playlists', 'energy_%']]
y = df_cleaned['streams']

print(f"Features (X) shape: {X.shape}")
print(f"Target (y) shape: {y.shape}")

# --- 4. Split the data into training and testing sets ---
# 80% training, 20% testing, with a fixed random state for reproducibility
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training data size: {X_train.shape[0]} samples")
print(f"Testing data size: {X_test.shape[0]} samples")

# --- 5. Train a Linear Regression model ---
model = LinearRegression()
model.fit(X_train, y_train)
print("\nLinear Regression model trained successfully.")

# --- 6. Evaluate the model on the test set ---
y_pred = model.predict(X_test)

# Calculate R-squared (R2) score
r2 = r2_score(y_test, y_pred)

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)

print("\n--- Model Evaluation ---")
print(f"R-squared (R2) Score: {r2:.4f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")

```