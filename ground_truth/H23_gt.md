```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor # Choosing RandomForestRegressor as an example
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import OneHotEncoder # Although pd.get_dummies is used, including for completeness as per prompt hint

# --- 1. Load the dataset ---
# Assume 'clash_royale_cards.csv' is in the current directory
try:
    df = pd.read_csv('clash_royale_cards.csv')
except FileNotFoundError:
    print("Error: 'clash_royale_cards.csv' not found. Please ensure the file is in the current directory.")
    exit()

# --- 2. Select features and target ---
# Define the features (X) and the target (y) columns
features = ['elixirCost', 'Usage', 'rarity']
target = 'Win Rate'

# Check if all required columns exist in the DataFrame
if not all(col in df.columns for col in features + [target]):
    missing_cols = [col for col in features + [target] if col not in df.columns]
    print(f"Error: Missing required columns in the dataset: {missing_cols}")
    exit()

X = df[features]
y = df[target]

# Handle potential missing values in selected columns (simple drop for demonstration)
# In a real-world scenario, more sophisticated imputation might be needed.
initial_rows = X.shape[0]
X.dropna(inplace=True)
y = y.loc[X.index] # Align target with cleaned features
if X.shape[0] < initial_rows:
    print(f"Warning: Dropped {initial_rows - X.shape[0]} rows due to missing values in features or target.")

# --- 3. Preprocess the 'rarity' column using one-hot encoding ---
# pd.get_dummies is a convenient way to perform one-hot encoding directly on a DataFrame
# drop_first=True avoids multicollinearity by dropping one category per feature
X = pd.get_dummies(X, columns=['rarity'], drop_first=True)

# --- 4. Split the data into training (80%) and testing (20%) sets ---
# Use random_state for reproducibility of the split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 5. Train a regression model (RandomForestRegressor chosen) ---
# Initialize the RandomForestRegressor model
# Set random_state for reproducibility of the model training
model = RandomForestRegressor(random_state=42)

# Train the model on the training data
print("Training the RandomForestRegressor model...")
model.fit(X_train, y_train)
print("Model training complete.")

# --- 6. Evaluate the model on the test set ---
# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate R-squared (R2) score
r2 = r2_score(y_test, y_pred)

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)

# --- 7. Print evaluation results ---
print("\n--- Model Evaluation Results ---")
print(f"R-squared (R2) Score: {r2:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print("------------------------------")
```