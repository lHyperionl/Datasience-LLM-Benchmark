```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder # Although pd.get_dummies is used, this import is often expected for encoding tasks.

# --- 1. Load the dataset ---
# Assume 'clash_royale_cards.csv' is in the current directory.
try:
    df = pd.read_csv('clash_royale_cards.csv')
except FileNotFoundError:
    print("Error: 'clash_royale_cards.csv' not found. Please ensure the file is in the current directory.")
    exit()

# --- 2. Create a new binary target column named `is_high_win_rate` ---
# This column should be 1 if `Win Rate` is greater than 55, and 0 otherwise.
df['is_high_win_rate'] = (df['Win Rate'] > 55).astype(int)

# --- 3. Select the specified features and the target column ---
# Features to be used for the model.
features = ['elixirCost', 'Usage', 'rarity']
target = 'is_high_win_rate'

# Create feature matrix X and target vector y
X = df[features].copy() # .copy() to avoid SettingWithCopyWarning
y = df[target]

# --- 4. Apply one-hot encoding to the categorical `rarity` column ---
# pandas get_dummies is a convenient way to perform one-hot encoding directly on a DataFrame.
# It will create new columns for each category in 'rarity'.
X = pd.get_dummies(X, columns=['rarity'], drop_first=False)

# --- 5. Split the data into training and testing sets ---
# Use an 80/20 split and a random_state of 42 for reproducibility.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# The data is now prepared:
# X_train: Training features (one-hot encoded 'rarity')
# X_test: Testing features (one-hot encoded 'rarity')
# y_train: Training target (is_high_win_rate)
# y_test: Testing target (is_high_win_rate)

# Optional: Print shapes to verify the split
# print("Shape of X_train:", X_train.shape)
# print("Shape of X_test:", X_test.shape)
# print("Shape of y_train:", y_train.shape)
# print("Shape of y_test:", y_test.shape)
# print("\nSample of X_train (first 5 rows):\n", X_train.head())
# print("\nSample of y_train (first 5 rows):\n", y_train.head())
```