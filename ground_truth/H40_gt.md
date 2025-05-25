```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

# --- Configuration ---
DATASET_PATH = 'imdb_top_250.csv'
FEATURES = ['Year', 'Runtime']
TARGET = 'Rating'
TEST_SIZE = 0.2
RANDOM_STATE = 42

# --- 1. Load the dataset ---
try:
    df = pd.read_csv(DATASET_PATH)
    print(f"Dataset '{DATASET_PATH}' loaded successfully. Shape: {df.shape}")
except FileNotFoundError:
    print(f"Error: The file '{DATASET_PATH}' was not found.")
    print("Please ensure the CSV file is in the same directory as the script.")
    exit()
except Exception as e:
    print(f"An error occurred while loading the dataset: {e}")
    exit()

# --- Data Preprocessing ---
# Ensure required columns exist
if not all(col in df.columns for col in FEATURES + [TARGET]):
    missing_cols = [col for col in FEATURES + [TARGET] if col not in df.columns]
    print(f"Error: Missing required columns in the dataset: {missing_cols}")
    exit()

# Clean and convert 'Runtime' column to numeric
# It might be in format 'X min', so remove ' min' and convert to numeric.
# Use errors='coerce' to turn non-convertible values into NaN.
if 'Runtime' in df.columns:
    df['Runtime'] = df['Runtime'].astype(str).str.replace(' min', '', regex=False)
    df['Runtime'] = pd.to_numeric(df['Runtime'], errors='coerce')
else:
    print("Warning: 'Runtime' column not found. Skipping runtime cleaning.")

# Ensure 'Year' and 'Rating' are numeric, coercing errors to NaN
df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce')

# Drop rows with any NaN values in the feature or target columns after cleaning
initial_rows = df.shape[0]
df.dropna(subset=FEATURES + [TARGET], inplace=True)
rows_after_dropna = df.shape[0]
if initial_rows - rows_after_dropna > 0:
    print(f"Dropped {initial_rows - rows_after_dropna} rows due to missing or invalid values in features/target.")

if df.empty:
    print("Error: No valid data remaining after preprocessing. Exiting.")
    exit()

# --- 2. Define features X and target y ---
X = df[FEATURES]
y = df[TARGET]
print(f"Features (X) selected: {FEATURES}")
print(f"Target (y) selected: {TARGET}")

# --- 3. Split the data into training (80%) and testing (20%) sets ---
# Using random_state for reproducibility
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
print(f"Data split into training ({len(X_train)} samples) and testing ({len(X_test)} samples).")

# --- 4. Train a Linear Regression model on the training data ---
model = LinearRegression()
model.fit(X_train, y_train)
print("Linear Regression model trained successfully.")

# --- 5. Make predictions on the test set ---
y_pred = model.predict(X_test)
print("Predictions made on the test set.")

# --- 6. Evaluate the model using R-squared (R2) score and Mean Absolute Error (MAE) ---
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("\n--- Model Evaluation ---")
print(f"R-squared (R2) Score: {r2:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")

# Optional: Display model coefficients
# print("\n--- Model Coefficients ---")
# for feature, coef in zip(FEATURES, model.coef_):
#     print(f"{feature}: {coef:.4f}")
# print(f"Intercept: {model.intercept_:.4f}")
```