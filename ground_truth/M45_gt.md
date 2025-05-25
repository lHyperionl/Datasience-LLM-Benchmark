```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# --- 1. Load the dataset ---
# Assume 'penguins.csv' is in the current directory.
try:
    df = pd.read_csv('penguins.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: 'penguins.csv' not found. Please ensure the file is in the current directory.")
    exit() # Exit if the file is not found

# --- 2. Select features X and target y ---
# Define the feature columns as specified.
feature_columns = ['culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g']
# Define the target column.
target_column = 'species'

# Check if all specified columns exist in the DataFrame
missing_cols = [col for col in feature_columns + [target_column] if col not in df.columns]
if missing_cols:
    print(f"Error: The following required columns are missing from the dataset: {missing_cols}")
    exit()

# Create a subset of the DataFrame with only the relevant columns for processing.
df_selected = df[feature_columns + [target_column]].copy()

# --- 3. Drop any rows where 'species' or any of the four selected feature columns have missing values (NA) ---
# Drop rows where any of the selected columns have NA values.
initial_rows = df_selected.shape[0]
df_cleaned = df_selected.dropna()
rows_dropped = initial_rows - df_cleaned.shape[0]
print(f"Dropped {rows_dropped} rows due to missing values in selected columns.")

# Separate features (X) and target (y) from the cleaned DataFrame.
X = df_cleaned[feature_columns]
y = df_cleaned[target_column]

# --- 4. Encode the categorical 'species' target variable into numerical labels ---
# Initialize LabelEncoder.
label_encoder = LabelEncoder()
# Fit LabelEncoder and transform the target variable 'y'.
y_encoded = label_encoder.fit_transform(y)
print(f"Species encoded into numerical labels: {label_encoder.classes_} -> {label_encoder.transform(label_encoder.classes_)}")

# --- 5. Split the cleaned data into training (80%) and testing (20%) sets ---
# Use train_test_split for splitting, with random_state for reproducibility.
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
print(f"Data split into training (X_train: {X_train.shape}, y_train: {y_train.shape}) "
      f"and testing (X_test: {X_test.shape}, y_test: {y_test.shape}) sets.")

# --- 6. Train a Logistic Regression model on the training data ---
# Initialize Logistic Regression model with max_iter=1000 for convergence.
model = LogisticRegression(max_iter=1000)
# Train the model using the training data.
model.fit(X_train, y_train)
print("Logistic Regression model trained successfully.")

# --- 7. Make predictions on the test set and report the accuracy score ---
# Make predictions on the test features.
y_pred = model.predict(X_test)
# Calculate the accuracy score by comparing predicted labels with actual test labels.
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy on the test set: {accuracy:.4f}")
```