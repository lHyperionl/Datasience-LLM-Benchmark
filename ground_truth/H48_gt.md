```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# --- 1. Load the dataset ---
try:
    df = pd.read_csv('penguins.csv')
    print("Dataset loaded successfully.")
    print(f"Initial dataset shape: {df.shape}")
    print("First 5 rows of the dataset:")
    print(df.head())
except FileNotFoundError:
    print("Error: penguins.csv not found. Please ensure the file is in the current directory.")
    exit()

# --- 2. Define features and target ---
# Features to be used for prediction
features = ['culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g', 'island', 'sex']
# Target variable to predict
target = 'species'

# Separate features (X) and target (y)
X = df[features].copy()
y = df[target].copy()

print(f"\nFeatures selected: {features}")
print(f"Target variable: {target}")

# --- 3. Preprocessing ---

# Identify numerical and categorical features for preprocessing
numerical_features = ['culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g']
categorical_features = ['island', 'sex']

# a) For numerical features, impute NAs with the column median.
# b) For 'sex', impute NAs with the mode.
# c) One-hot encode 'island' and 'sex' (after mode imputation for sex).

# Create preprocessing pipelines for numerical and categorical features
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')) # Impute NAs with median
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')), # Impute NAs with mode for categorical
    ('onehot', OneHotEncoder(handle_unknown='ignore')) # One-hot encode
])

# Create a column transformer to apply different transformations to different columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# d) Ensure 'species' is label encoded if it's not already numeric.
# Fit LabelEncoder on the entire target variable 'y'
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
print(f"\nOriginal species labels: {label_encoder.classes_}")
print(f"Encoded species labels (first 5): {y_encoded[:5]}")

# --- 4. Split the data into training (75%) and testing (25%) sets ---
# Ensure stratification by 'species' and use a random_state for reproducibility.
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.25, random_state=42, stratify=y_encoded
)

print(f"\nData split into training and testing sets:")
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

# --- 5. Train a RandomForestClassifier ---
# Combine preprocessing and classifier into a single pipeline
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

print("\nTraining RandomForestClassifier...")
model_pipeline.fit(X_train, y_train)
print("Model training complete.")

# --- 6. Make predictions on the test set ---
y_pred = model_pipeline.predict(X_test)
print("\nPredictions made on the test set.")

# --- 7. Print the accuracy score and a classification report ---
# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.4f}")

# Generate classification report
# We need to pass the original class names to classification_report for better readability
class_names = label_encoder.classes_
report = classification_report(y_test, y_pred, target_names=class_names)
print("\nClassification Report:")
print(report)

print("\n--- Script Finished ---")
```