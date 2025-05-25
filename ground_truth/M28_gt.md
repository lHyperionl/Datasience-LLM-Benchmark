```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# --- 1. Load the dataset ---
# Assume 'heart.csv' is in the current directory
try:
    df = pd.read_csv('heart.csv')
except FileNotFoundError:
    print("Error: 'heart.csv' not found. Please ensure the file is in the current directory.")
    exit()

# --- 2. Select features and target variable ---
# Define the features to be used for training
features = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Sex', 'ChestPainType']
# Define the target variable
target = 'HeartDisease'

# Separate features (X) and target (y)
X = df[features]
y = df[target]

# --- 3. Preprocess categorical features using one-hot encoding ---
# Identify categorical columns that need one-hot encoding
categorical_features = ['Sex', 'ChestPainType']

# Create a ColumnTransformer to apply OneHotEncoder to specified categorical columns
# 'remainder="passthrough"' ensures that numerical columns not specified are kept
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough'
)

# --- 4. Split the preprocessed data into training (75%) and testing (25%) sets ---
# Use random_state for reproducibility
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# --- 5. Train a Logistic Regression model ---
# Create a pipeline that first preprocesses the data and then applies Logistic Regression
# This ensures that preprocessing steps are applied consistently to both training and test data
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('classifier', LogisticRegression(max_iter=1000))]) # Increased max_iter for convergence

# Train the model on the training data
model.fit(X_train, y_train)

# --- 6. Make predictions on the test set and report the accuracy score ---
# Make predictions on the preprocessed test set
y_pred = model.predict(X_test)

# Calculate the accuracy score
accuracy = accuracy_score(y_test, y_pred)

# Report the accuracy
print(f"Model Accuracy on Test Set: {accuracy:.4f}")
```