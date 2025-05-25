```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# --- 1. Load the dataset ---
try:
    df = pd.read_csv('heart.csv')
except FileNotFoundError:
    print("Error: 'heart.csv' not found. Please ensure the file is in the current directory.")
    exit()

# --- 2. Define features X and target y ---
# Define the target variable
target = 'HeartDisease'
y = df[target]

# Define the feature columns as specified in the prompt
features = [
    'Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS',
    'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope'
]
X = df[features]

# --- 3. & 4. Preprocessing: One-hot encode categorical and scale numerical features ---

# Identify categorical and numerical features based on the prompt and remaining features
categorical_features = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
numerical_features = [col for col in features if col not in categorical_features]

# Create a preprocessor using ColumnTransformer
# 'num' applies StandardScaler to numerical features
# 'cat' applies OneHotEncoder to categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# --- 5. Split the preprocessed data into training and testing sets ---
# Split data before applying the preprocessor to avoid data leakage from test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y # Stratify to maintain class distribution
)

# --- 6. Train a RandomForestClassifier model ---
# Create a pipeline that first preprocesses the data and then trains the classifier
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train the model
print("Training RandomForestClassifier model...")
model_pipeline.fit(X_train, y_train)
print("Model training complete.")

# --- 7. Evaluate the model on the test set ---
print("\nEvaluating model performance on the test set...")

# Make predictions on the test set
y_pred = model_pipeline.predict(X_test)

# Get prediction probabilities for ROC-AUC
y_proba = model_pipeline.predict_proba(X_test)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision_macro = precision_score(y_test, y_pred, average='macro')
recall_macro = recall_score(y_test, y_pred, average='macro')
f1_macro = f1_score(y_test, y_pred, average='macro')
roc_auc = roc_auc_score(y_test, y_proba[:, 1]) # ROC-AUC for the positive class (class 1)

# Print evaluation results
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision (Macro Average): {precision_macro:.4f}")
print(f"Recall (Macro Average): {recall_macro:.4f}")
print(f"F1-Score (Macro Average): {f1_macro:.4f}")
print(f"ROC-AUC Score: {roc_auc:.4f}")

print("\nClassification task complete.")
```