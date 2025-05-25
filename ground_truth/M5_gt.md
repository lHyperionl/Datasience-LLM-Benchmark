```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay
)

# Set random seed for reproducibility
np.random.seed(42)

# --- 1. Load Data ---
try:
    df = pd.read_csv('titanic.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: titanic.csv not found. Please ensure the file is in the current directory.")
    exit()

# --- 2. Data Preprocessing ---

# Drop irrelevant columns
# PassengerId, Name, Ticket are unique identifiers or non-predictive strings.
# Cabin has too many missing values and might be too granular.
df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

# Handle missing values
# 'Age': Impute with median as it's less sensitive to outliers than mean.
df['Age'].fillna(df['Age'].median(), inplace=True)
# 'Embarked': Impute with mode as it's a categorical variable.
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Feature Engineering: Create FamilySize
# Combine SibSp (siblings/spouses) and Parch (parents/children) into a single feature.
# Add 1 for the individual themselves.
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
# Drop original SibSp and Parch columns
df.drop(['SibSp', 'Parch'], axis=1, inplace=True)

# Define target and features
X = df.drop('Survived', axis=1)
y = df['Survived']

# Identify numerical and categorical features for preprocessing
# Pclass is treated as categorical because its numerical value doesn't imply a linear relationship with survival.
numerical_features = ['Age', 'Fare', 'FamilySize']
categorical_features = ['Pclass', 'Sex', 'Embarked']

# Create preprocessing pipelines for numerical and categorical features
# Numerical features: Scale using StandardScaler
numerical_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

# Categorical features: One-hot encode
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Create a preprocessor using ColumnTransformer
# This applies different transformers to different columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# --- 3. Split Data into Training and Testing Sets (80/20) ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nTraining set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")

# --- 4. Train a Logistic Regression Model ---
# Create a pipeline that first preprocesses the data and then trains the Logistic Regression model
model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                 ('classifier', LogisticRegression(solver='liblinear', random_state=42))])

# Train the model
model_pipeline.fit(X_train, y_train)
print("\nLogistic Regression model trained successfully.")

# --- 5. Evaluate the Model ---
y_pred = model_pipeline.predict(X_test)
y_pred_proba = model_pipeline.predict_proba(X_test)[:, 1] # Probability of survival

print("\n--- Model Evaluation Metrics ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall: {recall_score(y_test, y_pred):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")

# --- 6. Display Feature Coefficients and their Interpretation ---
# Access the trained logistic regression model from the pipeline
logistic_model = model_pipeline.named_steps['classifier']

# Get feature names after one-hot encoding
# First, fit the preprocessor on the training data to get the transformed feature names
model_pipeline.named_steps['preprocessor'].fit(X_train)
encoded_feature_names = model_pipeline.named_steps['preprocessor'].get_feature_names_out()

# Create a DataFrame for coefficients
coefficients_df = pd.DataFrame({
    'Feature': encoded_feature_names,
    'Coefficient': logistic_model.coef_[0]
})

# Sort by absolute coefficient value to see most impactful features
coefficients_df['Abs_Coefficient'] = np.abs(coefficients_df['Coefficient'])
coefficients_df = coefficients_df.sort_values(by='Abs_Coefficient', ascending=False).drop('Abs_Coefficient', axis=1)

print("\n--- Feature Coefficients and Interpretation ---")
print(coefficients_df)
print("\nInterpretation:")
print("- A positive coefficient indicates that as the feature value increases, the log-odds of survival increase (higher probability of survival).")
print("- A negative coefficient indicates that as the feature value increases, the log-odds of survival decrease (lower probability of survival).")
print("- The magnitude of the coefficient indicates the strength of the relationship.")
print("- For one-hot encoded features (e.g., 'cat__Sex_male'), the coefficient compares that category to the omitted baseline category (e.g., 'Sex_female').")

# --- 7. Create Confusion Matrix and ROC Curve Visualization ---

# Confusion Matrix
plt.figure(figsize=(8, 6))
cm_display = ConfusionMatrixDisplay.from_estimator(model_pipeline, X_test, y_test, cmap='Blues', values_format='d')
cm_display.ax_.set_title('Confusion Matrix')
plt.show()

# ROC Curve
plt.figure(figsize=(8, 6))
roc_display = RocCurveDisplay.from_estimator(model_pipeline, X_test, y_test)
roc_display.ax_.set_title('ROC Curve')
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier') # Plot random classifier line
plt.legend()
plt.show()

# --- 8. Perform Cross-Validation to Assess Model Stability ---
print("\n--- Cross-Validation Results ---")

# Define scoring metrics for cross-validation
scoring = {
    'accuracy': 'accuracy',
    'precision': 'precision',
    'recall': 'recall',
    'f1': 'f1',
    'roc_auc': 'roc_auc'
}

# Perform 5-fold cross-validation
cv_results = cross_validate(model_pipeline, X, y, cv=5, scoring=scoring, return_train_score=False)

# Display mean and standard deviation of each metric
for metric_name, scores in cv_results.items():
    if metric_name.startswith('test_'):
        print(f"{metric_name.replace('test_', '').replace('_', ' ').title()}: {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")

print("\nCross-validation provides a more robust estimate of model performance by averaging results over multiple train-test splits.")
print("The standard deviation indicates the variability of the model's performance across different folds.")
```