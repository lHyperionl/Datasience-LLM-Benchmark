```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay
)

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

# --- 1. Load Data and Initial Preprocessing ---
print("--- 1. Loading Data and Initial Preprocessing ---")
try:
    df = pd.read_csv('water_potability.csv')
    print("Dataset loaded successfully.")
    print(f"Dataset shape: {df.shape}")
    print("Missing values before imputation:\n", df.isnull().sum())
except FileNotFoundError:
    print("Error: 'water_potability.csv' not found. Please ensure the file is in the current directory.")
    exit()

# Separate features (X) and target (y)
X = df.drop('Potability', axis=1)
y = df['Potability']

# --- 2. Data Preprocessing ---

# Handle missing values using median imputation
# Create an imputer object with a median strategy
imputer = SimpleImputer(strategy='median')

# Fit the imputer on the training data and transform both train and test sets
# (This will be done after train-test split to prevent data leakage)

# Train-test split (80/20 ratio, stratified to maintain class distribution)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain set shape: {X_train.shape}, Test set shape: {X_test.shape}")
print(f"Train target distribution:\n{y_train.value_counts(normalize=True)}")
print(f"Test target distribution:\n{y_test.value_counts(normalize=True)}")

# Apply imputation
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Convert back to DataFrame for easier handling and column names
X_train_imputed = pd.DataFrame(X_train_imputed, columns=X_train.columns, index=X_train.index)
X_test_imputed = pd.DataFrame(X_test_imputed, columns=X_test.columns, index=X_test.index)

print("\nMissing values after imputation (train set):\n", X_train_imputed.isnull().sum().sum())
print("Missing values after imputation (test set):\n", X_test_imputed.isnull().sum().sum())

# Feature Scaling using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

# Convert scaled arrays back to DataFrames
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

print("\nData preprocessing complete: Missing values handled and features scaled.")

# --- 3. Train Multiple Classification Algorithms ---
# --- 4. Evaluate Models with Comprehensive Metrics, Confusion Matrices, and ROC Curves ---

models = {
    'Logistic Regression': LogisticRegression(random_state=42, solver='liblinear'),
    'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
    'Support Vector Machine': SVC(random_state=42, probability=True) # probability=True for ROC-AUC
}

results = {}
confusion_matrices = {}
roc_curves = {}

print("\n--- 3 & 4. Training and Evaluating Models ---")

plt.figure(figsize=(18, 6))
roc_ax = plt.subplot(1, 3, 1) # Placeholder for combined ROC plot
plt.title('ROC Curves for All Models')
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess') # Baseline for ROC

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1] # Probability of positive class

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    results[name] = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'ROC-AUC': roc_auc
    }

    print(f"{name} Metrics:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(f"  ROC-AUC: {roc_auc:.4f}")

    # Create Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    confusion_matrices[name] = cm
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Not Potable', 'Potable'],
                yticklabels=['Not Potable', 'Potable'])
    plt.title(f'Confusion Matrix for {name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()

    # Create ROC Curve
    RocCurveDisplay.from_estimator(model, X_test_scaled, y_test, ax=roc_ax, name=name)

roc_ax.set_xlabel('False Positive Rate')
roc_ax.set_ylabel('True Positive Rate')
roc_ax.legend(loc='lower right')
plt.tight_layout()
plt.show()

# --- 5. Perform k-fold Cross-Validation (k=5) ---
print("\n--- 5. Performing k-fold Cross-Validation (k=5) ---")

cv_results = {}
cv_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    print(f"Performing 5-fold cross-validation for {name}...")
    # For cross_val_score, we need to apply imputation and scaling within the CV loop
    # or use a pipeline. For simplicity, we'll use the preprocessed data (X_scaled, y)
    # and assume the scaling/imputation process is robust.
    # A more robust approach would be to build a pipeline:
    # pipeline = Pipeline([('imputer', imputer), ('scaler', scaler), ('model', model)])
    # However, the prompt implies separate steps.
    # So, we'll use the full dataset X, y and let cross_val_score handle splitting,
    # but we need to apply the preprocessing steps inside a custom scoring function
    # or use a pipeline. Let's use a pipeline for proper CV.

    from sklearn.pipeline import Pipeline
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('classifier', model)
    ])

    # Evaluate using multiple scoring metrics
    scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    scores = cross_val_score(pipeline, X, y, cv=cv_splitter, scoring='accuracy', n_jobs=-1)
    precision_scores = cross_val_score(pipeline, X, y, cv=cv_splitter, scoring='precision', n_jobs=-1)
    recall_scores = cross_val_score(pipeline, X, y, cv=cv_splitter, scoring='recall', n_jobs=-1)
    f1_scores = cross_val_score(pipeline, X, y, cv=cv_splitter, scoring='f1', n_jobs=-1)
    roc_auc_scores = cross_val_score(pipeline, X, y, cv=cv_splitter, scoring='roc_auc', n_jobs=-1)

    cv_results[name] = {
        'Accuracy_mean': scores.mean(), 'Accuracy_std': scores.std(),
        'Precision_mean': precision_scores.mean(), 'Precision_std': precision_scores.std(),
        'Recall_mean': recall_scores.mean(), 'Recall_std': recall_scores.std(),
        'F1-Score_mean': f1_scores.mean(), 'F1-Score_std': f1_scores.std(),
        'ROC-AUC_mean': roc_auc_scores.mean(), 'ROC-AUC_std': roc_auc_scores.std()
    }

    print(f"  {name} CV Accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")
    print(f"  {name} CV F1-Score: {f1_scores.mean():.4f} (+/- {f1_scores.std():.4f})")
    print(f"  {name} CV ROC-AUC: {roc_auc_scores.mean():.4f} (+/- {roc_auc_scores.std():.4f})")

# --- 6. Compare Model Performances and Select the Best Performing Model ---
print("\n--- 6. Comparing Model Performances ---")

# Convert results to DataFrame for easy comparison
df_results = pd.DataFrame(results).T
print("\nTest Set Performance Metrics:")
print(df_results.round(4))

df_cv_results = pd.DataFrame(cv_results).T
print("\nCross-Validation Performance Metrics (Mean +/- Std):")
print(df_cv_results.round(4))

# Select the best model based on F1-Score from cross-validation results (more robust)
# F1-score is a good balance between precision and recall, suitable for imbalanced datasets.
best_model_name = df_cv_results['F1-Score_mean'].idxmax()
best_model = models[best_model_name]

print(f"\nBased on Cross-Validation F1-Score, the best model is: {best_model_name}")
print(f"Best model's CV F1-Score: {df_cv_results.loc[best_model_name, 'F1-Score_mean']:.4f}")

# Re-train the best model on the full preprocessed training data for feature importance
# (The model object in `models` dictionary is already trained on X_train_scaled, y_train)

# --- 7. Display Feature Importance for the Best Model ---
print(f"\n--- 7. Displaying Feature Importance for the Best Model ({best_model_name}) ---")

if best_model_name == 'Random Forest':
    feature_importances = best_model.feature_importances_
    features_df = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=False)
    print("\nFeature Importances (Random Forest):")
    print(features_df)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=features_df)
    plt.title(f'Feature Importances for {best_model_name}')
    plt.tight_layout()
    plt.show()

elif best_model_name == 'Logistic Regression':
    # For Logistic Regression, coefficients indicate importance (magnitude)
    # Need to use the trained model's coefficients
    coefficients = best_model.coef_[0]
    features_df = pd.DataFrame({
        'Feature': X_train.columns,
        'Coefficient': coefficients,
        'Abs_Coefficient': np.abs(coefficients)
    }).sort_values(by='Abs_Coefficient', ascending=False)
    print("\nFeature Coefficients (Logistic Regression):")
    print(features_df[['Feature', 'Coefficient']])

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Abs_Coefficient', y='Feature', data=features_df)
    plt.title(f'Absolute Feature Coefficients for {best_model_name}')
    plt.tight_layout()
    plt.show()

elif best_model_name == 'Support Vector Machine':
    print("\nSupport Vector Machine (SVC) does not directly provide feature importances like tree-based models or coefficients like linear models.")
    print("Feature importance for SVM can be derived using methods like permutation importance, which is beyond the scope of direct attribute display.")

print("\n--- Analysis Complete ---")
```