```python
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer # Although manual median imputation is used for 0s

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    roc_curve, auc, confusion_matrix, classification_report, ConfusionMatrixDisplay, RocCurveDisplay
)

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

# Set a random seed for reproducibility
np.random.seed(42)

# --- 1. Load the dataset ---
try:
    df = pd.read_csv('diabetes.csv')
    print("Dataset loaded successfully.")
    print(f"Initial dataset shape: {df.shape}")
    print("Initial dataset head:")
    print(df.head())
except FileNotFoundError:
    print("Error: diabetes.csv not found. Please ensure the file is in the current directory.")
    exit()

# --- 2. Preprocess data by handling missing values (zeros replaced with median) ---
# Columns where 0s represent missing values according to the problem description
cols_with_zeros_as_missing = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

# Replace 0s with NaN in specified columns
for col in cols_with_zeros_as_missing:
    df[col] = df[col].replace(0, np.nan)

# Impute missing values (NaNs) with the median of each respective column
for col in cols_with_zeros_as_missing:
    median_val = df[col].median()
    df[col].fillna(median_val, inplace=True)
print("\nMissing values (0s) in specified columns handled by replacing with their respective medians.")
print(f"Dataset shape after imputation: {df.shape}")

# --- 3. Create additional features ---
# Glucose_BMI_ratio
df['Glucose_BMI_ratio'] = df['Glucose'] / df['BMI']
print("\nCreated 'Glucose_BMI_ratio' feature.")

# Age_group (categorical)
# Define age bins and labels for categorization
age_bins = [0, 20, 30, 40, 50, 60, np.inf]
age_labels = ['0-20', '21-30', '31-40', '41-50', '51-60', '60+']
df['Age_group'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels, right=True)
print("Created 'Age_group' (categorical) feature.")

# Insulin_level (low/normal/high based on tertiles)
# pd.qcut is used to divide data into quantiles (tertiles in this case)
df['Insulin_level'] = pd.qcut(df['Insulin'], q=3, labels=['low', 'normal', 'high'])
print("Created 'Insulin_level' (categorical) feature based on tertiles.")

print("\nDataset head after feature engineering:")
print(df.head())

# --- 4. Encode categorical variables and split data into training and testing sets ---
# Define features (X) and target (y)
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Identify numerical and categorical features for preprocessing
numerical_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(include='object').columns.tolist()

# Create a ColumnTransformer for preprocessing:
# - Numerical features will be scaled using StandardScaler.
# - Categorical features will be one-hot encoded using OneHotEncoder.
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough' # Keep any other columns not specified (though all are covered here)
)

# Split data into training and testing sets (80-20 split)
# stratify=y ensures that the proportion of target variable 'Outcome' is the same in both train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"\nData split into training (80%) and testing (20%) sets.")
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

# Apply preprocessing to training and testing data
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Get feature names after one-hot encoding for later use (e.g., feature importance)
ohe_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
all_feature_names = numerical_features + list(ohe_feature_names)

# Convert processed data back to DataFrame for easier handling, especially for feature importance
X_train_processed_df = pd.DataFrame(X_train_processed, columns=all_feature_names)
X_test_processed_df = pd.DataFrame(X_test_processed, columns=all_feature_names)
print("\nCategorical features encoded and numerical features scaled.")
print(f"X_train_processed_df shape: {X_train_processed_df.shape}")
print(f"X_test_processed_df shape: {X_test_processed_df.shape}")


# --- 5. Train and compare multiple classification models ---
models = {
    'Logistic Regression': LogisticRegression(random_state=42, solver='liblinear'), # 'liblinear' is good for small datasets
    'Random Forest': RandomForestClassifier(random_state=42),
    'Support Vector Machine': SVC(random_state=42, probability=True), # probability=True is needed for ROC AUC calculation
    'Gradient Boosting': GradientBoostingClassifier(random_state=42)
}

results = {} # To store test set evaluation metrics
cv_results = {} # To store cross-validation results
roc_curves_data = {} # To store data for plotting ROC curves

print("\n--- Model Training and Cross-Validation ---")
# Define cross-validation strategy
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    print(f"\nTraining and evaluating {name}...")

    # --- 6. Use cross-validation to evaluate each model ---
    # Evaluate using cross_val_score for various metrics on the training data
    cv_accuracy = cross_val_score(model, X_train_processed_df, y_train, cv=cv_strategy, scoring='accuracy', n_jobs=-1)
    cv_precision = cross_val_score(model, X_train_processed_df, y_train, cv=cv_strategy, scoring='precision', n_jobs=-1)
    cv_recall = cross_val_score(model, X_train_processed_df, y_train, cv=cv_strategy, scoring='recall', n_jobs=-1)
    cv_f1 = cross_val_score(model, X_train_processed_df, y_train, cv=cv_strategy, scoring='f1', n_jobs=-1)
    cv_roc_auc = cross_val_score(model, X_train_processed_df, y_train, cv=cv_strategy, scoring='roc_auc', n_jobs=-1)

    cv_results[name] = {
        'Accuracy': np.mean(cv_accuracy),
        'Precision': np.mean(cv_precision),
        'Recall': np.mean(cv_recall),
        'F1-Score': np.mean(cv_f1),
        'ROC-AUC': np.mean(cv_roc_auc)
    }
    print(f"{name} Cross-Validation Results (Mean +/- Std Dev):")
    print(f"  Accuracy: {np.mean(cv_accuracy):.4f} +/- {np.std(cv_accuracy):.4f}")
    print(f"  Precision: {np.mean(cv_precision):.4f} +/- {np.std(cv_precision):.4f}")
    print(f"  Recall: {np.mean(cv_recall):.4f} +/- {np.std(cv_recall):.4f}")
    print(f"  F1-Score: {np.mean(cv_f1):.4f} +/- {np.std(cv_f1):.4f}")
    print(f"  ROC-AUC: {np.mean(cv_roc_auc):.4f} +/- {np.std(cv_roc_auc):.4f}")

    # Train the model on the full training data for final evaluation on the test set
    model.fit(X_train_processed_df, y_train)
    y_pred = model.predict(X_test_processed_df)
    # Get probability scores for ROC AUC calculation
    y_prob = model.predict_proba(X_test_processed_df)[:, 1] if hasattr(model, "predict_proba") else [0] * len(y_test)

    # Store metrics for test set evaluation
    results[name] = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_prob)
    }

    # Store ROC curve data for plotting
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    roc_curves_data[name] = {'fpr': fpr, 'tpr': tpr, 'auc': roc_auc}

print("\n--- Test Set Evaluation Results (Before Hyperparameter Tuning) ---")
for name, metrics in results.items():
    print(f"\n{name}:")
    for metric_name, value in metrics.items():
        print(f"  {metric_name}: {value:.4f}")

# --- 7. Perform hyperparameter tuning for the best model using GridSearchCV ---
# Based on general performance and tunability, Random Forest is chosen for hyperparameter tuning.
print("\n--- Hyperparameter Tuning for Random Forest Classifier ---")
rf_model_for_tuning = RandomForestClassifier(random_state=42)

# Define parameter grid for Random Forest
param_grid_rf = {
    'n_estimators': [100, 200, 300], # Number of trees in the forest
    'max_features': ['sqrt', 'log2'], # Number of features to consider when looking for the best split
    'max_depth': [10, 20, None], # Maximum depth of the tree (None means unlimited)
    'min_samples_split': [2, 5, 10], # Minimum number of samples required to split an internal node
    'min_samples_leaf': [1, 2, 4] # Minimum number of samples required to be at a leaf node
}

# GridSearchCV to find the best hyperparameters
grid_search_rf = GridSearchCV(estimator=rf_model_for_tuning, param_grid=param_grid_rf,
                              cv=cv_strategy, scoring='roc_auc', n_jobs=-1, verbose=1) # Use ROC-AUC as the scoring metric for tuning
grid_search_rf.fit(X_train_processed_df, y_train)

print(f"\nBest parameters for Random Forest: {grid_search_rf.best_params_}")
print(f"Best ROC-AUC score (CV) for Random Forest: {grid_search_rf.best_score_:.4f}")

# Evaluate the best Random Forest model on the test set
best_rf_model = grid_search_rf.best_estimator_
y_pred_best_rf = best_rf_model.predict(X_test_processed_df)
y_prob_best_rf = best_rf_model.predict_proba(X_test_processed_df)[:, 1]

# Update results and ROC curve data for the tuned Random Forest
results['Random Forest (Tuned)'] = {
    'Accuracy': accuracy_score(y_test, y_pred_best_rf),
    'Precision': precision_score(y_test, y_pred_best_rf),
    'Recall': recall_score(y_test, y_pred_best_rf),
    'F1-Score': f1_score(y_test, y_pred_best_rf),
    'ROC-AUC': roc_auc_score(y_test, y_prob_best_rf)
}
fpr_tuned_rf, tpr_tuned_rf, _ = roc_curve(y_test, y_prob_best_rf)
roc_auc_tuned_rf = auc(fpr_tuned_rf, tpr_tuned_rf)
roc_curves_data['Random Forest (Tuned)'] = {'fpr': fpr_tuned_rf, 'tpr': tpr_tuned_rf, 'auc': roc_auc_tuned_rf}

print("\n--- Final Test Set Evaluation Results (Including Tuned Random Forest) ---")
# Sort results by ROC-AUC for better comparison
sorted_results = sorted(results.items(), key=lambda item: item[1]['ROC-AUC'], reverse=True)

for name, metrics in sorted_results:
    print(f"\n{name}:")
    for metric_name, value in metrics.items():
        print(f"  {metric_name}: {value:.4f}")
    # Print classification report for the best model (tuned RF)
    if name == 'Random Forest (Tuned)':
        print(f"  Classification Report:\n{classification_report(y_test, y_pred_best_rf)}")
    # For other models, print their classification report
    elif name in models:
        print(f"  Classification Report:\n{classification_report(y_test, models[name].predict(X_test_processed_df))}")


# --- 8. Create confusion matrices and ROC curves for visualization ---
print("\n--- Visualizations ---")

# Plot Confusion Matrices for all models
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.flatten() # Flatten the 2x2 array of axes for easy iteration

# Models to plot (using the tuned RF model)
models_to_plot = {
    'Logistic Regression': models['Logistic Regression'],
    'Random Forest (Tuned)': best_rf_model,
    'Support Vector Machine': models['Support Vector Machine'],
    'Gradient Boosting': models['Gradient Boosting']
}

for i, (name, model) in enumerate(models_to_plot.items()):
    y_pred = model.predict(X_test_processed_df)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i], cbar=False,
                xticklabels=['Predicted Negative', 'Predicted Positive'],
                yticklabels=['Actual Negative', 'Actual Positive'])
    axes[i].set_title(f'Confusion Matrix: {name}', fontsize=14)
    axes[i].set_xlabel('Predicted Label', fontsize=12)
    axes[i].set_ylabel('True Label', fontsize=12)
plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
plt.suptitle('Confusion Matrices for Different Models', y=0.98, fontsize=18)
plt.show()

# Plot ROC Curves for all models
plt.figure(figsize=(10, 8))
for name, data in roc_curves_data.items():
    plt.plot(data['fpr'], data['tpr'], label=f'{name} (AUC = {data["auc"]:.2f})', linewidth=2)

plt.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.50)', linewidth=1) # Diagonal line for random classifier
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curves for Classification Models', fontsize=16)
plt.legend(loc='lower right', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# --- 9. Display feature importance for tree-based models ---
print("\n--- Feature Importance ---")

# Feature importance for Tuned Random Forest
if hasattr(best_rf_model, 'feature_importances_'):
    rf_importances = pd.DataFrame({
        'Feature': all_feature_names,
        'Importance': best_rf_model.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=rf_importances.head(15), palette='viridis') # Top 15 features
    plt.title('Feature Importance for Tuned Random Forest Classifier', fontsize=16)
    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.show()
    print("\nTop 10 Feature Importances for Tuned Random Forest:")
    print(rf_importances.head(10).to_string())

# Feature importance for Gradient Boosting
gb_model = models['Gradient Boosting']
if hasattr(gb_model, 'feature_importances_'):
    gb_importances = pd.DataFrame({
        'Feature': all_feature_names,
        'Importance': gb_model.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=gb_importances.head(15), palette='magma') # Top 15 features
    plt.title('Feature Importance for Gradient Boosting Classifier', fontsize=16)
    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.show()
    print("\nTop 10 Feature Importances for Gradient Boosting:")
    print(gb_importances.head(10).to_string())

print("\n--- Analysis Complete ---")
```