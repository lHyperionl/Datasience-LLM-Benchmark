```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, VotingRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Set random seed for reproducibility
np.random.seed(42)

# --- 1. Load and preprocess the dataset ---
print("--- 1. Loading and Preprocessing Data ---")
try:
    df = pd.read_csv('video_games_sales.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: 'video_games_sales.csv' not found. Please ensure the file is in the current directory.")
    exit()

# Drop irrelevant columns
df = df.drop(columns=['Name', 'Rank'], errors='ignore')

# Convert 'User_Score' to numeric, coercing errors (e.g., 'tbd' will become NaN)
df['User_Score'] = pd.to_numeric(df['User_Score'], errors='coerce')

# Define target variable
target = 'Global_Sales'
X = df.drop(columns=[target])
y = df[target]

# Identify categorical and numerical features
categorical_features = X.select_dtypes(include=['object']).columns
numerical_features = X.select_dtypes(include=np.number).columns

# Create preprocessing pipelines for numerical and categorical features
# Numerical Imputer: Median for robustness against outliers
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])

# Categorical Imputer: Most frequent (mode)
# One-Hot Encoder: Handles new categories during transform by ignoring them (handle_unknown='ignore')
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Create a preprocessor using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough' # Keep other columns (if any) - though none expected here
)

# --- 2. Create a regression target to predict Global_Sales (already done above) ---
# X and y are defined.

# --- 3. Implement and compare multiple ensemble methods ---
# --- 4. Use cross-validation with 5 folds to evaluate each model ---
print("\n--- 3 & 4. Implementing and Evaluating Ensemble Models with 5-Fold Cross-Validation ---")

# Define models
models = {
    'RandomForest': RandomForestRegressor(random_state=42),
    'GradientBoosting': GradientBoostingRegressor(random_state=42),
    'XGBoost': XGBRegressor(random_state=42, n_jobs=-1), # n_jobs=-1 uses all available cores
    'AdaBoost': AdaBoostRegressor(random_state=42)
}

# Store results
results = {}
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    print(f"\nEvaluating {name}...")
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('regressor', model)])

    mae_scores = []
    mse_scores = []
    rmse_scores = []
    r2_scores = []

    fold_num = 1
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        mae_scores.append(mean_absolute_error(y_test, y_pred))
        mse_scores.append(mean_squared_error(y_test, y_pred))
        rmse_scores.append(np.sqrt(mean_squared_error(y_test, y_pred)))
        r2_scores.append(r2_score(y_test, y_pred))
        print(f"  Fold {fold_num}: MAE={mae_scores[-1]:.3f}, RMSE={rmse_scores[-1]:.3f}, R2={r2_scores[-1]:.3f}")
        fold_num += 1

    results[name] = {
        'MAE': np.mean(mae_scores),
        'MSE': np.mean(mse_scores),
        'RMSE': np.mean(rmse_scores),
        'R2': np.mean(r2_scores)
    }
    print(f"  {name} - Avg MAE: {results[name]['MAE']:.3f}, Avg RMSE: {results[name]['RMSE']:.3f}, Avg R2: {results[name]['R2']:.3f}")

# Display all individual model results
print("\n--- Individual Model Cross-Validation Results ---")
for name, metrics in results.items():
    print(f"{name}:")
    for metric_name, value in metrics.items():
        print(f"  {metric_name}: {value:.3f}")

# --- 5. Implement a voting ensemble that combines all four models ---
print("\n--- 5. Implementing Voting Ensemble ---")

# Re-initialize models for VotingRegressor (they need to be untransformed estimators)
rf_base = RandomForestRegressor(random_state=42)
gb_base = GradientBoostingRegressor(random_state=42)
xgb_base = XGBRegressor(random_state=42, n_jobs=-1)
ada_base = AdaBoostRegressor(random_state=42)

voting_regressor = VotingRegressor(estimators=[
    ('rf', rf_base),
    ('gb', gb_base),
    ('xgb', xgb_base),
    ('ada', ada_base)
], n_jobs=-1)

voting_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('regressor', voting_regressor)])

print("Evaluating Voting Ensemble...")
voting_mae_scores = []
voting_mse_scores = []
voting_rmse_scores = []
voting_r2_scores = []

fold_num = 1
for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    voting_pipeline.fit(X_train, y_train)
    y_pred_voting = voting_pipeline.predict(X_test)

    voting_mae_scores.append(mean_absolute_error(y_test, y_pred_voting))
    voting_mse_scores.append(mean_squared_error(y_test, y_pred_voting))
    voting_rmse_scores.append(np.sqrt(mean_squared_error(y_test, y_pred_voting)))
    voting_r2_scores.append(r2_score(y_test, y_pred_voting))
    print(f"  Fold {fold_num}: MAE={voting_mae_scores[-1]:.3f}, RMSE={voting_rmse_scores[-1]:.3f}, R2={voting_r2_scores[-1]:.3f}")
    fold_num += 1

results['VotingEnsemble'] = {
    'MAE': np.mean(voting_mae_scores),
    'MSE': np.mean(voting_mse_scores),
    'RMSE': np.mean(voting_rmse_scores),
    'R2': np.mean(voting_r2_scores)
}
print(f"  Voting Ensemble - Avg MAE: {results['VotingEnsemble']['MAE']:.3f}, Avg RMSE: {results['VotingEnsemble']['RMSE']:.3f}, Avg R2: {results['VotingEnsemble']['R2']:.3f}")


# --- 6. Perform hyperparameter tuning using GridSearchCV for the best individual model ---
print("\n--- 6. Hyperparameter Tuning for the Best Individual Model ---")

# Determine the best individual model based on average MAE
best_model_name = min(results, key=lambda k: results[k]['MAE'] if k not in ['VotingEnsemble', 'StackingEnsemble'] else float('inf'))
best_model_initial_mae = results[best_model_name]['MAE']
print(f"Best individual model (by MAE): {best_model_name} (MAE: {best_model_initial_mae:.3f})")

# Define parameter grid for the best model
param_grid = {}
if best_model_name == 'RandomForest':
    best_model_estimator = RandomForestRegressor(random_state=42)
    param_grid = {
        'regressor__n_estimators': [100, 200],
        'regressor__max_depth': [10, 20, None],
        'regressor__min_samples_split': [2, 5]
    }
elif best_model_name == 'GradientBoosting':
    best_model_estimator = GradientBoostingRegressor(random_state=42)
    param_grid = {
        'regressor__n_estimators': [100, 200],
        'regressor__learning_rate': [0.05, 0.1],
        'regressor__max_depth': [3, 5]
    }
elif best_model_name == 'XGBoost':
    best_model_estimator = XGBRegressor(random_state=42, n_jobs=-1)
    param_grid = {
        'regressor__n_estimators': [100, 200],
        'regressor__learning_rate': [0.05, 0.1],
        'regressor__max_depth': [3, 5]
    }
elif best_model_name == 'AdaBoost':
    best_model_estimator = AdaBoostRegressor(random_state=42)
    param_grid = {
        'regressor__n_estimators': [50, 100],
        'regressor__learning_rate': [0.01, 0.1, 1.0]
    }
else:
    print("No specific tuning grid defined for the identified best model. Skipping tuning.")
    best_tuned_model = None # Placeholder
    tuned_model_name = f"{best_model_name}_Tuned" # Placeholder
    results[tuned_model_name] = results[best_model_name] # Copy original results
    print("Tuning skipped.")

if param_grid:
    tuning_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                      ('regressor', best_model_estimator)])

    grid_search = GridSearchCV(tuning_pipeline, param_grid, cv=kf,
                               scoring='neg_mean_absolute_error', n_jobs=-1, verbose=1)
    grid_search.fit(X, y)

    best_tuned_model = grid_search.best_estimator_
    tuned_model_name = f"{best_model_name}_Tuned"

    print(f"\nBest parameters for {best_model_name}: {grid_search.best_params_}")
    print(f"Best MAE (from GridSearchCV): {-grid_search.best_score_:.3f}")

    # Evaluate the best tuned model using cross-validation
    print(f"\nEvaluating Tuned {best_model_name} with 5-Fold Cross-Validation...")
    tuned_mae_scores = []
    tuned_mse_scores = []
    tuned_rmse_scores = []
    tuned_r2_scores = []

    fold_num = 1
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        best_tuned_model.fit(X_train, y_train) # Fit the entire pipeline
        y_pred_tuned = best_tuned_model.predict(X_test)

        tuned_mae_scores.append(mean_absolute_error(y_test, y_pred_tuned))
        tuned_mse_scores.append(mean_squared_error(y_test, y_pred_tuned))
        tuned_rmse_scores.append(np.sqrt(mean_squared_error(y_test, y_pred_tuned)))
        tuned_r2_scores.append(r2_score(y_test, y_pred_tuned))
        print(f"  Fold {fold_num}: MAE={tuned_mae_scores[-1]:.3f}, RMSE={tuned_rmse_scores[-1]:.3f}, R2={tuned_r2_scores[-1]:.3f}")
        fold_num += 1

    results[tuned_model_name] = {
        'MAE': np.mean(tuned_mae_scores),
        'MSE': np.mean(tuned_mse_scores),
        'RMSE': np.mean(tuned_rmse_scores),
        'R2': np.mean(tuned_r2_scores)
    }
    print(f"  {tuned_model_name} - Avg MAE: {results[tuned_model_name]['MAE']:.3f}, Avg RMSE: {results[tuned_model_name]['RMSE']:.3f}, Avg R2: {results[tuned_model_name]['R2']:.3f}")


# --- 9. Implement stacking ensemble with a meta-learner (Linear Regression) ---
print("\n--- 9. Implementing Stacking Ensemble ---")

# Base estimators for stacking (using the untransformed estimators)
# It's good practice to use slightly different parameters or untuned versions for base estimators
# if they are also part of the tuning process, to avoid overfitting the meta-learner.
# Here, we'll use the default untuned versions for simplicity.
stack_estimators = [
    ('rf', RandomForestRegressor(random_state=42)),
    ('gb', GradientBoostingRegressor(random_state=42)),
    ('xgb', XGBRegressor(random_state=42, n_jobs=-1)),
    ('ada', AdaBoostRegressor(random_state=42))
]

# Meta-learner
meta_learner = LinearRegression()

stacking_regressor = StackingRegressor(
    estimators=stack_estimators,
    final_estimator=meta_learner,
    cv=kf, # Use the same KFold object for consistency
    n_jobs=-1
)

stacking_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                    ('regressor', stacking_regressor)])

print("Evaluating Stacking Ensemble...")
stacking_mae_scores = []
stacking_mse_scores = []
stacking_rmse_scores = []
stacking_r2_scores = []

fold_num = 1
for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    stacking_pipeline.fit(X_train, y_train)
    y_pred_stacking = stacking_pipeline.predict(X_test)

    stacking_mae_scores.append(mean_absolute_error(y_test, y_pred_stacking))
    stacking_mse_scores.append(mean_squared_error(y_test, y_pred_stacking))
    stacking_rmse_scores.append(np.sqrt(mean_squared_error(y_test, y_pred_stacking)))
    stacking_r2_scores.append(r2_score(y_test, y_pred_stacking))
    print(f"  Fold {fold_num}: MAE={stacking_mae_scores[-1]:.3f}, RMSE={stacking_rmse_scores[-1]:.3f}, R2={stacking_r2_scores[-1]:.3f}")
    fold_num += 1

results['StackingEnsemble'] = {
    'MAE': np.mean(stacking_mae_scores),
    'MSE': np.mean(stacking_mse_scores),
    'RMSE': np.mean(stacking_rmse_scores),
    'R2': np.mean(stacking_r2_scores)
}
print(f"  Stacking Ensemble - Avg MAE: {results['StackingEnsemble']['MAE']:.3f}, Avg RMSE: {results['StackingEnsemble']['RMSE']:.3f}, Avg R2: {results['StackingEnsemble']['R2']:.3f}")


# --- 8. Evaluate models using MAE, MSE, RMSE, and R² score (Summary) ---
print("\n--- 8. Comprehensive Model Evaluation Summary ---")
print(pd.DataFrame(results).T.round(3))


# --- 7. Create feature importance plots for ensemble models ---
print("\n--- 7. Generating Feature Importance Plots ---")

# To get feature importances, we need to fit the preprocessor and then the model on the full dataset
# or on a single train/test split. For plotting, fitting on the full dataset is common.
# We'll use the best individual model (tuned if available, else original) and the Voting/Stacking ensembles.

# Fit the preprocessor on the entire dataset to get transformed feature names
X_processed_array = preprocessor.fit_transform(X)

# Get feature names after one-hot encoding
ohe_feature_names = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_features)
all_feature_names = list(numerical_features) + list(ohe_feature_names)

# Ensure the number of features matches
if X_processed_array.shape[1] != len(all_feature_names):
    print("Warning: Mismatch in number of processed features and generated feature names. Feature importance might be inaccurate.")
    # This can happen if 'remainder' is not 'passthrough' or if there are issues with column selection.
    # For this specific problem, it should align.

# Fit the models on the preprocessed data to get feature importances
# Note: AdaBoostRegressor does not have feature_importances_ directly if base_estimator is not tree-based.
# For simplicity, we'll focus on RF, GB, XGBoost.

# Best individual model (tuned or untuned)
final_best_model_for_importance = best_tuned_model if 'best_tuned_model' in locals() and best_tuned_model is not None else models[best_model_name]
final_best_model_for_importance.fit(X_train, y_train) # Fit the pipeline on a train split for consistency

# Extract the regressor from the pipeline for feature importances
if hasattr(final_best_model_for_importance.named_steps['regressor'], 'feature_importances_'):
    importances = final_best_model_for_importance.named_steps['regressor'].feature_importances_
    feature_importances = pd.Series(importances, index=all_feature_names)
    top_features = feature_importances.nlargest(20)

    plt.figure(figsize=(12, 8))
    sns.barplot(x=top_features.values, y=top_features.index)
    plt.title(f'Top 20 Feature Importances for {tuned_model_name if "tuned_model_name" in locals() else best_model_name}')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.show()
else:
    print(f"Feature importances not available for {best_model_name} (or its tuned version).")

# For VotingRegressor, feature importances are not directly available.
# For StackingRegressor, feature importances are not directly available for the ensemble as a whole.
# One could inspect the base estimators' importances or the meta-learner's coefficients.
# For simplicity, we'll just show the best individual model's importances.


# --- 10. Analyze prediction errors and create residual plots ---
print("\n--- 10. Analyzing Prediction Errors and Creating Residual Plots ---")

# Use the best performing model (e.g., the best tuned model or the best ensemble)
# For this example, let's use the Stacking Ensemble as it often performs well.
# If tuning was skipped, use the best individual model.

final_model_for_residuals = stacking_pipeline # Use the fitted stacking pipeline from the last fold
# Or, refit on the entire dataset for a single prediction set
final_model_for_residuals.fit(X, y)
y_pred_final = final_model_for_residuals.predict(X)

residuals = y - y_pred_final

plt.figure(figsize=(14, 6))

# Residuals vs. Predicted Values Plot
plt.subplot(1, 2, 1)
sns.scatterplot(x=y_pred_final, y=residuals, alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Global Sales')
plt.ylabel('Residuals (Actual - Predicted)')
plt.title('Residual Plot')
plt.grid(True, linestyle='--', alpha=0.7)

# Distribution of Residuals
plt.subplot(1, 2, 2)
sns.histplot(residuals, kde=True, bins=50)
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Distribution of Residuals')
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

print("\n--- Analysis Complete ---")
```