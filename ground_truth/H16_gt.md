```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler, OneHotEncoder, SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import (
    VotingRegressor, BaggingRegressor, AdaBoostRegressor,
    GradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.calibration import CalibratedRegressorCV # More for classification, but can be adapted
from scipy.optimize import minimize

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

# Stacking
from sklearn.ensemble import StackingRegressor

# Hyperparameter Tuning
import optuna

# Model Interpretation
import shap
import lime
import lime.lime_tabular

# Set random seed for reproducibility
np.random.seed(42)

# --- 1. Data Loading ---
try:
    df = pd.read_csv('world-happiness-report-2021.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: 'world-happiness-report-2021.csv' not found. Please ensure the file is in the current directory.")
    exit()

# --- 2. Comprehensive Preprocessing & Feature Engineering ---

# Define target and features
TARGET = 'Ladder score'
# Drop columns that are identifiers, directly related to target calculation (potential leakage), or redundant
# 'Standard error of ladder score', 'upperwhisker', 'lowerwhisker' are related to uncertainty/intervals
# 'Ladder score in Dystopia' and 'Dystopia Residual' are components of 'Ladder score'
# 'Country name' is too granular, 'Regional indicator' is better for encoding
DROP_COLS = [
    'Country name', 'Standard error of ladder score', 'upperwhisker',
    'lowerwhisker', 'Ladder score in Dystopia', 'Dystopia Residual'
]
df_processed = df.drop(columns=DROP_COLS, errors='ignore')

# Identify numerical and categorical features
numerical_features = df_processed.select_dtypes(include=np.number).columns.tolist()
numerical_features.remove(TARGET) # Ensure target is not in features
categorical_features = df_processed.select_dtypes(include='object').columns.tolist()

# Preprocessing pipelines
# Numerical pipeline: Impute missing values with median, then scale
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Categorical pipeline: Impute missing values with most frequent, then one-hot encode
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Create a preprocessor using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Split data into features (X) and target (y)
X = df_processed.drop(columns=[TARGET])
y = df_processed[TARGET]

# Create a full pipeline for preprocessing and model training
# This will be used for nested CV and final model training
full_pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

# Fit preprocessor to get transformed feature names for interpretation later
X_transformed = full_pipeline.fit_transform(X)
# Get feature names after preprocessing
# This is a bit tricky with ColumnTransformer, need to get names from each transformer
ohe_feature_names = full_pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features)
all_feature_names = numerical_features + list(ohe_feature_names)

# --- 3. Implement Multiple Ensemble Methods ---

# Define base estimators for ensembles
# Using a mix of linear, tree-based, and instance-based models for diversity
base_estimators = {
    'LinearRegression': LinearRegression(),
    'Ridge': Ridge(random_state=42),
    'Lasso': Lasso(random_state=42),
    'DecisionTree': DecisionTreeRegressor(random_state=42),
    'RandomForest': RandomForestRegressor(random_state=42, n_estimators=100, max_depth=10),
    'ExtraTrees': ExtraTreesRegressor(random_state=42, n_estimators=100, max_depth=10),
    'SVR': SVR(),
    'KNeighbors': KNeighborsRegressor(),
    'XGBoost': xgb.XGBRegressor(random_state=42, n_estimators=100, learning_rate=0.1, use_label_encoder=False, eval_metric='rmse'),
    'LightGBM': lgb.LGBMRegressor(random_state=42, n_estimators=100, learning_rate=0.1),
    'CatBoost': CatBoostRegressor(random_state=42, verbose=0, n_estimators=100, learning_rate=0.1)
}

print("\n--- Ensemble Methods Training ---")

# --- Voting Regressor (Hard and Soft Voting) ---
# For regression, "hard voting" typically means simple averaging, "soft voting" means weighted averaging.
# VotingRegressor directly supports weighted averaging.

# Define estimators for VotingRegressor (a diverse set)
estimators_for_voting = [
    ('rf', base_estimators['RandomForest']),
    ('xgb', base_estimators['XGBoost']),
    ('lgbm', base_estimators['LightGBM']),
    ('ridge', base_estimators['Ridge'])
]

# Simple Averaging (equivalent to "hard voting" for regression)
voting_regressor_hard = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', VotingRegressor(estimators=estimators_for_voting, weights=None)) # weights=None for simple average
])
voting_regressor_hard.fit(X, y)
print(f"Voting Regressor (Hard/Simple Average) trained. R2: {voting_regressor_hard.score(X, y):.4f}")

# Weighted Averaging (Soft Voting) - weights can be optimized or set manually
# For demonstration, let's assign arbitrary weights. In practice, these would be tuned.
voting_regressor_soft = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', VotingRegressor(estimators=estimators_for_voting, weights=[0.25, 0.3, 0.3, 0.15]))
])
voting_regressor_soft.fit(X, y)
print(f"Voting Regressor (Soft/Weighted Average) trained. R2: {voting_regressor_soft.score(X, y):.4f}")

# --- Bagging with different base estimators ---
bagging_dt = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', BaggingRegressor(base_estimator=DecisionTreeRegressor(random_state=42), n_estimators=50, random_state=42))
])
bagging_dt.fit(X, y)
print(f"Bagging (DecisionTree) trained. R2: {bagging_dt.score(X, y):.4f}")

bagging_knn = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', BaggingRegressor(base_estimator=KNeighborsRegressor(), n_estimators=50, random_state=42))
])
bagging_knn.fit(X, y)
print(f"Bagging (KNeighbors) trained. R2: {bagging_knn.score(X, y):.4f}")

# --- Boosting algorithms ---
boosting_models = {}
for name, estimator in {
    'AdaBoost': AdaBoostRegressor(base_estimator=DecisionTreeRegressor(max_depth=5), n_estimators=50, random_state=42),
    'GradientBoosting': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
    'XGBoost': xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42, use_label_encoder=False, eval_metric='rmse'),
    'LightGBM': lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
    'CatBoost': CatBoostRegressor(n_estimators=100, learning_rate=0.1, random_state=42, verbose=0)
}.items():
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', estimator)
    ])
    model_pipeline.fit(X, y)
    boosting_models[name] = model_pipeline
    print(f"{name} trained. R2: {model_pipeline.score(X, y):.4f}")

# --- 4. Create Stacking Ensemble ---
# Base learners for Stacking
level0_estimators = [
    ('rf', base_estimators['RandomForest']),
    ('xgb', base_estimators['XGBoost']),
    ('lgbm', base_estimators['LightGBM']),
    ('svr', base_estimators['SVR'])
]

# Meta-learner
meta_learner = Ridge(random_state=42)

stacking_regressor = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', StackingRegressor(estimators=level0_estimators, final_estimator=meta_learner, cv=5, n_jobs=-1))
])
stacking_regressor.fit(X, y)
print(f"Stacking Regressor trained. R2: {stacking_regressor.score(X, y):.4f}")

# --- 5. Bayesian Optimization for Hyperparameter Tuning (Optuna) ---
print("\n--- Hyperparameter Tuning with Optuna (for XGBoost) ---")

# Define an objective function for Optuna
def objective(trial):
    # Hyperparameters to tune for XGBoost
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_uniform('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_loguniform('gamma', 1e-8, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-8, 1.0),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-8, 1.0),
        'random_state': 42,
        'use_label_encoder': False,
        'eval_metric': 'rmse'
    }

    # Create a pipeline with preprocessor and XGBoost
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', xgb.XGBRegressor(**params))
    ])

    # Use cross-validation to evaluate the model
    # Using a smaller number of folds for faster tuning
    cv_scores = cross_val_score(model_pipeline, X, y, cv=3, scoring='neg_root_mean_squared_error', n_jobs=-1)
    rmse = -np.mean(cv_scores)
    return rmse

# Create an Optuna study and optimize
study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
# Run a limited number of trials for demonstration purposes
study.optimize(objective, n_trials=20, show_progress_bar=True)

print(f"Number of finished trials: {len(study.trials)}")
print(f"Best trial: {study.best_trial.value:.4f} RMSE")
print("Best hyperparameters: ", study.best_trial.params)

# Store the best XGBoost model with tuned hyperparameters
best_xgb_params = study.best_trial.params
best_xgb_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', xgb.XGBRegressor(**best_xgb_params, use_label_encoder=False, eval_metric='rmse'))
])
best_xgb_model.fit(X, y)
print(f"Best XGBoost model (tuned) trained. R2: {best_xgb_model.score(X, y):.4f}")


# --- 6. Nested Cross-Validation for Robust Model Evaluation ---
print("\n--- Nested Cross-Validation ---")

# Outer CV loop for robust evaluation
outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Inner CV loop for hyperparameter tuning (Optuna will handle this implicitly)
# For simplicity, we'll use the best_xgb_model found above as an example for nested CV evaluation.
# In a full nested CV, Optuna would be run *within* each outer fold.

# Let's demonstrate nested CV for the best_xgb_model and the stacking_regressor
models_for_nested_cv = {
    'Tuned_XGBoost': best_xgb_model,
    'Stacking_Regressor': stacking_regressor,
    'Voting_Soft': voting_regressor_soft
}

nested_cv_results = {}
for model_name, model_pipeline in models_for_nested_cv.items():
    print(f"\nEvaluating {model_name} with Nested CV...")
    fold_rmse_scores = []
    fold_r2_scores = []

    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # For a true nested CV, hyperparameter tuning (e.g., Optuna) would happen here
        # on X_train, y_train. For demonstration, we use the already tuned model.
        model_pipeline.fit(X_train, y_train)
        y_pred = model_pipeline.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        fold_rmse_scores.append(rmse)
        fold_r2_scores.append(r2)
        print(f"  Fold {fold+1}: RMSE = {rmse:.4f}, R2 = {r2:.4f}")

    avg_rmse = np.mean(fold_rmse_scores)
    avg_r2 = np.mean(fold_r2_scores)
    nested_cv_results[model_name] = {'RMSE': avg_rmse, 'R2': avg_r2}
    print(f"  Average {model_name} (Nested CV): RMSE = {avg_rmse:.4f}, R2 = {avg_r2:.4f}")

print("\nNested CV Results Summary:")
for model_name, metrics in nested_cv_results.items():
    print(f"  {model_name}: Avg RMSE = {metrics['RMSE']:.4f}, Avg R2 = {metrics['R2']:.4f}")


# --- 7. Custom Ensemble with Dynamic Weight Assignment ---
print("\n--- Custom Ensemble with Dynamic Weight Assignment ---")

# Train a set of diverse models and get their out-of-fold (OOF) predictions
# This is crucial for avoiding data leakage when calculating weights.
models_for_custom_ensemble = {
    'RF': RandomForestRegressor(random_state=42, n_estimators=100, max_depth=10),
    'XGB': xgb.XGBRegressor(random_state=42, n_estimators=100, use_label_encoder=False, eval_metric='rmse'),
    'LGBM': lgb.LGBMRegressor(random_state=42, n_estimators=100),
    'GBM': GradientBoostingRegressor(random_state=42, n_estimators=100),
    'Ridge': Ridge(random_state=42)
}

oof_predictions = pd.DataFrame(index=X.index)
oof_errors = {}

# Use KFold for OOF predictions
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for name, estimator in models_for_custom_ensemble.items():
    print(f"Generating OOF predictions for {name}...")
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', estimator)
    ])
    # Get OOF predictions
    oof_preds = cross_val_predict(model_pipeline, X, y, cv=kf, n_jobs=-1)
    oof_predictions[name] = oof_preds
    # Calculate OOF RMSE
    rmse = np.sqrt(mean_squared_error(y, oof_preds))
    oof_errors[name] = rmse
    print(f"  {name} OOF RMSE: {rmse:.4f}")

# Calculate dynamic weights based on inverse of OOF RMSE
total_inverse_rmse = sum(1/error for error in oof_errors.values())
dynamic_weights = {name: (1/error) / total_inverse_rmse for name, error in oof_errors.items()}

print("\nDynamic Weights based on OOF RMSE:")
for name, weight in dynamic_weights.items():
    print(f"  {name}: {weight:.4f}")

# Combine OOF predictions using dynamic weights
weighted_oof_predictions = np.zeros(len(X))
for name, weight in dynamic_weights.items():
    weighted_oof_predictions += oof_predictions[name] * weight

custom_ensemble_rmse = np.sqrt(mean_squared_error(y, weighted_oof_predictions))
custom_ensemble_r2 = r2_score(y, weighted_oof_predictions)
print(f"\nCustom Ensemble (Dynamic Weights) OOF RMSE: {custom_ensemble_rmse:.4f}")
print(f"Custom Ensemble (Dynamic Weights) OOF R2: {custom_ensemble_r2:.4f}")

# To use this ensemble for new predictions, you'd train each base model on the full dataset
# and then combine their predictions with the calculated dynamic_weights.
final_custom_ensemble_models = {}
for name, estimator in models_for_custom_ensemble.items():
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', estimator)
    ])
    model_pipeline.fit(X, y)
    final_custom_ensemble_models[name] = model_pipeline

def predict_custom_ensemble(X_new, models, weights):
    predictions = np.zeros(len(X_new))
    for name, model in models.items():
        predictions += model.predict(X_new) * weights[name]
    return predictions

# Example prediction with custom ensemble
# y_pred_custom_ensemble = predict_custom_ensemble(X, final_custom_ensemble_models, dynamic_weights)


# --- 8. Advanced Techniques: Blending and Multi-level Stacking ---
print("\n--- Advanced Techniques: Blending and Multi-level Stacking ---")

# Blending: Split data into train_blend and val_blend.
# Train base models on train_blend, predict on val_blend.
# Train meta-model on val_blend predictions.

X_train_blend, X_val_blend, y_train_blend, y_val_blend = train_test_split(X, y, test_size=0.2, random_state=42)

# Base models for blending (using pipelines)
blending_base_models = {
    'RF': Pipeline(steps=[('preprocessor', preprocessor), ('regressor', RandomForestRegressor(random_state=42, n_estimators=50))]),
    'XGB': Pipeline(steps=[('preprocessor', preprocessor), ('regressor', xgb.XGBRegressor(random_state=42, n_estimators=50, use_label_encoder=False, eval_metric='rmse'))]),
    'LGBM': Pipeline(steps=[('preprocessor', preprocessor), ('regressor', lgb.LGBMRegressor(random_state=42, n_estimators=50))])
}

val_blend_predictions = pd.DataFrame()
for name, model_pipeline in blending_base_models.items():
    model_pipeline.fit(X_train_blend, y_train_blend)
    val_blend_predictions[name] = model_pipeline.predict(X_val_blend)
    print(f"  Blending Base Model {name} trained.")

# Meta-learner for blending
blending_meta_learner = LinearRegression()
blending_meta_learner.fit(val_blend_predictions, y_val_blend)

# Evaluate blending ensemble
blending_preds = blending_meta_learner.predict(val_blend_predictions)
blending_rmse = np.sqrt(mean_squared_error(y_val_blend, blending_preds))
blending_r2 = r2_score(y_val_blend, blending_preds)
print(f"Blending Ensemble RMSE on validation set: {blending_rmse:.4f}")
print(f"Blending Ensemble R2 on validation set: {blending_r2:.4f}")

# Multi-level Stacking (simplified 2-level example)
# Level 1: StackingRegressor
level1_stack = StackingRegressor(
    estimators=[
        ('rf', RandomForestRegressor(random_state=42, n_estimators=50)),
        ('xgb', xgb.XGBRegressor(random_state=42, n_estimators=50, use_label_encoder=False, eval_metric='rmse'))
    ],
    final_estimator=Ridge(random_state=42),
    cv=3, n_jobs=-1
)

# Level 1 pipeline
level1_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', level1_stack)
])

# Level 2: Another StackingRegressor or a simple model taking Level 1's output
# For simplicity, let's use a simple LinearRegression as Level 2 meta-learner
# and feed it the predictions from Level 1.
# This requires getting OOF predictions from Level 1.

# Get OOF predictions from Level 1
level1_oof_preds = cross_val_predict(level1_pipeline, X, y, cv=5, n_jobs=-1)

# Train a simple meta-learner on Level 1 OOF predictions
level2_meta_learner = LinearRegression()
level2_meta_learner.fit(level1_oof_preds.reshape(-1, 1), y)

# Evaluate multi-level stacking (using OOF from Level 1 as input to Level 2)
multi_level_rmse = np.sqrt(mean_squared_error(y, level2_meta_learner.predict(level1_oof_preds.reshape(-1, 1))))
multi_level_r2 = r2_score(y, level2_meta_learner.predict(level1_oof_preds.reshape(-1, 1)))
print(f"Multi-level Stacking (2-level) OOF RMSE: {multi_level_rmse:.4f}")
print(f"Multi-level Stacking (2-level) OOF R2: {multi_level_r2:.4f}")


# --- 9. Model Interpretation (SHAP values and LIME) ---
print("\n--- Model Interpretation (SHAP and LIME) ---")

# For interpretation, let's use the best_xgb_model from Optuna, as it's a powerful single model.
# SHAP values
print("\nGenerating SHAP explanations...")
# Need to get the preprocessed data and the actual regressor from the pipeline
X_processed_for_shap = best_xgb_model.named_steps['preprocessor'].transform(X)
xgb_regressor = best_xgb_model.named_steps['regressor']

# Create a DataFrame for SHAP with feature names
X_processed_df = pd.DataFrame(X_processed_for_shap, columns=all_feature_names)

# Create a SHAP explainer
explainer = shap.TreeExplainer(xgb_regressor)
shap_values = explainer.shap_values(X_processed_df)

# SHAP Summary Plot (Global Interpretation)
print("Displaying SHAP summary plot (may take a moment)...")
shap.summary_plot(shap_values, X_processed_df, plot_type="bar", show=False)
plt.title("SHAP Feature Importance (Bar Plot)")
plt.tight_layout()
plt.show()

shap.summary_plot(shap_values, X_processed_df, show=False)
plt.title("SHAP Summary Plot (Beeswarm)")
plt.tight_layout()
plt.show()

# SHAP Force Plot (Local Interpretation for a single instance)
print("Displaying SHAP force plot for a sample instance...")
# Choose an instance (e.g., the first one)
sample_instance_idx = 0
shap.initjs() # Initialize JS for interactive plots
shap.force_plot(explainer.expected_value, shap_values[sample_instance_idx,:], X_processed_df.iloc[sample_instance_idx,:], show=False)
plt.title(f"SHAP Force Plot for Instance {sample_instance_idx}")
plt.tight_layout()
plt.show()


# LIME (Local Interpretation)
print("\nGenerating LIME explanations...")
# LIME requires a prediction function that takes raw data and returns predictions
# and feature names.
# For LIME, we need the preprocessor and the model to be combined.
# The `predict_proba` method is usually for classification, for regression, it's just `predict`.
# LIME explainer needs a function that takes a 2D numpy array of *raw* data and returns predictions.

def predict_fn_for_lime(data):
    # data is a numpy array, convert back to DataFrame for pipeline
    data_df = pd.DataFrame(data, columns=X.columns)
    return best_xgb_model.predict(data_df)

# Create a LIME explainer
# mode='regression' for regression tasks
# feature_names should be the original feature names
explainer_lime = lime.lime_tabular.LimeTabularExplainer(
    training_data=X.values,
    feature_names=X.columns.tolist(),
    class_names=[TARGET], # For regression, this is just the target name
    mode='regression'
)

# Explain a single instance (e.g., the first one from the original dataset)
print("Displaying LIME explanation for a sample instance...")
instance_to_explain = X.iloc[sample_instance_idx]
explanation = explainer_lime.explain_instance(
    data_row=instance_to_explain.values,
    predict_fn=predict_fn_for_lime,
    num_features=5 # Number of features to show in explanation
)

# Display the explanation
explanation.show_in_notebook(show_table=True, show_all=False)
plt.title(f"LIME Explanation for Instance {sample_instance_idx}")
plt.tight_layout()
plt.show()


# --- 10. Model Calibration and Uncertainty Quantification ---
print("\n--- Model Calibration and Uncertainty Quantification ---")

# Model Calibration (for regression, often involves analyzing residuals)
# A well-calibrated regression model should have residuals that are randomly distributed
# around zero, with constant variance (homoscedasticity).

# Let's use the best_xgb_model for this analysis
y_pred_xgb = best_xgb_model.predict(X)
residuals = y - y_pred_xgb

plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_pred_xgb, y=residuals, alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Residuals vs. Predicted Values (Best XGBoost Model)")
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# Quantile Regression for Uncertainty Quantification (Prediction Intervals)
# GradientBoostingRegressor can predict quantiles.
print("\nQuantile Regression for Prediction Intervals...")

# Train three GBR models: one for the median (0.5), one for lower (0.05), one for upper (0.95) quantiles
gbr_lower = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', GradientBoostingRegressor(loss='quantile', alpha=0.05, random_state=42))
])
gbr_median = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', GradientBoostingRegressor(loss='quantile', alpha=0.5, random_state=42))
])
gbr_upper = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', GradientBoostingRegressor(loss='quantile', alpha=0.95, random_state=42))
])

gbr_lower.fit(X, y)
gbr_median.fit(X, y)
gbr_upper.fit(X, y)

# Get predictions for a sample
sample_X = X.sample(n=50, random_state=42) # Take a small sample for plotting
y_pred_lower = gbr_lower.predict(sample_X)
y_pred_median = gbr_median.predict(sample_X)
y_pred_upper = gbr_upper.predict(sample_X)

# Plot prediction intervals
plt.figure(figsize=(12, 7))
plt.scatter(sample_X.index, y.loc[sample_X.index], label='Actual Values', color='blue', alpha=0.7)
plt.plot(sample_X.index, y_pred_median, label='Predicted Median', color='red', linestyle='--')
plt.fill_between(sample_X.index, y_pred_lower, y_pred_upper, color='green', alpha=0.2, label='90% Prediction Interval')
plt.xlabel("Sample Index")
plt.ylabel("Ladder Score")
plt.title("Prediction Intervals using Quantile Regression")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()


# --- 11. Ensemble Diversity Analysis and Model Combination Optimization ---
print("\n--- Ensemble Diversity Analysis and Model Combination Optimization ---")

# Ensemble Diversity Analysis: Correlation of OOF predictions
print("\nAnalyzing Ensemble Diversity (Correlation of OOF Predictions):")
# Use the oof_predictions DataFrame created earlier for custom ensemble
if not oof_predictions.empty:
    correlation_matrix = oof_predictions.corr()
    print("Correlation Matrix of OOF Predictions:")
    print(correlation_matrix)

    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Matrix of Out-of-Fold Predictions")
    plt.show()
else:
    print("OOF predictions not available for diversity analysis. Run custom ensemble section first.")


# Model Combination Optimization (finding optimal weights for linear combination)
print("\nOptimizing Model Combination Weights:")

# We want to find weights w_i that minimize RMSE for combined predictions: sum(w_i * pred_i)
# Constraints: sum(w_i) = 1, w_i >= 0

# Define the objective function to minimize (RMSE)
def objective_weights(weights, predictions, actual_values):
    combined_predictions = np.dot(predictions, weights)
    return np.sqrt(mean_squared_error(actual_values, combined_predictions))

# Initial guess for weights (equal weights)
initial_weights = np.array([1.0 / len(oof_predictions.columns)] * len(oof_predictions.columns))

# Bounds for weights (0 to 1)
bounds = [(0.0, 1.0)] * len(oof_predictions.columns)

# Constraints: sum of weights must be 1
constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})

# Perform optimization
if not oof_predictions.empty:
    result = minimize(
        objective_weights,
        initial_weights,
        args=(oof_predictions.values, y.values),
        method='SLSQP', # Sequential Least Squares Programming
        bounds=bounds,
        constraints=constraints
    )

    optimal_weights = result.x
    optimized_rmse = result.fun

    print("\nOptimization Results:")
    print(f"Optimal RMSE: {optimized_rmse:.4f}")
    print("Optimal Weights:")
    for i, col in enumerate(oof_predictions.columns):
        print(f"  {col}: {optimal_weights[i]:.4f}")

    # Verify sum of weights
    print(f"Sum of optimal weights: {np.sum(optimal_weights):.4f}")

    # Compare with simple average
    simple_average_weights = np.array([1.0 / len(oof_predictions.columns)] * len(oof_predictions.columns))
    simple_average_rmse = objective_weights(simple_average_weights, oof_predictions.values, y.values)
    print(f"Simple Average RMSE: {simple_average_rmse:.4f}")

else:
    print("OOF predictions not available for weight optimization. Run custom ensemble section first.")

print("\n--- All tasks completed ---")
```