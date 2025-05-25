```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import scipy.stats as stats
import warnings

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, learning_curve, validation_curve
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression # For Stacking meta-learner and Blending
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix, brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.utils import resample
from sklearn.base import BaseEstimator, TransformerMixin, clone # For custom transformer and cloning models
from sklearn.exceptions import ConvergenceWarning

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=ConvergenceWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# For XGBoost
try:
    import xgboost as xgb
except ImportError:
    print("XGBoost not installed. Please install it using 'pip install xgboost'")
    xgb = None

# Set random seed for reproducibility
np.random.seed(42)

# --- 1. Data Loading and Initial Inspection ---
# Load the dataset
try:
    df = pd.read_csv('water_potability.csv')
except FileNotFoundError:
    print("Error: 'water_potability.csv' not found. Please ensure the file is in the current directory.")
    exit()

# Separate features (X) and target (y)
X = df.drop('Potability', axis=1)
y = df['Potability']

# Identify numerical features for preprocessing (all features are numerical in this dataset)
numerical_features = X.columns.tolist()

# --- 2. Comprehensive Data Preprocessing ---

# Custom transformer for outlier capping (Winsorization)
class OutlierCapper(BaseEstimator, TransformerMixin):
    def __init__(self, lower_quantile=0.01, upper_quantile=0.99):
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.lower_bounds = {}
        self.upper_bounds = {}

    def fit(self, X, y=None):
        for col in X.columns:
            self.lower_bounds[col] = X[col].quantile(self.lower_quantile)
            self.upper_bounds[col] = X[col].quantile(self.upper_quantile)
        return self

    def transform(self, X):
        X_transformed = X.copy()
        for col in X.columns:
            X_transformed[col] = np.clip(X_transformed[col], self.lower_bounds[col], self.upper_bounds[col])
        return X_transformed

# Preprocessing pipeline: Imputation -> Outlier Capping -> Scaling -> Polynomial Features
preprocessor = Pipeline(steps=[
    ('imputer', IterativeImputer(max_iter=10, random_state=42)), # Multiple Imputation (MICE)
    ('outlier_capper', OutlierCapper(lower_quantile=0.01, upper_quantile=0.99)), # Outlier Treatment
    ('scaler', StandardScaler()), # Feature Scaling
    ('poly_features', PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)) # Polynomial and interaction features
])

# Create a ColumnTransformer to apply preprocessing to all numerical features
full_preprocessor = ColumnTransformer(
    transformers=[
        ('num', preprocessor, numerical_features)
    ],
    remainder='passthrough' # Keep other columns if any (not applicable here)
)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Fit and transform the training data, transform the test data
X_train_processed = full_preprocessor.fit_transform(X_train)
X_test_processed = full_preprocessor.transform(X_test)

# Get feature names after preprocessing for interpretability
# This requires accessing the fitted PolynomialFeatures step within the pipeline
poly_feature_names = full_preprocessor.named_transformers_['num'].named_steps['poly_features'].get_feature_names_out(numerical_features)
processed_feature_names = poly_feature_names

# Convert processed arrays back to DataFrames for easier handling (especially for SHAP)
X_train_processed_df = pd.DataFrame(X_train_processed, columns=processed_feature_names, index=X_train.index)
X_test_processed_df = pd.DataFrame(X_test_processed, columns=processed_feature_names, index=X_test.index)


# --- 3. Base Model Building and Optimization ---

# Define models and their parameter grids for GridSearchCV
models = {
    'RandomForest': {
        'model': RandomForestClassifier(random_state=42),
        'params': {
            'n_estimators': [100, 200],
            'max_depth': [10, 20],
            'min_samples_leaf': [1, 5]
        }
    },
    'GradientBoosting': {
        'model': GradientBoostingClassifier(random_state=42),
        'params': {
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1],
            'max_depth': [3, 5]
        }
    },
    'XGBoost': {
        'model': xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', use_label_encoder=False, random_state=42) if xgb else None,
        'params': {
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1],
            'max_depth': [3, 5]
        } if xgb else {}
    },
    'NeuralNetwork': {
        'model': MLPClassifier(random_state=42, max_iter=500),
        'params': {
            'hidden_layer_sizes': [(50,), (100, 50)],
            'alpha': [0.0001, 0.001], # L2 regularization
            'learning_rate_init': [0.001, 0.01]
        }
    },
    'SVM': {
        'model': SVC(probability=True, random_state=42), # probability=True for ROC AUC and calibration
        'params': {
            'C': [0.1, 1],
            'kernel': ['rbf']
        }
    }
}

best_models = {}
model_results = {}
cv_scores_per_model = {} # To store scores per fold for statistical testing

print("--- Starting Base Model Optimization (GridSearchCV) ---")
for name, config in models.items():
    if config['model'] is None:
        print(f"Skipping {name} as its library is not available.")
        continue

    print(f"\nOptimizing {name}...")
    grid_search = GridSearchCV(
        estimator=config['model'],
        param_grid=config['params'],
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring='roc_auc', # Use ROC AUC as primary scoring metric
        n_jobs=-1, # Use all available cores
        verbose=0 # Set to 1 for more verbose output
    )
    grid_search.fit(X_train_processed_df, y_train)

    best_models[name] = grid_search.best_estimator_
    model_results[name] = {
        'best_score': grid_search.best_score_,
        'best_params': grid_search.best_params_
    }
    print(f"Best ROC AUC for {name}: {grid_search.best_score_:.4f}")
    print(f"Best params for {name}: {grid_search.best_params_}")

    # Re-run CV for the best model to get individual fold scores for paired t-test
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_scores = []
    for train_idx, val_idx in skf.split(X_train_processed_df, y_train):
        model = clone(best_models[name]) # Clone to ensure fresh model for each fold
        model.fit(X_train_processed_df.iloc[train_idx], y_train.iloc[train_idx])
        y_pred_proba = model.predict_proba(X_train_processed_df.iloc[val_idx])[:, 1]
        fold_scores.append(roc_auc_score(y_train.iloc[val_idx], y_pred_proba))
    cv_scores_per_model[name] = fold_scores

print("\n--- Base Model Optimization Complete ---")
for name, results in model_results.items():
    print(f"{name}: Best ROC AUC = {results['best_score']:.4f}")

# --- 4. Ensemble Models ---

print("\n--- Building Ensemble Models ---")

# Voting Classifier
# Use the best base models found. Ensure they support predict_proba for 'soft' voting.
estimators_for_voting = [(name, model) for name, model in best_models.items() if hasattr(model, 'predict_proba')]
if estimators_for_voting:
    voting_clf = VotingClassifier(estimators=estimators_for_voting, voting='soft', n_jobs=-1)
    voting_clf.fit(X_train_processed_df, y_train)
    y_pred_proba_voting = voting_clf.predict_proba(X_test_processed_df)[:, 1]
    roc_auc_voting = roc_auc_score(y_test, y_pred_proba_voting)
    print(f"Voting Classifier ROC AUC: {roc_auc_voting:.4f}")
    best_models['VotingClassifier'] = voting_clf
    model_results['VotingClassifier'] = {'best_score': roc_auc_voting, 'best_params': 'N/A'}
else:
    print("No probability-enabled base models available for Voting Classifier.")

# Stacking Classifier
# Use a meta-learner (e.g., Logistic Regression)
estimators_for_stacking = [(name, model) for name, model in best_models.items() if hasattr(model, 'predict_proba')]
if estimators_for_stacking:
    stacking_clf = StackingClassifier(
        estimators=estimators_for_stacking,
        final_estimator=LogisticRegression(random_state=42), # Meta-learner
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        n_jobs=-1
    )
    stacking_clf.fit(X_train_processed_df, y_train)
    y_pred_proba_stacking = stacking_clf.predict_proba(X_test_processed_df)[:, 1]
    roc_auc_stacking = roc_auc_score(y_test, y_pred_proba_stacking)
    print(f"Stacking Classifier ROC AUC: {roc_auc_stacking:.4f}")
    best_models['StackingClassifier'] = stacking_clf
    model_results['StackingClassifier'] = {'best_score': roc_auc_stacking, 'best_params': 'N/A'}
else:
    print("No probability-enabled base models available for Stacking Classifier.")

# Blending (Custom Implementation)
print("\nImplementing Blending...")
# Split X_train into meta_train and meta_val for blending
X_blend_train, X_blend_val, y_blend_train, y_blend_val = train_test_split(
    X_train_processed_df, y_train, test_size=0.3, random_state=42, stratify=y_train
)

# Train base models on X_blend_train
blend_base_models = {}
for name, model in best_models.items():
    # Exclude ensemble models themselves and ensure predict_proba capability
    if name not in ['VotingClassifier', 'StackingClassifier'] and hasattr(model, 'predict_proba'):
        print(f"Training blending base model: {name}")
        model_clone = clone(model) # Clone to avoid modifying original best_models
        model_clone.fit(X_blend_train, y_blend_train)
        blend_base_models[name] = model_clone

# Generate meta-features (predictions from base models on X_blend_val)
meta_features_val = []
for name, model in blend_base_models.items():
    meta_features_val.append(model.predict_proba(X_blend_val)[:, 1])
meta_features_val = np.array(meta_features_val).T # Transpose to (n_samples, n_models)

# Train meta-learner on meta_features_val
meta_learner = LogisticRegression(random_state=42)
meta_learner.fit(meta_features_val, y_blend_val)

# Generate meta-features for X_test
meta_features_test = []
for name, model in blend_base_models.items():
    meta_features_test.append(model.predict_proba(X_test_processed_df)[:, 1])
meta_features_test = np.array(meta_features_test).T

# Predict with blending
y_pred_proba_blending = meta_learner.predict_proba(meta_features_test)[:, 1]
roc_auc_blending = roc_auc_score(y_test, y_pred_proba_blending)
print(f"Blending Classifier ROC AUC: {roc_auc_blending:.4f}")
# Store blending results (store the meta-learner and its performance)
best_models['BlendingClassifier'] = meta_learner
model_results['BlendingClassifier'] = {'best_score': roc_auc_blending, 'best_params': 'N/A'}


# --- 5. Advanced Evaluation ---

# Function to evaluate a model and return various metrics
def evaluate_model(model, X_data, y_data):
    # Check if model has predict_proba, otherwise use predict and warn
    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X_data)[:, 1]
        y_pred = (y_pred_proba > 0.5).astype(int) # Convert probabilities to binary predictions
    else:
        y_pred = model.predict(X_data)
        y_pred_proba = np.zeros_like(y_pred, dtype=float) # Placeholder if no probabilities
        print(f"Warning: Model {model.__class__.__name__} does not support predict_proba. ROC AUC and Brier Score will be 0 or invalid.")

    accuracy = accuracy_score(y_data, y_pred)
    precision = precision_score(y_data, y_pred, zero_division=0)
    recall = recall_score(y_data, y_pred, zero_division=0)
    f1 = f1_score(y_data, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_data, y_pred_proba) if hasattr(model, 'predict_proba') else 0.0
    brier = brier_score_loss(y_data, y_pred_proba) if hasattr(model, 'predict_proba') else 0.0

    return {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'ROC AUC': roc_auc,
        'Brier Score': brier,
        'y_pred_proba': y_pred_proba,
        'y_pred': y_pred
    }

# Store all evaluation results for dashboard
all_model_eval_results = {}
for name, model in best_models.items():
    if name == 'BlendingClassifier': # Special handling for blending as 'model' is just the meta-learner
        all_model_eval_results[name] = {
            'Accuracy': accuracy_score(y_test, (y_pred_proba_blending > 0.5).astype(int)),
            'Precision': precision_score(y_test, (y_pred_proba_blending > 0.5).astype(int), zero_division=0),
            'Recall': recall_score(y_test, (y_pred_proba_blending > 0.5).astype(int), zero_division=0),
            'F1-Score': f1_score(y_test, (y_pred_proba_blending > 0.5).astype(int), zero_division=0),
            'ROC AUC': roc_auc_blending,
            'Brier Score': brier_score_loss(y_test, y_pred_proba_blending),
            'y_pred_proba': y_pred_proba_blending,
            'y_pred': (y_pred_proba_blending > 0.5).astype(int)
        }
    else:
        all_model_eval_results[name] = evaluate_model(model, X_test_processed_df, y_test)


# --- Learning Curves ---
print("\n--- Generating Learning Curves ---")
# Only plot learning curves for base models, not ensembles or models without predict_proba
base_models_for_lc = {k: v for k, v in best_models.items() if k not in ['VotingClassifier', 'StackingClassifier', 'BlendingClassifier'] and hasattr(v, 'predict_proba')}

if base_models_for_lc:
    plt.figure(figsize=(15, 10))
    for i, (name, model) in enumerate(base_models_for_lc.items()):
        train_sizes, train_scores, test_scores = learning_curve(
            estimator=model,
            X=X_train_processed_df,
            y=y_train,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring='roc_auc',
            n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10),
            random_state=42
        )
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        plt.subplot(2, 3, i + 1) # Adjust subplot grid as needed
        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1, color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1, color="g")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
        plt.title(f"Learning Curve ({name})")
        plt.xlabel("Training examples")
        plt.ylabel("ROC AUC Score")
        plt.legend(loc="best")
        plt.grid(True)
    plt.tight_layout()
    plt.suptitle("Learning Curves for Base Models", y=1.02, fontsize=16)
    plt.show()
else:
    print("No suitable base models found for generating learning curves.")


# --- Validation Curves (Example for RandomForest: max_depth) ---
print("\n--- Generating Validation Curves (Example: RandomForest max_depth) ---")
if 'RandomForest' in best_models and hasattr(best_models['RandomForest'], 'predict_proba'):
    param_range = np.arange(1, 20, 2) # Example range for max_depth
    train_scores, test_scores = validation_curve(
        estimator=best_models['RandomForest'],
        X=X_train_processed_df,
        y=y_train,
        param_name="max_depth",
        param_range=param_range,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring="roc_auc",
        n_jobs=-1
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.figure(figsize=(8, 6))
    plt.plot(param_range, train_scores_mean, label="Training score", color="r")
    plt.fill_between(param_range, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2, color="r")
    plt.plot(param_range, test_scores_mean, label="Cross-validation score", color="g")
    plt.fill_between(param_range, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.2, color="g")
    plt.title("Validation Curve for RandomForest (max_depth)")
    plt.xlabel("Max Depth")
    plt.ylabel("ROC AUC Score")
    plt.legend(loc="best")
    plt.grid(True)
    plt.show()
else:
    print("RandomForest model not available or does not support predict_proba for validation curve example.")


# --- Bootstrap Sampling for Confidence Intervals ---
print("\n--- Performing Bootstrap Sampling for Confidence Intervals ---")
n_bootstraps = 100
bootstrap_metrics = {name: {'ROC AUC': [], 'F1-Score': []} for name in all_model_eval_results.keys()}

for model_name, results in all_model_eval_results.items():
    print(f"Bootstrapping for {model_name}...")
    for i in range(n_bootstraps):
        # Resample with replacement from the test set
        X_resample, y_resample = resample(X_test_processed_df, y_test, random_state=i, stratify=y_test)
        
        # Get predictions for the resampled data
        if model_name == 'BlendingClassifier':
            # Re-generate meta-features for resampled data
            meta_features_resample = []
            for name_base, model_base in blend_base_models.items():
                meta_features_resample.append(model_base.predict_proba(X_resample)[:, 1])
            meta_features_resample = np.array(meta_features_resample).T
            y_pred_proba_resample = best_models[model_name].predict_proba(meta_features_resample)[:, 1]
        else:
            model = best_models[model_name]
            if not hasattr(model, 'predict_proba'):
                continue # Skip models without predict_proba for ROC AUC/F1 based on proba
            y_pred_proba_resample = model.predict_proba(X_resample)[:, 1]
        
        y_pred_resample = (y_pred_proba_resample > 0.5).astype(int)

        bootstrap_metrics[model_name]['ROC AUC'].append(roc_auc_score(y_resample, y_pred_proba_resample))
        bootstrap_metrics[model_name]['F1-Score'].append(f1_score(y_resample, y_pred_resample, zero_division=0))

# Calculate confidence intervals
confidence_intervals = {}
for model_name, metrics in bootstrap_metrics.items():
    confidence_intervals[model_name] = {}
    for metric_name, scores in metrics.items():
        if scores: # Ensure there are scores to calculate CI
            lower = np.percentile(scores, 2.5)
            upper = np.percentile(scores, 97.5)
            confidence_intervals[model_name][metric_name] = (lower, upper)
            print(f"{model_name} {metric_name} 95% CI: ({lower:.4f}, {upper:.4f})")
        else:
            print(f"Warning: No bootstrap scores for {model_name} {metric_name}. CI not calculated.")


# --- 6. Statistical Significance Testing (Paired t-tests) ---
print("\n--- Performing Statistical Significance Testing (Paired t-tests) ---")

# Find the best performing model based on ROC AUC on test set
# Filter out models that don't have ROC AUC calculated
valid_models_for_comparison = {k: v for k, v in all_model_eval_results.items() if v['ROC AUC'] > 0}
if not valid_models_for_comparison:
    print("No valid models with ROC AUC scores for statistical comparison.")
else:
    best_overall_model_name = max(valid_models_for_comparison, key=lambda k: valid_models_for_comparison[k]['ROC AUC'])
    print(f"\nBest overall model (on test set ROC AUC): {best_overall_model_name}")

    # Compare the best model against all others using paired t-tests on CV scores
    if best_overall_model_name in cv_scores_per_model:
        best_model_cv_scores = cv_scores_per_model[best_overall_model_name]
        print(f"\nComparing {best_overall_model_name} against other models using paired t-tests (on CV ROC AUC scores):")
        for model_name, scores in cv_scores_per_model.items():
            if model_name == best_overall_model_name:
                continue
            if len(best_model_cv_scores) != len(scores):
                print(f"Skipping t-test for {model_name} due to unequal number of CV scores.")
                continue
            
            t_stat, p_value = stats.ttest_rel(best_model_cv_scores, scores)
            print(f"  {best_overall_model_name} vs {model_name}: t-statistic={t_stat:.3f}, p-value={p_value:.3f}")
            if p_value < 0.05:
                print(f"    (Statistically significant difference at alpha=0.05)")
            else:
                print(f"    (No statistically significant difference at alpha=0.05)")
    else:
        print(f"CV scores for {best_overall_model_name} not available for paired t-test.")


# --- 7. Model Interpretability (SHAP) ---
print("\n--- Performing Model Interpretability (SHAP) ---")

# Choose a tree-based model for SHAP (e.g., XGBoost or RandomForest)
shap_model_name = 'XGBoost' if 'XGBoost' in best_models and xgb else 'RandomForest'
if shap_model_name in best_models and hasattr(best_models[shap_model_name], 'predict_proba'):
    print(f"Generating SHAP values for {shap_model_name}...")
    model_for_shap = best_models[shap_model_name]

    try:
        # For tree models, TreeExplainer is efficient
        explainer = shap.TreeExplainer(model_for_shap)
        shap_values = explainer.shap_values(X_test_processed_df)

        # If binary classification, shap_values will be a list of two arrays (for class 0 and class 1)
        # We usually care about the positive class (class 1)
        if isinstance(shap_values, list):
            shap_values = shap_values[1] # For binary classification, take values for the positive class

        # SHAP Summary Plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_test_processed_df, show=False)
        plt.title(f"SHAP Summary Plot for {shap_model_name}")
        plt.tight_layout()
        plt.show()

        # SHAP Dependence Plot for top feature
        # Get mean absolute SHAP values to find top feature
        mean_abs_shap = np.abs(shap_values).mean(0)
        feature_importance_shap = pd.Series(mean_abs_shap, index=X_test_processed_df.columns)
        top_feature = feature_importance_shap.nlargest(1).index[0]

        plt.figure(figsize=(10, 6))
        shap.dependence_plot(top_feature, shap_values, X_test_processed_df, show=False)
        plt.title(f"SHAP Dependence Plot for {top_feature} ({shap_model_name})")
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Could not generate SHAP plots for {shap_model_name}: {e}")
else:
    print(f"Skipping SHAP: {shap_model_name} model not available, not tree-based, or does not support predict_proba.")


# --- 8. Model Calibration and Probability Calibration Plots ---
print("\n--- Performing Model Calibration and Plotting Calibration Curves ---")

# Choose the best overall model for calibration (must support predict_proba)
if 'best_overall_model_name' in locals() and hasattr(best_models[best_overall_model_name], 'predict_proba'):
    model_for_calibration = best_models[best_overall_model_name]

    print(f"Calibrating {best_overall_model_name}...")
    # CalibratedClassifierCV uses cross-validation internally to fit the calibrator
    calibrated_model_cv = CalibratedClassifierCV(model_for_calibration, method='isotonic', cv=5)
    calibrated_model_cv.fit(X_train_processed_df, y_train) # Fit on training data

    # Get probabilities from uncalibrated and calibrated models on test set
    y_prob_uncalibrated = model_for_calibration.predict_proba(X_test_processed_df)[:, 1]
    y_prob_calibrated = calibrated_model_cv.predict_proba(X_test_processed_df)[:, 1]

    # Plot calibration curves (Reliability Diagram)
    plt.figure(figsize=(10, 7))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))

    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

    # Uncalibrated
    fraction_of_positives, mean_predicted_value = calibration_curve(y_test, y_prob_uncalibrated, n_bins=10)
    ax1.plot(mean_predicted_value, fraction_of_positives, "s-", label=f"{best_overall_model_name} (Uncalibrated)")
    ax2.hist(y_prob_uncalibrated, range=(0, 1), bins=10, label="Uncalibrated", histtype="step", lw=2)

    # Calibrated
    fraction_of_positives_cal, mean_predicted_value_cal = calibration_curve(y_test, y_prob_calibrated, n_bins=10)
    ax1.plot(mean_predicted_value_cal, fraction_of_positives_cal, "s-", label=f"{best_overall_model_name} (Calibrated)")
    ax2.hist(y_prob_calibrated, range=(0, 1), bins=10, label="Calibrated", histtype="step", lw=2)

    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title(f'Calibration plots ({best_overall_model_name})')

    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")
    ax2.legend(loc="upper center", ncol=2)

    plt.tight_layout()
    plt.show()

    # Calculate Brier Score for calibrated vs uncalibrated
    brier_uncalibrated = brier_score_loss(y_test, y_prob_uncalibrated)
    brier_calibrated = brier_score_loss(y_test, y_prob_calibrated)
    print(f"Brier Score (Uncalibrated {best_overall_model_name}): {brier_uncalibrated:.4f}")
    print(f"Brier Score (Calibrated {best_overall_model_name}): {brier_calibrated:.4f}")
else:
    print("Skipping model calibration: Best overall model not found or does not support predict_proba.")


# --- 9. Comprehensive Model Comparison Dashboard ---
print("\n--- Generating Model Comparison Dashboard ---")

# Prepare data for dashboard
dashboard_data = []
for name, metrics in all_model_eval_results.items():
    row = {'Model': name}
    row.update({k: v for k, v in metrics.items() if k not in ['y_pred_proba', 'y_pred']})
    
    # Add confidence intervals
    if name in confidence_intervals:
        for metric_name, ci in confidence_intervals[name].items():
            row[f'{metric_name} CI Lower'] = ci[0]
            row[f'{metric_name} CI Upper'] = ci[1]
    dashboard_data.append(row)

dashboard_df = pd.DataFrame(dashboard_data)
print("\n--- Model Performance Summary ---")
print(dashboard_df.round(4).to_string())

# Plotting Performance Metrics
metrics_to_plot = ['ROC AUC', 'F1-Score', 'Accuracy', 'Precision', 'Recall']
fig, axes = plt.subplots(len(metrics_to_plot), 1, figsize=(12, 5 * len(metrics_to_plot)))
fig.suptitle('Model Performance Comparison', fontsize=16, y=1.02)

for i, metric in enumerate(metrics_to_plot):
    ax = axes[i]
    sns.barplot(x='Model', y=metric, data=dashboard_df, ax=ax, palette='viridis')
    ax.set_title(f'{metric} Comparison')
    ax.set_ylabel(metric)
    ax.set_xlabel('')
    ax.tick_params(axis='x', rotation=45)
    
    # Add confidence interval error bars
    if f'{metric} CI Lower' in dashboard_df.columns:
        for idx, row in dashboard_df.iterrows():
            lower = row.get(f'{metric} CI Lower')
            upper = row.get(f'{metric} CI Upper')
            if pd.notna(lower) and pd.notna(upper):
                # Calculate error bar length from mean to CI bounds
                yerr_lower = row[metric] - lower
                yerr_upper = upper - row[metric]
                ax.errorbar(idx, row[metric], yerr=[[yerr_lower], [yerr_upper]],
                            fmt='none', color='black', capsize=5)

plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.show()

# ROC Curves
plt.figure(figsize=(10, 8))
for name, results in all_model_eval_results.items():
    if 'y_pred_proba' in results and results['ROC AUC'] > 0: # Only plot if probabilities are available and AUC is valid
        fpr, tpr, _ = roc_curve(y_test, results['y_pred_proba'])
        roc_auc = results['ROC AUC']
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.4f})')

plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for All Models')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

print("\n--- Model Comparison Dashboard Complete ---")
print("All tasks completed successfully.")
```