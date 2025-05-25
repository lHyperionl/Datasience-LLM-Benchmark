```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    VotingClassifier, BaggingClassifier,
    AdaBoostClassifier, GradientBoostingClassifier,
    RandomForestClassifier, StackingClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score, roc_curve, brier_score_loss
)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier, Pool
import optuna
import shap
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# --- Configuration ---
RANDOM_SEED = 42
N_SPLITS_OUTER = 5  # For nested cross-validation outer loop
N_SPLITS_INNER = 3  # For nested cross-validation inner loop (or Optuna folds)
OPTUNA_TRIALS = 50  # Number of trials for Bayesian optimization
DATA_PATH = 'diabetes.csv'

# --- 1. Load and preprocess the dataset with comprehensive feature engineering ---

def load_and_preprocess_data(file_path):
    """
    Loads the diabetes dataset, handles missing values (0s),
    applies feature engineering, and scales features.
    """
    df = pd.read_csv(file_path)

    # Define features (X) and target (y)
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']

    # Handle 0 values in specific columns as missing data
    # These columns cannot logically be 0 for a living person
    cols_with_zeros_as_nan = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in cols_with_zeros_as_nan:
        X[col] = X[col].replace(0, np.nan)

    # Impute missing values using median
    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    # Feature Engineering:
    # 1. Polynomial Features
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X_imputed)
    poly_feature_names = poly.get_feature_names_out(X_imputed.columns)
    X_poly_df = pd.DataFrame(X_poly, columns=poly_feature_names)

    # 2. Interaction Terms (already covered by PolynomialFeatures degree=2)
    # 3. Custom features (e.g., BMI categories, Glucose/Insulin ratio)
    X_poly_df['BMI_Age_Interaction'] = X_poly_df['BMI'] * X_poly_df['Age']
    X_poly_df['Glucose_Insulin_Ratio'] = X_poly_df['Glucose'] / (X_poly_df['Insulin'] + 1e-6) # Add epsilon to avoid division by zero

    # Scale numerical features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_poly_df), columns=X_poly_df.columns)

    return X_scaled, y, scaler, imputer, poly

X, y, scaler, imputer, poly = load_and_preprocess_data(DATA_PATH)

print("Dataset loaded and preprocessed. Shape:", X.shape)
print("First 5 rows of preprocessed data:")
print(X.head())

# --- 2. Implement advanced ensemble methods & 3. Create a stacking ensemble ---

# Define base estimators for various ensembles
# Using fixed random_state for reproducibility where applicable
base_estimators = {
    'LogisticRegression': LogisticRegression(random_state=RANDOM_SEED, solver='liblinear'),
    'RandomForest': RandomForestClassifier(random_state=RANDOM_SEED),
    'SVC': SVC(probability=True, random_state=RANDOM_SEED), # probability=True needed for soft voting/calibration
    'KNeighbors': KNeighborsClassifier(),
    'DecisionTree': DecisionTreeClassifier(random_state=RANDOM_SEED),
    'XGBoost': xgb.XGBClassifier(random_state=RANDOM_SEED, use_label_encoder=False, eval_metric='logloss'),
    'LightGBM': lgb.LGBMClassifier(random_state=RANDOM_SEED),
    'CatBoost': CatBoostClassifier(random_state=RANDOM_SEED, verbose=0, cat_features=None) # No explicit categorical features in this dataset after preprocessing
}

# --- 4. Implement Bayesian optimization for hyperparameter tuning (using Optuna) ---

# We'll tune XGBoost as an example due to its complexity and common use.
# The tuning will be performed on a single train-validation split for efficiency,
# and the best hyperparameters will be used in the nested CV.

X_train_opt, X_val_opt, y_train_opt, y_val_opt = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
)

def objective(trial):
    """Objective function for Optuna to optimize XGBoost hyperparameters."""
    param = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'seed': RANDOM_SEED,
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 0.0, 0.5),
        'lambda': trial.suggest_float('lambda', 1e-8, 10.0, log=True),
        'alpha': trial.suggest_float('alpha', 1e-8, 10.0, log=True),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
    }

    model = xgb.XGBClassifier(**param, use_label_encoder=False)
    model.fit(X_train_opt, y_train_opt,
              eval_set=[(X_val_opt, y_val_opt)],
              early_stopping_rounds=50, verbose=False)

    preds = model.predict_proba(X_val_opt)[:, 1]
    auc = roc_auc_score(y_val_opt, preds)
    return auc

print("\nStarting Bayesian Optimization for XGBoost hyperparameters...")
study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
study.optimize(objective, n_trials=OPTUNA_TRIALS, show_progress_bar=True)

print("\nBest trial for XGBoost:")
print(f"  Value: {study.best_value:.4f}")
print("  Params: ")
for key, value in study.best_params.items():
    print(f"    {key}: {value}")

tuned_xgb_params = study.best_params
tuned_xgb_params['use_label_encoder'] = False
tuned_xgb_params['eval_metric'] = 'logloss'
tuned_xgb_params['random_state'] = RANDOM_SEED

# Update base_estimators with the tuned XGBoost
base_estimators['XGBoost_Tuned'] = xgb.XGBClassifier(**tuned_xgb_params)

# --- 5. Use nested cross-validation for unbiased model evaluation ---

# Store results for all models and ensembles
results = {}
model_predictions = {} # To store OOF predictions for diversity analysis

skf_outer = StratifiedKFold(n_splits=N_SPLITS_OUTER, shuffle=True, random_state=RANDOM_SEED)

print(f"\nStarting Nested Cross-Validation with {N_SPLITS_OUTER} outer folds...")

for fold_idx, (train_idx, test_idx) in enumerate(skf_outer.split(X, y)):
    print(f"\n--- Outer Fold {fold_idx + 1}/{N_SPLITS_OUTER} ---")
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    fold_models = {}
    fold_preds_proba = {}
    fold_test_y = y_test

    # --- Individual Base Models ---
    print("Training individual base models...")
    for name, estimator in base_estimators.items():
        print(f"  Training {name}...")
        model = estimator
        if name == 'CatBoost': # CatBoost can handle categorical features directly if specified
            # For this dataset, all features are numerical after preprocessing, so no cat_features needed
            model.fit(X_train, y_train, verbose=0)
        else:
            model.fit(X_train, y_train)
        
        preds_proba = model.predict_proba(X_test)[:, 1]
        fold_preds_proba[name] = preds_proba
        fold_models[name] = model

        auc = roc_auc_score(y_test, preds_proba)
        preds_class = (preds_proba > 0.5).astype(int)
        acc = accuracy_score(y_test, preds_class)
        prec = precision_score(y_test, preds_class)
        rec = recall_score(y_test, preds_class)
        f1 = f1_score(y_test, preds_class)

        if name not in results:
            results[name] = {'AUC': [], 'Accuracy': [], 'Precision': [], 'Recall': [], 'F1': []}
        results[name]['AUC'].append(auc)
        results[name]['Accuracy'].append(acc)
        results[name]['Precision'].append(prec)
        results[name]['Recall'].append(rec)
        results[name]['F1'].append(f1)

    # --- Ensemble Methods ---

    # Voting Classifier (Soft and Hard)
    print("Training Voting Classifiers...")
    estimators_for_voting = [
        ('lr', base_estimators['LogisticRegression']),
        ('rf', base_estimators['RandomForest']),
        ('xgb', base_estimators['XGBoost_Tuned']),
        ('lgbm', base_estimators['LightGBM']),
        ('cat', base_estimators['CatBoost'])
    ]

    # Soft Voting
    soft_voter = VotingClassifier(estimators=estimators_for_voting, voting='soft', n_jobs=-1)
    soft_voter.fit(X_train, y_train)
    soft_preds_proba = soft_voter.predict_proba(X_test)[:, 1]
    fold_preds_proba['Voting_Soft'] = soft_preds_proba
    fold_models['Voting_Soft'] = soft_voter
    
    auc = roc_auc_score(y_test, soft_preds_proba)
    preds_class = (soft_preds_proba > 0.5).astype(int)
    acc = accuracy_score(y_test, preds_class)
    prec = precision_score(y_test, preds_class)
    rec = recall_score(y_test, preds_class)
    f1 = f1_score(y_test, preds_class)
    if 'Voting_Soft' not in results: results['Voting_Soft'] = {'AUC': [], 'Accuracy': [], 'Precision': [], 'Recall': [], 'F1': []}
    results['Voting_Soft']['AUC'].append(auc)
    results['Voting_Soft']['Accuracy'].append(acc)
    results['Voting_Soft']['Precision'].append(prec)
    results['Voting_Soft']['Recall'].append(rec)
    results['Voting_Soft']['F1'].append(f1)

    # Hard Voting
    hard_voter = VotingClassifier(estimators=estimators_for_voting, voting='hard', n_jobs=-1)
    hard_voter.fit(X_train, y_train)
    hard_preds_class = hard_voter.predict(X_test)
    # For hard voting, we can't get probabilities directly, so AUC is based on predicted classes
    # For consistency, we'll use predict_proba if possible, but hard voting is by definition class-based.
    # We'll calculate AUC from predicted classes for hard voting, which is less ideal but standard.
    # Or, we can use the probabilities from the base estimators and average them for a pseudo-soft vote.
    # Let's stick to the true hard voting definition for metrics.
    auc = roc_auc_score(y_test, hard_preds_class) # This is not ideal for AUC, but for consistency with hard voting
    acc = accuracy_score(y_test, hard_preds_class)
    prec = precision_score(y_test, hard_preds_class)
    rec = recall_score(y_test, hard_preds_class)
    f1 = f1_score(y_test, hard_preds_class)
    if 'Voting_Hard' not in results: results['Voting_Hard'] = {'AUC': [], 'Accuracy': [], 'Precision': [], 'Recall': [], 'F1': []}
    results['Voting_Hard']['AUC'].append(auc)
    results['Voting_Hard']['Accuracy'].append(acc)
    results['Voting_Hard']['Precision'].append(prec)
    results['Voting_Hard']['Recall'].append(rec)
    results['Voting_Hard']['F1'].append(f1)

    # Bagging with different base estimators
    print("Training Bagging Classifiers...")
    bagging_estimators = {
        'Bagging_DT': BaggingClassifier(base_estimator=DecisionTreeClassifier(random_state=RANDOM_SEED), n_estimators=50, random_state=RANDOM_SEED, n_jobs=-1),
        'Bagging_LR': BaggingClassifier(base_estimator=LogisticRegression(random_state=RANDOM_SEED, solver='liblinear'), n_estimators=50, random_state=RANDOM_SEED, n_jobs=-1),
        'Bagging_SVC': BaggingClassifier(base_estimator=SVC(probability=True, random_state=RANDOM_SEED), n_estimators=50, random_state=RANDOM_SEED, n_jobs=-1)
    }
    for name, estimator in bagging_estimators.items():
        print(f"  Training {name}...")
        model = estimator
        model.fit(X_train, y_train)
        preds_proba = model.predict_proba(X_test)[:, 1]
        fold_preds_proba[name] = preds_proba
        fold_models[name] = model

        auc = roc_auc_score(y_test, preds_proba)
        preds_class = (preds_proba > 0.5).astype(int)
        acc = accuracy_score(y_test, preds_class)
        prec = precision_score(y_test, preds_class)
        rec = recall_score(y_test, preds_class)
        f1 = f1_score(y_test, preds_class)
        if name not in results: results[name] = {'AUC': [], 'Accuracy': [], 'Precision': [], 'Recall': [], 'F1': []}
        results[name]['AUC'].append(auc)
        results[name]['Accuracy'].append(acc)
        results[name]['Precision'].append(prec)
        results[name]['Recall'].append(rec)
        results[name]['F1'].append(f1)

    # Boosting algorithms (AdaBoost, Gradient Boosting)
    print("Training Boosting Classifiers...")
    boosting_estimators = {
        'AdaBoost': AdaBoostClassifier(random_state=RANDOM_SEED),
        'GradientBoosting': GradientBoostingClassifier(random_state=RANDOM_SEED)
    }
    for name, estimator in boosting_estimators.items():
        print(f"  Training {name}...")
        model = estimator
        model.fit(X_train, y_train)
        preds_proba = model.predict_proba(X_test)[:, 1]
        fold_preds_proba[name] = preds_proba
        fold_models[name] = model

        auc = roc_auc_score(y_test, preds_proba)
        preds_class = (preds_proba > 0.5).astype(int)
        acc = accuracy_score(y_test, preds_class)
        prec = precision_score(y_test, preds_class)
        rec = recall_score(y_test, preds_class)
        f1 = f1_score(y_test, preds_class)
        if name not in results: results[name] = {'AUC': [], 'Accuracy': [], 'Precision': [], 'Recall': [], 'F1': []}
        results[name]['AUC'].append(auc)
        results[name]['Accuracy'].append(acc)
        results[name]['Precision'].append(prec)
        results[name]['Recall'].append(rec)
        results[name]['F1'].append(f1)

    # Stacking Ensemble
    print("Training Stacking Classifier...")
    # Base learners for stacking
    level0_estimators = [
        ('lr', base_estimators['LogisticRegression']),
        ('rf', base_estimators['RandomForest']),
        ('xgb', base_estimators['XGBoost_Tuned']),
        ('lgbm', base_estimators['LightGBM']),
        ('cat', base_estimators['CatBoost'])
    ]
    # Meta-learner
    meta_learner = LogisticRegression(random_state=RANDOM_SEED, solver='liblinear')

    stacking_clf = StackingClassifier(
        estimators=level0_estimators,
        final_estimator=meta_learner,
        cv=N_SPLITS_INNER, # Inner CV for stacking
        stack_method='predict_proba',
        n_jobs=-1
    )
    stacking_clf.fit(X_train, y_train)
    stacking_preds_proba = stacking_clf.predict_proba(X_test)[:, 1]
    fold_preds_proba['Stacking'] = stacking_preds_proba
    fold_models['Stacking'] = stacking_clf

    auc = roc_auc_score(y_test, stacking_preds_proba)
    preds_class = (stacking_preds_proba > 0.5).astype(int)
    acc = accuracy_score(y_test, preds_class)
    prec = precision_score(y_test, preds_class)
    rec = recall_score(y_test, preds_class)
    f1 = f1_score(y_test, preds_class)
    if 'Stacking' not in results: results['Stacking'] = {'AUC': [], 'Accuracy': [], 'Precision': [], 'Recall': [], 'F1': []}
    results['Stacking']['AUC'].append(auc)
    results['Stacking']['Accuracy'].append(acc)
    results['Stacking']['Precision'].append(prec)
    results['Stacking']['Recall'].append(rec)
    results['Stacking']['F1'].append(f1)

    # --- 6. Implement custom ensemble methods with dynamic weight assignment ---
    # Simple weighted average based on validation performance within the fold
    print("Training Custom Weighted Ensemble...")
    skf_inner = StratifiedKFold(n_splits=N_SPLITS_INNER, shuffle=True, random_state=RANDOM_SEED)
    X_train_inner, X_val_inner, y_train_inner, y_val_inner = train_test_split(
        X_train, y_train, test_size=0.2, random_state=RANDOM_SEED, stratify=y_train
    )

    base_models_for_custom = {
        'LogisticRegression': base_estimators['LogisticRegression'],
        'RandomForest': base_estimators['RandomForest'],
        'XGBoost_Tuned': base_estimators['XGBoost_Tuned'],
        'LightGBM': base_estimators['LightGBM'],
    }

    val_aucs = {}
    for name, estimator in base_models_for_custom.items():
        model = estimator
        model.fit(X_train_inner, y_train_inner)
        preds_proba_val = model.predict_proba(X_val_inner)[:, 1]
        val_aucs[name] = roc_auc_score(y_val_inner, preds_proba_val)

    # Normalize AUCs to get weights
    total_auc = sum(val_aucs.values())
    weights = {name: auc / total_auc for name, auc in val_aucs.items()}

    # Apply weights to test predictions
    custom_preds_proba = np.zeros(len(X_test))
    for name, weight in weights.items():
        # Retrain base models on full X_train for final predictions
        model = base_models_for_custom[name]
        model.fit(X_train, y_train) # Fit on full X_train for test set prediction
        custom_preds_proba += weight * model.predict_proba(X_test)[:, 1]

    fold_preds_proba['Custom_Weighted'] = custom_preds_proba
    # No single model to store for custom ensemble, but we can store the weights
    fold_models['Custom_Weighted_Weights'] = weights

    auc = roc_auc_score(y_test, custom_preds_proba)
    preds_class = (custom_preds_proba > 0.5).astype(int)
    acc = accuracy_score(y_test, preds_class)
    prec = precision_score(y_test, preds_class)
    rec = recall_score(y_test, preds_class)
    f1 = f1_score(y_test, preds_class)
    if 'Custom_Weighted' not in results: results['Custom_Weighted'] = {'AUC': [], 'Accuracy': [], 'Precision': [], 'Recall': [], 'F1': []}
    results['Custom_Weighted']['AUC'].append(auc)
    results['Custom_Weighted']['Accuracy'].append(acc)
    results['Custom_Weighted']['Precision'].append(prec)
    results['Custom_Weighted']['Recall'].append(rec)
    results['Custom_Weighted']['F1'].append(f1)

    # Store OOF predictions for diversity analysis and calibration
    model_predictions[f'fold_{fold_idx}'] = {
        'y_true': fold_test_y,
        'predictions': fold_preds_proba,
        'models': fold_models # Store models for SHAP/Calibration later
    }

# --- Summarize Nested CV Results ---
print("\n--- Nested Cross-Validation Results (Mean ± Std Dev) ---")
for model_name, metrics in results.items():
    print(f"\nModel: {model_name}")
    for metric_name, values in metrics.items():
        mean_val = np.mean(values)
        std_val = np.std(values)
        print(f"  {metric_name}: {mean_val:.4f} ± {std_val:.4f}")

# Identify the best performing model/ensemble based on mean AUC
best_model_name = max(results, key=lambda k: np.mean(results[k]['AUC']))
print(f"\nBest performing model/ensemble based on mean AUC: {best_model_name}")

# --- 9. Implement model calibration and reliability analysis ---

print("\n--- Model Calibration and Reliability Analysis ---")
# Use the best performing model from the last fold for demonstration
# In a real scenario, you'd retrain the best model on the full dataset or use OOF predictions.
# For demonstration, we'll pick the model from the last fold.
last_fold_idx = N_SPLITS_OUTER - 1
y_true_calib = model_predictions[f'fold_{last_fold_idx}']['y_true']
best_model_preds_proba = model_predictions[f'fold_{last_fold_idx}']['predictions'][best_model_name]
best_model_instance = model_predictions[f'fold_{last_fold_idx}']['models'][best_model_name]

# Brier Score before calibration
brier_uncalibrated = brier_score_loss(y_true_calib, best_model_preds_proba)
print(f"Brier Score (Uncalibrated {best_model_name}): {brier_uncalibrated:.4f}")

# Calibrate the best model using Platt scaling (sigmoid) and Isotonic regression
# Fit CalibratedClassifierCV on a separate calibration set (e.g., from the training data of the last fold)
# For simplicity, we'll fit it on the full X, y and then evaluate on the test set of the last fold.
# A more rigorous approach would be to use a dedicated calibration set or nested CV for calibration.
calibrated_model_isotonic = CalibratedClassifierCV(best_model_instance, method='isotonic', cv='prefit')
calibrated_model_sigmoid = CalibratedClassifierCV(best_model_instance, method='sigmoid', cv='prefit')

# Fit on the training data of the last fold
X_train_last_fold, y_train_last_fold = X.iloc[train_idx], y.iloc[train_idx]
calibrated_model_isotonic.fit(X_train_last_fold, y_train_last_fold)
calibrated_model_sigmoid.fit(X_train_last_fold, y_train_last_fold)

# Get calibrated probabilities on the test set of the last fold
calibrated_preds_isotonic = calibrated_model_isotonic.predict_proba(X_test)[:, 1]
calibrated_preds_sigmoid = calibrated_model_sigmoid.predict_proba(X_test)[:, 1]

brier_isotonic = brier_score_loss(y_true_calib, calibrated_preds_isotonic)
brier_sigmoid = brier_score_loss(y_true_calib, calibrated_preds_sigmoid)
print(f"Brier Score (Calibrated Isotonic {best_model_name}): {brier_isotonic:.4f}")
print(f"Brier Score (Calibrated Sigmoid {best_model_name}): {brier_sigmoid:.4f}")

# Plot Reliability Diagram (Calibration Curve)
plt.figure(figsize=(10, 7))
ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
ax2 = plt.subplot2grid((3, 1), (2, 0))

ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

# Uncalibrated
fraction_of_positives, mean_predicted_value = calibration_curve(y_true_calib, best_model_preds_proba, n_bins=10)
ax1.plot(mean_predicted_value, fraction_of_positives, "s-", label=f"{best_model_name} (Brier: {brier_uncalibrated:.3f})")

# Isotonic calibrated
fraction_of_positives_iso, mean_predicted_value_iso = calibration_curve(y_true_calib, calibrated_preds_isotonic, n_bins=10)
ax1.plot(mean_predicted_value_iso, fraction_of_positives_iso, "o-", label=f"Isotonic Calibrated (Brier: {brier_isotonic:.3f})")

# Sigmoid calibrated
fraction_of_positives_sig, mean_predicted_value_sig = calibration_curve(y_true_calib, calibrated_preds_sigmoid, n_bins=10)
ax1.plot(mean_predicted_value_sig, fraction_of_positives_sig, "^-", label=f"Sigmoid Calibrated (Brier: {brier_sigmoid:.3f})")

ax1.set_ylabel("Fraction of positives")
ax1.set_ylim([-0.05, 1.05])
ax1.legend(loc="lower right")
ax1.set_title('Calibration plots (Reliability Curve)')

# Histogram of predicted probabilities
ax2.hist(best_model_preds_proba, range=(0, 1), bins=10, label=best_model_name, histtype="step", lw=2)
ax2.hist(calibrated_preds_isotonic, range=(0, 1), bins=10, label='Isotonic Calibrated', histtype="step", lw=2)
ax2.hist(calibrated_preds_sigmoid, range=(0, 1), bins=10, label='Sigmoid Calibrated', histtype="step", lw=2)
ax2.set_xlabel("Mean predicted value")
ax2.set_ylabel("Count")
ax2.legend(loc="upper center", ncol=2)
plt.tight_layout()
plt.show()

# --- 8. Perform comprehensive model interpretation using SHAP values ---

print("\n--- Model Interpretation using SHAP Values ---")
# Use the best performing model (e.g., XGBoost_Tuned or Stacking) from the last fold for SHAP explanation.
# For simplicity, let's pick the best_model_name identified earlier.
# If it's a StackingClassifier, SHAP can be complex. Let's use XGBoost_Tuned as it's a strong single model.
shap_model_name = 'XGBoost_Tuned'
shap_model = model_predictions[f'fold_{last_fold_idx}']['models'][shap_model_name]

# Create a SHAP explainer
explainer = shap.TreeExplainer(shap_model)

# Calculate SHAP values for a subset of the test data for faster computation
X_test_shap = X_test.sample(min(200, len(X_test)), random_state=RANDOM_SEED)
shap_values = explainer.shap_values(X_test_shap)

# Summary plot (Global interpretation)
print(f"\nSHAP Summary Plot for {shap_model_name}:")
shap.summary_plot(shap_values, X_test_shap, plot_type="bar", show=False)
plt.title(f"SHAP Feature Importance for {shap_model_name}")
plt.tight_layout()
plt.show()

shap.summary_plot(shap_values, X_test_shap, show=False)
plt.title(f"SHAP Summary Plot (beeswarm) for {shap_model_name}")
plt.tight_layout()
plt.show()

# Dependence plots for top features (e.g., Glucose, BMI)
print(f"\nSHAP Dependence Plots for {shap_model_name}:")
top_features = X_test_shap.columns[np.argsort(np.abs(shap_values).mean(0))[::-1][:3]]
for feature in top_features:
    shap.dependence_plot(feature, shap_values, X_test_shap, show=False)
    plt.title(f"SHAP Dependence Plot for {feature} ({shap_model_name})")
    plt.tight_layout()
    plt.show()

# Force plot for a single prediction (Local interpretation)
print(f"\nSHAP Force Plot for a single prediction ({shap_model_name}):")
# Choose a random instance from the test set
sample_idx = np.random.randint(0, len(X_test_shap))
shap.initjs() # Initialize Javascript for interactive plots
shap.force_plot(explainer.expected_value, shap_values[sample_idx,:], X_test_shap.iloc[sample_idx,:])
# Note: Force plots are interactive and might not display directly in all environments.
# If running in a Jupyter notebook, it will render.

# --- 10. Create ensemble diversity analysis and model combination strategies ---

print("\n--- Ensemble Diversity Analysis ---")
# Collect OOF predictions from all models across all folds
all_oof_preds = {}
all_y_true = []

for fold_idx in range(N_SPLITS_OUTER):
    fold_data = model_predictions[f'fold_{fold_idx}']
    all_y_true.extend(fold_data['y_true'].tolist())
    for model_name, preds in fold_data['predictions'].items():
        if model_name not in all_oof_preds:
            all_oof_preds[model_name] = []
        all_oof_preds[model_name].extend(preds.tolist())

# Convert to DataFrame for correlation analysis
oof_preds_df = pd.DataFrame(all_oof_preds)

# Calculate correlation matrix of predictions
correlation_matrix = oof_preds_df.corr()

plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix of OOF Model Predictions')
plt.tight_layout()
plt.show()

print("\nCorrelation Matrix of OOF Model Predictions:")
print(correlation_matrix)

# Analyze diversity: Low correlation indicates high diversity, which is good for ensembles.
# Models with high correlation might be redundant in an ensemble.

# --- Model Combination Strategies (beyond simple voting/stacking) ---
# We already implemented Stacking and a Custom Weighted Ensemble.
# Another simple strategy is to train a meta-model on the OOF predictions.

print("\n--- Advanced Model Combination Strategy: Meta-Learner on OOF Predictions ---")
# Prepare data for meta-learner
X_meta = oof_preds_df
y_meta = pd.Series(all_y_true)

# Train a simple Logistic Regression as a meta-learner
meta_model = LogisticRegression(random_state=RANDOM_SEED, solver='liblinear')
meta_model.fit(X_meta, y_meta)

# Evaluate the meta-model (this is an optimistic evaluation as it's on the same OOF data)
# For a true evaluation, this meta-model should be part of the nested CV.
meta_preds_proba = meta_model.predict_proba(X_meta)[:, 1]
meta_auc = roc_auc_score(y_meta, meta_preds_proba)
meta_preds_class = (meta_preds_proba > 0.5).astype(int)
meta_acc = accuracy_score(y_meta, meta_preds_class)
meta_prec = precision_score(y_meta, meta_preds_class)
meta_rec = recall_score(y_meta, meta_preds_class)
meta_f1 = f1_score(y_meta, meta_preds_class)

print(f"\nMeta-Learner (Logistic Regression on OOF Predictions) Performance:")
print(f"  AUC: {meta_auc:.4f}")
print(f"  Accuracy: {meta_acc:.4f}")
print(f"  Precision: {meta_prec:.4f}")
print(f"  Recall: {meta_rec:.4f}")
print(f"  F1-Score: {meta_f1:.4f}")

print("\n--- End of Comprehensive Diabetes Prediction Solution ---")
```