```python
# Required imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import re

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, RandomizedSearchCV, learning_curve, validation_curve
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif, RFE, SelectFromModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix

# For advanced imputation (Multiple Imputation by Chained Equations)
from sklearn.experimental import enable_iterative_imputer # noqa
from sklearn.impute import IterativeImputer

# For XGBoost
import xgboost as xgb

# For model interpretability
import shap
import lime
import lime.lime_tabular

# For statistical tests
from scipy import stats

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# --- 1. Data Loading and Initial Inspection ---
try:
    df = pd.read_csv('titanic.csv')
except FileNotFoundError:
    print("Error: titanic.csv not found. Please ensure the file is in the current directory.")
    exit()

# Make a copy to avoid modifying the original DataFrame directly
df_original = df.copy()

# --- 2. Comprehensive Data Preprocessing ---

# Custom Transformer for Name and Cabin feature engineering
class FeatureEngineerTransformer(BaseEstimator, TransformerMixin):
    """
    A custom transformer to perform advanced feature engineering from 'Name' and 'Cabin' columns,
    and create new features like FamilySize, IsAlone, and Fare_Per_Person.
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()

        # 2.1. Advanced Feature Engineering from Name
        # Extract Title from Name
        X_copy['Title'] = X_copy['Name'].apply(lambda name: re.search(' ([A-Za-z]+)\.', name).group(1) if re.search(' ([A-Za-z]+)\.', name) else '')
        
        # Group rare titles and standardize common ones
        rare_titles = ['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona', 'Mlle', 'Ms', 'Mme']
        X_copy['Title'] = X_copy['Title'].replace(['Mlle', 'Ms'], 'Miss')
        X_copy['Title'] = X_copy['Title'].replace('Mme', 'Mrs')
        X_copy['Title'] = X_copy['Title'].replace(rare_titles, 'Rare')
        
        # Create 'Is_Married' feature based on 'Mrs' title
        X_copy['Is_Married'] = (X_copy['Title'] == 'Mrs').astype(int)
        
        # Create 'Name_Length' feature
        X_copy['Name_Length'] = X_copy['Name'].apply(len)

        # 2.2. Handling Cabin (extract Deck)
        # Extract the first letter of the Cabin (Deck) or 'Unknown' if missing
        X_copy['Deck'] = X_copy['Cabin'].apply(lambda x: x[0] if pd.notna(x) else 'Unknown')

        # 2.3. FamilySize, IsAlone, Fare_Per_Person
        # Calculate FamilySize
        X_copy['FamilySize'] = X_copy['SibSp'] + X_copy['Parch'] + 1
        
        # Create 'IsAlone' feature
        X_copy['IsAlone'] = (X_copy['FamilySize'] == 1).astype(int)
        
        # Calculate 'Fare_Per_Person', handling potential division by zero or NaN Fare
        X_copy['Fare_Per_Person'] = X_copy['Fare'] / X_copy['FamilySize']
        # Replace infinite values (e.g., from 0/0 or X/0 if FamilySize could be 0, though it's min 1) with NaN
        X_copy['Fare_Per_Person'] = X_copy['Fare_Per_Person'].replace([np.inf, -np.inf], np.nan) 

        # Drop original 'Name', 'Cabin', and 'Ticket' columns as they've been processed or are not directly useful
        X_copy = X_copy.drop(columns=['Name', 'Cabin', 'Ticket'])

        return X_copy

# Define target and features
X = df.drop('Survived', axis=1)
y = df['Survived']

# Split data into training and testing sets, stratified by 'Survived'
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --- Preprocessing Pipeline Definition ---
# Step 1: Apply custom feature engineering
feature_engineer = FeatureEngineerTransformer()
X_train_fe = feature_engineer.fit_transform(X_train)
X_test_fe = feature_engineer.transform(X_test)

# Identify numerical and categorical features after feature engineering
# 'PassengerId' is an identifier and should be dropped before numerical processing
numerical_features = X_train_fe.select_dtypes(include=np.number).columns.tolist()
categorical_features = X_train_fe.select_dtypes(include='object').columns.tolist()

if 'PassengerId' in numerical_features:
    numerical_features.remove('PassengerId')
    X_train_fe = X_train_fe.drop(columns=['PassengerId'])
    X_test_fe = X_test_fe.drop(columns=['PassengerId'])

# Define preprocessing steps for ColumnTransformer
# Numerical pipeline: IterativeImputer for Age/Fare, then StandardScaler, then PolynomialFeatures
numerical_transformer = Pipeline(steps=[
    ('imputer', IterativeImputer(random_state=42)), # Multiple Imputation by Chained Equations
    ('scaler', StandardScaler()), # Feature Scaling
    ('poly', PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)) # Polynomial and interaction terms
])

# Categorical pipeline: SimpleImputer for Embarked (mode), then OneHotEncoder
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')), # Impute missing 'Embarked' with mode
    ('onehot', OneHotEncoder(handle_unknown='ignore')) # One-hot encode categorical features
])

# Create a preprocessor using ColumnTransformer to apply different transformations to different columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough' # Keep any other columns (e.g., if some were not specified)
)

# --- 3. Build and Tune Multiple Base Models ---

# Define models and their parameter distributions for RandomizedSearchCV
models = {
    'RandomForest': {
        'model': RandomForestClassifier(random_state=42),
        'params': {
            'n_estimators': [100, 200, 300],
            'max_features': ['sqrt', 'log2'],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    },
    'GradientBoosting': {
        'model': GradientBoostingClassifier(random_state=42),
        'params': {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7],
            'subsample': [0.7, 0.8, 0.9],
            'max_features': ['sqrt', None]
        }
    },
    'XGBoost': { # XGBoost Classifier
        'model': xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', use_label_encoder=False, random_state=42),
        'params': {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9]
        }
    },
    'SVC': { # Support Vector Machine Classifier
        'model': SVC(probability=True, random_state=42), # probability=True is needed for soft voting in VotingClassifier
        'params': {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto'],
            'kernel': ['rbf', 'linear']
        }
    },
    'MLPClassifier': { # Neural Network Classifier
        'model': MLPClassifier(random_state=42, max_iter=1000), # Increased max_iter for better convergence
        'params': {
            'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
            'activation': ['relu', 'tanh'],
            'solver': ['adam', 'sgd'],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate_init': [0.001, 0.01]
        }
    }
}

best_models = {}
model_results = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) # Stratified K-Fold for cross-validation

print("--- Tuning Base Models ---")
for name, config in models.items():
    print(f"Tuning {name}...")
    # Create a pipeline for each model including preprocessing and the classifier
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', config['model'])])
    
    # Prefix parameters with 'classifier__' for RandomizedSearchCV to target the classifier step in the pipeline
    param_grid = {f'classifier__{k}': v for k, v in config['params'].items()}
    
    # Perform RandomizedSearchCV for hyperparameter tuning
    random_search = RandomizedSearchCV(pipeline, param_distributions=param_grid, n_iter=50, cv=cv, 
                                       scoring='roc_auc', verbose=0, n_jobs=-1, random_state=42)
    # Fit on the original X_train; the pipeline handles feature engineering and preprocessing internally
    random_search.fit(X_train, y_train) 
    
    # Store the best estimator and its performance
    best_models[name] = random_search.best_estimator_
    model_results[name] = {
        'best_score': random_search.best_score_,
        'best_params': random_search.best_params_
    }
    print(f"  Best ROC AUC for {name}: {model_results[name]['best_score']:.4f}")
    print(f"  Best Params for {name}: {model_results[name]['best_params']}")

# --- 4. Create Stacking and Voting Ensemble Classifiers ---

print("\n--- Building Ensemble Models ---")

# Prepare estimators for stacking and voting classifiers using the best tuned base models
estimators_for_ensemble = [
    ('rf', best_models['RandomForest']),
    ('gb', best_models['GradientBoosting']),
    ('xgb', best_models['XGBoost']),
    ('svc', best_models['SVC']),
    ('mlp', best_models['MLPClassifier'])
]

# Stacking Classifier
# Uses a Logistic Regression as the final meta-learner
stacking_classifier = StackingClassifier(
    estimators=estimators_for_ensemble,
    final_estimator=LogisticRegression(random_state=42),
    cv=cv, # Use the same CV strategy for stacking
    n_jobs=-1
)

# Voting Classifier
# Uses 'soft' voting, which averages predicted probabilities
voting_classifier = VotingClassifier(
    estimators=estimators_for_ensemble,
    voting='soft', # 'soft' voting requires probability=True for SVC
    n_jobs=-1
)

# Train and evaluate ensemble models
ensemble_models = {
    'StackingClassifier': stacking_classifier,
    'VotingClassifier': voting_classifier
}

for name, model in ensemble_models.items():
    print(f"Training and evaluating {name}...")
    # The models in estimators_for_ensemble are already full pipelines, so fit on raw X_train, y_train
    model.fit(X_train, y_train)
    
    # Evaluate on the test set for an initial performance check
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    
    model_results[name] = {'accuracy': accuracy, 'roc_auc': roc_auc}
    print(f"  {name} Test Accuracy: {accuracy:.4f}, Test ROC AUC: {roc_auc:.4f}")

# --- 5. Implement Feature Selection using multiple techniques and compare their impact ---

print("\n--- Feature Selection Analysis ---")

# To apply feature selection, we need the preprocessed data and their corresponding feature names.
# Fit the preprocessor on X_train and transform both X_train and X_test.
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Helper function to get feature names after ColumnTransformer and PolynomialFeatures
def get_feature_names_after_preprocessing(preprocessor, numerical_features, categorical_features):
    """
    Extracts feature names after preprocessing steps including ColumnTransformer,
    StandardScaler, PolynomialFeatures, and OneHotEncoder.
    """
    feature_names = []
    
    # Get names from numerical pipeline (scaler and polynomial features)
    num_pipeline = preprocessor.named_transformers_['num']
    poly = num_pipeline.named_steps['poly']
    
    # Get original numerical feature names (before polynomial transformation)
    original_num_features = numerical_features
    
    # Get polynomial feature names based on original numerical features
    poly_feature_names = poly.get_feature_names_out(original_num_features)
    feature_names.extend(poly_feature_names)
    
    # Get names from categorical pipeline (one-hot encoder)
    cat_pipeline = preprocessor.named_transformers_['cat']
    onehot = cat_pipeline.named_steps['onehot']
    
    # Get one-hot encoded feature names
    onehot_feature_names = onehot.get_feature_names_out(categorical_features)
    feature_names.extend(onehot_feature_names)
    
    return feature_names

processed_feature_names = get_feature_names_after_preprocessing(preprocessor, numerical_features, categorical_features)

# Convert processed data back to DataFrame for easier handling with feature names
X_train_processed_df = pd.DataFrame(X_train_processed, columns=processed_feature_names, index=X_train.index)
X_test_processed_df = pd.DataFrame(X_test_processed, columns=processed_feature_names, index=X_test.index)

# Define base models for feature selection techniques
fs_base_model_rfe = LogisticRegression(solver='liblinear', random_state=42) # For RFE
fs_base_model_sfm = RandomForestClassifier(n_estimators=100, random_state=42) # For SelectFromModel

# Dictionary to store feature selection results
fs_results = {}

# 5.1. SelectKBest (ANOVA F-value)
print("  Applying SelectKBest (ANOVA F-value)...")
selector_kbest = SelectKBest(f_classif, k=20) # Select top 20 features based on ANOVA F-value
X_train_kbest = selector_kbest.fit_transform(X_train_processed_df, y_train)
X_test_kbest = selector_kbest.transform(X_test_processed_df)
selected_features_kbest = X_train_processed_df.columns[selector_kbest.get_support()]

# Train a RandomForest model with the selected features to compare impact
model_kbest = RandomForestClassifier(random_state=42)
model_kbest.fit(X_train_kbest, y_train)
y_pred_kbest = model_kbest.predict(X_test_kbest)
y_proba_kbest = model_kbest.predict_proba(X_test_kbest)[:, 1]
fs_results['SelectKBest_RF'] = {'accuracy': accuracy_score(y_test, y_pred_kbest), 'roc_auc': roc_auc_score(y_test, y_proba_kbest)}
print(f"    SelectKBest (k=20) RF Test ROC AUC: {fs_results['SelectKBest_RF']['roc_auc']:.4f}")
print(f"    Selected Features by SelectKBest ({len(selected_features_kbest)}): {list(selected_features_kbest)}")

# 5.2. RFE (Recursive Feature Elimination)
print("  Applying RFE (Recursive Feature Elimination)...")
selector_rfe = RFE(estimator=fs_base_model_rfe, n_features_to_select=20, step=1, verbose=0)
X_train_rfe = selector_rfe.fit_transform(X_train_processed_df, y_train)
X_test_rfe = selector_rfe.transform(X_test_processed_df)
selected_features_rfe = X_train_processed_df.columns[selector_rfe.get_support()]

# Train a RandomForest model with the selected features
model_rfe = RandomForestClassifier(random_state=42)
model_rfe.fit(X_train_rfe, y_train)
y_pred_rfe = model_rfe.predict(X_test_rfe)
y_proba_rfe = model_rfe.predict_proba(X_test_rfe)[:, 1]
fs_results['RFE_RF'] = {'accuracy': accuracy_score(y_test, y_pred_rfe), 'roc_auc': roc_auc_score(y_test, y_proba_rfe)}
print(f"    RFE (k=20) RF Test ROC AUC: {fs_results['RFE_RF']['roc_auc']:.4f}")
print(f"    Selected Features by RFE ({len(selected_features_rfe)}): {list(selected_features_rfe)}")

# 5.3. SelectFromModel (using RandomForest feature importances)
print("  Applying SelectFromModel (RandomForest feature importances)...")
selector_sfm = SelectFromModel(fs_base_model_sfm, prefit=False, threshold='median') # Select features with importance > median
X_train_sfm = selector_sfm.fit_transform(X_train_processed_df, y_train)
X_test_sfm = selector_sfm.transform(X_test_processed_df)
selected_features_sfm = X_train_processed_df.columns[selector_sfm.get_support()]

# Train a RandomForest model with the selected features
model_sfm = RandomForestClassifier(random_state=42)
model_sfm.fit(X_train_sfm, y_train)
y_pred_sfm = model_sfm.predict(X_test_sfm)
y_proba_sfm = model_sfm.predict_proba(X_test_sfm)[:, 1]
fs_results['SelectFromModel_RF'] = {'accuracy': accuracy_score(y_test, y_pred_sfm), 'roc_auc': roc_auc_score(y_test, y_proba_sfm)}
print(f"    SelectFromModel (RF) RF Test ROC AUC: {fs_results['SelectFromModel_RF']['roc_auc']:.4f}")
print(f"    Selected Features by SelectFromModel ({len(selected_features_sfm)}): {list(selected_features_sfm)}")

# --- 6. Perform Extensive Model Evaluation ---

print("\n--- Extensive Model Evaluation ---")

# Consolidate all models (best base models and ensembles) for cross-validation
all_models_for_cv = {
    'RandomForest_Best': best_models['RandomForest'],
    'GradientBoosting_Best': best_models['GradientBoosting'],
    'XGBoost_Best': best_models['XGBoost'],
    'SVC_Best': best_models['SVC'],
    'MLPClassifier_Best': best_models['MLPClassifier'],
    'StackingClassifier': stacking_classifier,
    'VotingClassifier': voting_classifier
}

# Perform Stratified K-Fold Cross-Validation for all models
cv_results = {}
for name, model in all_models_for_cv.items():
    print(f"  Performing 5-fold CV for {name}...")
    # For all models (base and ensemble), they are already pipelines that handle preprocessing.
    # So, fit on the original X_train, y_train.
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
    cv_results[name] = scores
    print(f"    {name} CV ROC AUC: {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")

# Plotting CV results for visual comparison
plt.figure(figsize=(12, 6))
sns.boxplot(data=pd.DataFrame(cv_results))
plt.title('Model ROC AUC Scores Across 5-Fold Cross-Validation')
plt.ylabel('ROC AUC Score')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Learning Curves (for one representative model, e.g., StackingClassifier)
print("\n--- Learning Curve (StackingClassifier) ---")
train_sizes, train_scores, test_scores = learning_curve(
    stacking_classifier, X_train, y_train, cv=cv, n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 10), scoring='roc_auc', random_state=42
)

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.figure(figsize=(10, 6))
plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.1, color="r")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.1, color="g")
plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
plt.title('Learning Curve (StackingClassifier)')
plt.xlabel('Training examples')
plt.ylabel('ROC AUC Score')
plt.legend(loc="best")
plt.grid(True)
plt.show()

# Validation Curve (for one hyperparameter of a base model, e.g., RandomForest n_estimators)
print("\n--- Validation Curve (RandomForest - n_estimators) ---")
# Create a new pipeline for the RandomForest model to plot its validation curve
rf_pipeline_for_vc = Pipeline(steps=[('preprocessor', preprocessor),
                                     ('classifier', RandomForestClassifier(random_state=42))])

param_range = [50, 100, 150, 200, 250, 300] # Range of n_estimators to evaluate
train_scores, test_scores = validation_curve(
    rf_pipeline_for_vc, X_train, y_train, param_name="classifier__n_estimators", param_range=param_range,
    cv=cv, scoring="roc_auc", n_jobs=-1
)

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.figure(figsize=(10, 6))
plt.plot(param_range, train_scores_mean, label="Training score", color="r")
plt.fill_between(param_range, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2, color="r")
plt.plot(param_range, test_scores_mean, label="Cross-validation score", color="g")
plt.fill_between(param_range, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.2, color="g")
plt.title('Validation Curve (RandomForest - n_estimators)')
plt.xlabel('Number of Estimators')
plt.ylabel('ROC AUC Score')
plt.legend(loc="best")
plt.grid(True)
plt.show()

# Statistical Significance Tests (Paired t-test)
print("\n--- Statistical Significance Tests (Paired t-test) ---")
# Compare the best ensemble (StackingClassifier) with other models using their cross-validation scores
model_names = list(cv_results.keys())
best_ensemble_name = 'StackingClassifier' # Assuming StackingClassifier is the best ensemble

if best_ensemble_name in cv_results:
    for name in model_names:
        if name != best_ensemble_name:
            # Perform a paired t-test to compare the performance of two models
            stat, p = stats.ttest_rel(cv_results[best_ensemble_name], cv_results[name])
            print(f"  {best_ensemble_name} vs {name}: t-statistic={stat:.3f}, p-value={p:.3f}")
            if p < 0.05:
                print(f"    Difference is statistically significant (p < 0.05).")
            else:
                print(f"    Difference is NOT statistically significant (p >= 0.05).")
else:
    print(f"  {best_ensemble_name} not found in CV results for statistical testing.")


# --- 7. Apply SHAP or LIME for model interpretability and feature importance analysis ---

print("\n--- Model Interpretability (SHAP & LIME) ---")

# SHAP for XGBoost (as it's a tree-based model, TreeExplainer is efficient)
print("\n  SHAP Explanations (for XGBoost model) ---")
# Extract the trained XGBoost classifier from its pipeline
xgb_model_pipeline = best_models['XGBoost']
xgb_classifier = xgb_model_pipeline.named_steps['classifier']

# Create a SHAP TreeExplainer using the trained XGBoost model
explainer = shap.TreeExplainer(xgb_classifier)
# Calculate SHAP values for the preprocessed test data
shap_values = explainer.shap_values(X_test_processed_df)

# Plot summary (beeswarm) plot of SHAP values
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_test_processed_df, show=False)
plt.title('SHAP Summary Plot (XGBoost)')
plt.tight_layout()
plt.show()

# Plot bar plot of mean absolute SHAP values (feature importance)
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_test_processed_df, plot_type="bar", show=False)
plt.title('SHAP Feature Importance (XGBoost)')
plt.tight_layout()
plt.show()

# Explain a single prediction (e.g., the first instance in the test set)
print("\n  SHAP Explanation for a single prediction (XGBoost) ---")
sample_idx = 0 # Index of the instance to explain
# Initialize Javascript for interactive plots (if running in a Jupyter environment)
shap.initjs() 
# Generate and display a force plot for the selected instance
# Note: force_plot is interactive and might not render in all environments (e.g., plain script output)
# It will typically open in a browser or display inline in a Jupyter notebook.
print(f"  Force plot for instance {sample_idx} (Predicted: {xgb_model_pipeline.predict(X_test.iloc[[sample_idx]])[0]}, Actual: {y_test.iloc[sample_idx]}):")
shap.force_plot(explainer.expected_value, shap_values[sample_idx,:], X_test_processed_df.iloc[sample_idx,:])


# LIME for StackingClassifier (model-agnostic local explanations)
print("\n  LIME Explanations (for StackingClassifier) ---")

# LIME explainer needs training data (preprocessed) and feature names
# Use X_train_processed_df for LIME's training data to learn feature distributions
explainer_lime = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train_processed_df.values, # LIME expects numpy array
    feature_names=processed_feature_names,
    class_names=['Not Survived', 'Survived'],
    mode='classification'
)

# Define a prediction function for LIME that takes raw data and uses the full pipeline
# LIME will generate perturbed samples in the original feature space (raw X_test format)
# and pass them to this function. The stacking_classifier (which is a pipeline)
# will handle the preprocessing internally.
def predict_proba_for_lime(data):
    # 'data' here is expected to be a numpy array of raw input features (like X_test)
    # Convert back to DataFrame with original column names for the pipeline
    return stacking_classifier.predict_proba(pd.DataFrame(data, columns=X_test.columns))

# Explain a single prediction (e.g., the first instance in the test set)
print("\n  LIME Explanation for a single prediction (StackingClassifier) ---")
# LIME expects a numpy array for the instance to explain
instance_to_explain = X_test.iloc[sample_idx].values

# Generate LIME explanation for the selected instance
explanation = explainer_lime.explain_instance(
    data_row=instance_to_explain,
    predict_fn=predict_proba_for_lime,
    num_features=10 # Number of top features to show in the explanation
)

# Print the LIME explanation as a list of (feature, weight) tuples
print(f"  LIME explanation for instance {sample_idx} (Predicted: {stacking_classifier.predict(X_test.iloc[[sample_idx]])[0]}, Actual: {y_test.iloc[sample_idx]}):")
print(explanation.as_list())

# To visualize LIME explanations, you would typically use:
# explanation.show_in_notebook(show_table=True, show_all=False)
# This is best run in a Jupyter notebook environment.

print("\n--- Script Finished ---")
```