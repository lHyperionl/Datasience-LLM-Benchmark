```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE, SelectKBest, f_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split # Not strictly needed for feature selection, but good practice for full ML pipeline

# --- 1. Load Data ---
# Assume 'winequality-red.csv' is in the current directory
try:
    df = pd.read_csv('winequality-red.csv')
    print("Dataset loaded successfully. Initial shape:", df.shape)
except FileNotFoundError:
    print("Error: 'winequality-red.csv' not found. Please ensure the file is in the current directory.")
    exit()

# --- 2. Feature Engineering ---
# Create at least three new meaningful features
# Add a small epsilon to denominators to prevent division by zero
EPSILON = 1e-6

# Feature 1: Acidity Ratio (Fixed Acidity / Volatile Acidity)
# A higher ratio might indicate a more stable or less "vinegary" wine.
df['acidity_ratio'] = df['fixed acidity'] / (df['volatile acidity'] + EPSILON)

# Feature 2: Alcohol-Sulphates Interaction
# Both alcohol and sulphates are known to be important for wine quality.
# Their interaction might capture a synergistic effect.
df['alcohol_sulfates_interaction'] = df['alcohol'] * df['sulphates']

# Feature 3: Free to Total Sulfur Dioxide Ratio
# This ratio indicates the proportion of free SO2, which is important for preventing oxidation.
df['free_to_total_sulfur_dioxide_ratio'] = df['free sulfur dioxide'] / (df['total sulfur dioxide'] + EPSILON)

# Feature 4 (Bonus): Density-Alcohol Difference
# This might capture aspects of the wine's body or sweetness relative to its alcohol content.
df['density_alcohol_diff'] = df['density'] - df['alcohol']

print("\nEngineered features created. New shape:", df.shape)

# Handle potential NaN/inf values introduced by feature engineering (e.g., division by zero)
# Replace infinite values with NaN, then impute NaNs with the median of their respective columns.
df.replace([np.inf, -np.inf], np.nan, inplace=True)
for col in df.columns:
    if df[col].isnull().any():
        median_val = df[col].median()
        df[col].fillna(median_val, inplace=True)
        print(f"Filled NaN values in '{col}' with median: {median_val}")

# Separate features (X) and target (y)
X = df.drop('quality', axis=1)
y = df['quality']

# Scale features - important for many ML algorithms, though RFE with RandomForest and SelectKBest with f_regression
# are less sensitive to it than, say, SVM or PCA. It's good practice.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
print("\nFeatures scaled using StandardScaler.")

# Define the number of top features to select
N_FEATURES_TO_SELECT = 7

# --- 3. Feature Selection - Method 1: Recursive Feature Elimination (RFE) with RandomForestRegressor ---
print(f"\n--- Feature Selection Method 1: RFE with RandomForestRegressor (Top {N_FEATURES_TO_SELECT} features) ---")

# Initialize a RandomForestRegressor as the estimator for RFE
# n_estimators: number of trees in the forest
# random_state: for reproducibility
estimator_rfe = RandomForestRegressor(n_estimators=100, random_state=42)

# Initialize RFE
# n_features_to_select: the number of features to select
# step: the number of features to remove at each iteration
selector_rfe = RFE(estimator=estimator_rfe, n_features_to_select=N_FEATURES_TO_SELECT, step=1)

# Fit RFE to the scaled data
selector_rfe.fit(X_scaled_df, y)

# Get the selected features and their rankings
selected_features_rfe_mask = selector_rfe.support_
selected_features_rfe_names = X_scaled_df.columns[selected_features_rfe_mask].tolist()

# To get importance scores for RFE, we train the estimator on the selected features
# and then extract feature importances.
# First, get the indices of the selected features
selected_feature_indices_rfe = np.where(selected_features_rfe_mask)[0]
# Create a DataFrame with only the selected features
X_selected_rfe = X_scaled_df.iloc[:, selected_feature_indices_rfe]

# Train the RandomForestRegressor on the selected features
estimator_rfe.fit(X_selected_rfe, y)
rfe_feature_importances = estimator_rfe.feature_importances_

# Create a DataFrame for RFE results
rfe_results = pd.DataFrame({
    'Feature': selected_features_rfe_names,
    'Importance (from RF on selected)': rfe_feature_importances
}).sort_values(by='Importance (from RF on selected)', ascending=False).reset_index(drop=True)

print("\nSelected Features by RFE:")
print(rfe_results)


# --- 4. Feature Selection - Method 2: SelectKBest with f_regression ---
print(f"\n--- Feature Selection Method 2: SelectKBest with f_regression (Top {N_FEATURES_TO_SELECT} features) ---")

# Initialize SelectKBest with f_regression score function
# f_regression: calculates the F-value for regression tasks, estimating the degree of linear dependency
selector_kbest = SelectKBest(score_func=f_regression, k=N_FEATURES_TO_SELECT)

# Fit SelectKBest to the scaled data
selector_kbest.fit(X_scaled_df, y)

# Get the selected features and their scores
selected_features_kbest_mask = selector_kbest.get_support()
selected_features_kbest_names = X_scaled_df.columns[selected_features_kbest_mask].tolist()
kbest_scores = selector_kbest.scores_[selected_features_kbest_mask]

# Create a DataFrame for SelectKBest results
kbest_results = pd.DataFrame({
    'Feature': selected_features_kbest_names,
    'Score (f_regression)': kbest_scores
}).sort_values(by='Score (f_regression)', ascending=False).reset_index(drop=True)

print("\nSelected Features by SelectKBest:")
print(kbest_results)

# --- 5. Compare and Contrast the Feature Sets ---
print("\n--- Comparison of Selected Feature Sets ---")

set_rfe = set(selected_features_rfe_names)
set_kbest = set(selected_features_kbest_names)

common_features = list(set_rfe.intersection(set_kbest))
unique_to_rfe = list(set_rfe.difference(set_kbest))
unique_to_kbest = list(set_kbest.difference(set_rfe))

print(f"\nFeatures selected by RFE ({len(set_rfe)}): {selected_features_rfe_names}")
print(f"Features selected by SelectKBest ({len(set_kbest)}): {selected_features_kbest_names}")

print(f"\nCommon features ({len(common_features)}): {common_features}")
print(f"Features unique to RFE ({len(unique_to_rfe)}): {unique_to_rfe}")
print(f"Features unique to SelectKBest ({len(unique_to_kbest)}): {unique_to_kbest}")

print("\n--- Analysis of Differences ---")
print("RFE (Recursive Feature Elimination) with RandomForest is a wrapper method. It iteratively removes features "
      "based on model performance (or feature importance from the model). It considers feature interactions "
      "implicitly through the RandomForest model.")
print("\nSelectKBest with f_regression is a filter method. It selects features based on their individual "
      "statistical relationship with the target variable (univariate analysis). It does not consider "
      "interactions between features or their combined effect on model performance.")
print("\nDifferences in selected features often arise because RFE considers multivariate relationships and "
      "model performance, while SelectKBest considers only univariate statistical correlation. "
      "Engineered features like 'alcohol_sulfates_interaction' or 'density_alcohol_diff' might be "
      "highly ranked by RFE if they contribute significantly to the RandomForest model's predictive power, "
      "whereas SelectKBest might rank them based purely on their linear correlation with 'quality'.")
print("For instance, 'alcohol' and 'sulphates' are often highly correlated with quality individually, "
      "so their interaction might also show up in SelectKBest. However, 'volatile acidity' and 'alcohol' "
      "are consistently important across many wine quality studies, which is reflected here.")

# --- 6. Display Final Selected Features with Importance/Rankings ---
print("\n--- Final Selected Features and Their Scores/Rankings ---")

print("\nFeatures selected by RFE (ranked by importance from RandomForest trained on selected features):")
print(rfe_results.to_string(index=False))

print("\nFeatures selected by SelectKBest (ranked by f_regression score):")
print(kbest_results.to_string(index=False))
```