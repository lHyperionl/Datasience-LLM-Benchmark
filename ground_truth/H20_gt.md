```python
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, f1_score, classification_report
import shap
import matplotlib.pyplot as plt
import warnings

# Suppress warnings for cleaner output, especially from sklearn and XGBoost
warnings.filterwarnings('ignore')

# --- 1. Load Data, Parse 'date' ---
print("1. Loading data and parsing 'date'...")
try:
    df = pd.read_csv('synthetic_covid19_data.csv')
except FileNotFoundError:
    print("Error: synthetic_covid19_data.csv not found. Please ensure the file is in the correct directory.")
    exit()

df['date'] = pd.to_datetime(df['date'])
print("Data loaded successfully. Date column parsed.")

# Sort data by location and date for time-series operations
df = df.sort_values(by=['location', 'date']).reset_index(drop=True)

# --- 2. For each 'location', determine its 90th percentile for 'new_deaths' (where new_deaths > 0) and create the binary target variable ---
print("2. Creating target variable 'high_daily_death_increase'...")

# To calculate the 90th percentile, we only consider days with new_deaths > 0.
# Create a temporary column for this purpose, setting 0 or less deaths to NaN.
df['new_deaths_for_percentile'] = df['new_deaths'].apply(lambda x: x if x > 0 else np.nan)

# Calculate the 90th percentile for each location, ignoring NaN values (which correspond to zero deaths)
location_death_percentiles = df.groupby('location')['new_deaths_for_percentile'].quantile(0.90)

# Map the calculated percentile back to the main DataFrame for each row's location
df['location_90th_percentile_deaths'] = df['location'].map(location_death_percentiles)

# Define the target variable:
# 'high_daily_death_increase' is 1 if 'new_deaths' on a given day is greater than
# the location's 90th percentile of non-zero 'new_deaths' AND 'new_deaths' > 0.
# Otherwise, it's 0. If a location's 90th percentile is NaN (e.g., all new_deaths were 0),
# the comparison will result in False, correctly setting the target to 0.
df['high_daily_death_increase'] = ((df['new_deaths'] > df['location_90th_percentile_deaths']) & (df['new_deaths'] > 0)).astype(int)

# Drop the temporary columns used for percentile calculation
df = df.drop(columns=['new_deaths_for_percentile', 'location_90th_percentile_deaths'])

print(f"Target variable created. 'high_daily_death_increase' distribution:\n{df['high_daily_death_increase'].value_counts()}")
print(f"Percentage of 'high_daily_death_increase' events: {df['high_daily_death_increase'].mean() * 100:.2f}%")

# --- 3. Engineer features ---
print("3. Engineering features...")

# Ensure data is sorted by location and date for correct time-series calculations
df = df.sort_values(by=['location', 'date'])

# Rolling 7-day averages and standard deviations for specified columns
rolling_features = ['new_cases', 'hosp_patients', 'icu_patients']
for col in rolling_features:
    # Use transform to apply rolling window operations within each group and broadcast back
    df[f'{col}_rolling_7d_mean'] = df.groupby('location')[col].transform(lambda x: x.rolling(window=7, min_periods=1).mean())
    df[f'{col}_rolling_7d_std'] = df.groupby('location')[col].transform(lambda x: x.rolling(window=7, min_periods=1).std())

# Lag features (1, 3, 7 days) for specified columns
lag_features = ['new_cases', 'reproduction_rate']
for col in lag_features:
    for lag in [1, 3, 7]:
        # Use transform with shift to get lagged values within each group
        df[f'{col}_lag_{lag}d'] = df.groupby('location')[col].transform(lambda x: x.shift(lag))

# Days since first case for each location
# Calculate the difference in days from the earliest date for each location
df['days_since_first_case'] = df.groupby('location')['date'].transform(lambda x: (x - x.min()).dt.days)

print("Features engineered.")

# --- 4. Handle missing values from feature engineering and other relevant columns ---
print("4. Handling missing values...")

# Identify all newly engineered columns
engineered_cols = [col for col in df.columns if any(s in col for s in ['_rolling_', '_lag_', 'days_since_first_case'])]

# For engineered features, fill NaNs.
# First, forward fill (ffill) within each location group to propagate values from previous days.
# Then, fill any remaining NaNs (e.g., at the very beginning of a series where ffill can't apply, or if all values were NaN) with 0.
# This strategy avoids data leakage from the future (like backfill) and provides a sensible default.
for col in engineered_cols:
    df[col] = df.groupby('location')[col].transform(lambda x: x.ffill().fillna(0))

# Also handle NaNs in original numerical features that might be used in the model.
# For simplicity and consistency with the prompt's suggestion for FE NaNs, fill all numerical NaNs with 0.
# A more sophisticated approach might use mean/median imputation for static features like 'median_age'.
numerical_cols_to_fill = df.select_dtypes(include=np.number).columns.tolist()
# Exclude the target variable from this general fillna, as it's already defined
if 'high_daily_death_increase' in numerical_cols_to_fill:
    numerical_cols_to_fill.remove('high_daily_death_increase')

for col in numerical_cols_to_fill:
    if df[col].isnull().any():
        df[col] = df[col].fillna(0) # Fill with 0

print("Missing values handled.")

# --- 5. Select features for modeling ---
print("5. Selecting features for modeling...")

# Calculate 'people_fully_vaccinated' / 'population' ratio
# Handle potential division by zero or NaN population by filling with 0.
df['people_fully_vaccinated_ratio'] = df['people_fully_vaccinated'] / df['population']
df['people_fully_vaccinated_ratio'] = df['people_fully_vaccinated_ratio'].fillna(0) # Fill NaNs from missing data or 0/0
df['people_fully_vaccinated_ratio'] = df['people_fully_vaccinated_ratio'].replace([np.inf, -np.inf], 0) # Replace inf values with 0

# List of all features to be used in the model
selected_features = [
    # Engineered temporal features
    'new_cases_rolling_7d_mean', 'new_cases_rolling_7d_std',
    'hosp_patients_rolling_7d_mean', 'hosp_patients_rolling_7d_std',
    'icu_patients_rolling_7d_mean', 'icu_patients_rolling_7d_std',
    'new_cases_lag_1d', 'new_cases_lag_3d', 'new_cases_lag_7d',
    'reproduction_rate_lag_1d', 'reproduction_rate_lag_3d', 'reproduction_rate_lag_7d',
    'days_since_first_case',
    # Original static/dynamic features
    'stringency_index',
    'people_fully_vaccinated_ratio',
    'median_age',
    'gdp_per_capita',
    'diabetes_prevalence',
    'continent' # This will be one-hot encoded
]

# Verify all selected features exist in the DataFrame
missing_selected_features = [f for f in selected_features if f not in df.columns]
if missing_selected_features:
    print(f"Warning: The following selected features are missing from the DataFrame and will be excluded: {missing_selected_features}")
    selected_features = [f for f in selected_features if f in df.columns] # Filter out missing ones

X = df[selected_features]
y = df['high_daily_death_increase']

print(f"Selected {len(selected_features)} features for modeling.")

# --- 6. Encode 'continent' (one-hot). Scale numerical features. ---
print("6. Preprocessing: Encoding and Scaling...")

# Identify categorical and numerical features for the ColumnTransformer
categorical_features = ['continent']
# Numerical features are all selected features that are not in the categorical list
numerical_features = [col for col in selected_features if col not in categorical_features]

# Create preprocessing pipelines for numerical and categorical features
# Numerical features will be scaled using StandardScaler
numerical_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

# Categorical features will be one-hot encoded
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore')) # 'handle_unknown='ignore'' prevents errors if unseen categories appear in test set
])

# Combine transformers using ColumnTransformer
# This allows different transformations to be applied to different columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough' # Keep any other columns not specified (though none expected here)
)

print("Preprocessing pipeline created.")

# --- 7. Train an XGBoost classifier. Perform hyperparameter tuning using GridSearchCV ---
print("7. Training XGBoost classifier with GridSearchCV...")

# Define the full model pipeline: preprocessing followed by XGBoost classifier
model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                 ('classifier', XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'))])

# Define the parameter grid for GridSearchCV. These are hyperparameters for the XGBoost classifier.
param_grid = {
    'classifier__n_estimators': [100, 200, 300], # Number of boosting rounds
    'classifier__max_depth': [3, 5, 7],         # Maximum depth of a tree
    'classifier__learning_rate': [0.01, 0.05, 0.1] # Step size shrinkage to prevent overfitting
}

# Time-based split: last 20% of data for testing
# It's crucial to sort the DataFrame by date before splitting to maintain temporal order.
df_sorted_for_split = df.sort_values(by='date').reset_index(drop=True)

split_idx = int(len(df_sorted_for_split) * 0.8) # Calculate the index for the 80/20 split
X_train_df = df_sorted_for_split.iloc[:split_idx][selected_features]
X_test_df = df_sorted_for_split.iloc[split_idx:][selected_features]
y_train = df_sorted_for_split.iloc[:split_idx]['high_daily_death_increase']
y_test = df_sorted_for_split.iloc[split_idx:]['high_daily_death_increase']

print(f"Data split into training ({len(X_train_df)} samples) and testing ({len(X_test_df)} samples) sets.")

# Initialize GridSearchCV with the model pipeline, parameter grid, and desired scoring
# 'cv=3' means 3-fold cross-validation on the training set.
# 'scoring='roc_auc'' means ROC-AUC will be the metric optimized during tuning.
# 'n_jobs=-1' uses all available CPU cores.
grid_search = GridSearchCV(model_pipeline, param_grid, cv=3, scoring='roc_auc', verbose=1, n_jobs=-1)

# Fit GridSearchCV on the training data to find the best hyperparameters
print("Starting GridSearchCV fit. This may take some time...")
grid_search.fit(X_train_df, y_train)

print("GridSearchCV complete.")
print(f"Best parameters found: {grid_search.best_params_}")
print(f"Best ROC-AUC score on training data (cross-validation): {grid_search.best_score_:.4f}")

# Get the best model found by GridSearchCV
best_model = grid_search.best_estimator_

# --- 8. Evaluate using ROC-AUC and F1-score on a time-based split ---
print("8. Evaluating the best model on the test set...")

# Predict probabilities for ROC-AUC and class labels for F1-score
y_pred_proba = best_model.predict_proba(X_test_df)[:, 1] # Probability of the positive class
y_pred = best_model.predict(X_test_df) # Predicted class labels

# Calculate evaluation metrics
roc_auc = roc_auc_score(y_test, y_pred_proba)
f1 = f1_score(y_test, y_pred)

print(f"Test Set ROC-AUC: {roc_auc:.4f}")
print(f"Test Set F1-score: {f1:.4f}")
print("\nClassification Report on Test Set:")
print(classification_report(y_test, y_pred))

# --- 9. Use SHAP to explain feature importance for the top 10 features ---
print("9. Explaining feature importance using SHAP...")

# To use SHAP, we need the feature names after preprocessing.
# The ColumnTransformer reorders and renames features.
# Get the feature names from the numerical transformer
processed_numerical_features = numerical_features
# Get the feature names from the one-hot encoder for categorical features
ohe_feature_names = best_model.named_steps['preprocessor'].named_transformers_['cat']['onehot'].get_feature_names_out(categorical_features)
# Combine all feature names
all_processed_feature_names = list(processed_numerical_features) + list(ohe_feature_names)

# Transform the training data using the preprocessor from the best model
# SHAP explainer should be fit on the data format the model was trained on.
X_train_processed = best_model.named_steps['preprocessor'].transform(X_train_df)

# Convert the processed training data into a DataFrame with correct column names for SHAP
X_train_processed_df = pd.DataFrame(X_train_processed, columns=all_processed_feature_names)

# Initialize SHAP TreeExplainer for the XGBoost classifier
explainer = shap.TreeExplainer(best_model.named_steps['classifier'])

# Calculate SHAP values for the processed training data
# Using training data for global feature importance is common.
# For very large datasets, a sample of the data can be used.
print("Calculating SHAP values (this might take a moment)...")
shap_values = explainer.shap_values(X_train_processed_df)

# Plot SHAP summary plot (bar plot for global feature importance) for the top 10 features
print("Generating SHAP summary plot for top 10 features...")
shap.summary_plot(shap_values, X_train_processed_df, plot_type="bar", max_display=10, show=False)
plt.title("SHAP Feature Importance (Top 10)")
plt.tight_layout() # Adjust layout to prevent labels from overlapping
plt.show()

print("\nSHAP analysis complete. Code execution finished.")
```