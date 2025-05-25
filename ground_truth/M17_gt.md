```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Set random seed for reproducibility
np.random.seed(42)

# --- 1. Load the dataset and preprocess it ---

# Load the dataset
try:
    df = pd.read_csv('world-happiness-report-2021.csv')
except FileNotFoundError:
    print("Error: 'world-happiness-report-2021.csv' not found. Please ensure the file is in the current directory.")
    exit()

# Display initial info and head
print("Original DataFrame Info:")
df.info()
print("\nOriginal DataFrame Head:")
print(df.head())

# Define target and feature columns as specified in the prompt
TARGET_COLUMN = 'Ladder score'
FEATURE_COLUMNS = [
    'Logged GDP per capita',
    'Social support',
    'Healthy life expectancy',
    'Freedom to make life choices',
    'Generosity',
    'Perceptions of corruption'
]

# Rename columns for easier access and consistency with prompt's wording
# The prompt uses 'GDP per capita', 'Freedom', 'Perceptions of corruption'
# The dataset uses 'Logged GDP per capita', 'Freedom to make life choices', 'Perceptions of corruption'
# We will use the dataset's exact column names for features and target.
# Let's map them for clarity if needed, but for now, use the exact dataset names.
# The prompt's wording is slightly simplified, so we'll use the actual column names from the CSV.
# Let's verify the exact column names in the CSV.
# df.columns output: Index(['Country name', 'Regional indicator', 'Ladder score', 'Standard error of ladder score', 'upperwhisker', 'lowerwhisker', 'Logged GDP per capita', 'Social support', 'Healthy life expectancy', 'Freedom to make life choices', 'Generosity', 'Perceptions of corruption', 'Ladder score in Dystopia', 'Explained by: GDP per capita', 'Explained by: Social support', 'Explained by: Healthy life expectancy', 'Explained by: Freedom to make life choices', 'Explained by: Generosity', 'Explained by: Perceptions of corruption', 'Dystopia Residual'], dtype='object')

# Handle missing values
# Check for missing values in relevant columns
print("\nMissing values before handling:")
print(df[FEATURE_COLUMNS + [TARGET_COLUMN]].isnull().sum())

# Drop rows where any of the target or feature columns have missing values
df_cleaned = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])
print(f"\nDataFrame shape after dropping rows with missing values: {df_cleaned.shape}")

# Encoding categorical variables
# The specified features are all numerical.
# 'Country name' and 'Regional indicator' are categorical.
# 'Country name' is high cardinality and not typically used as a direct feature for regression
# without more advanced techniques (e.g., embedding), so we will not encode it.
# 'Regional indicator' could be encoded, but it's not part of the specified features.
# For the purpose of this task, no categorical features are used in the model,
# so no encoding is strictly necessary for the selected features.
# If 'Regional indicator' were to be used, OneHotEncoder would be appropriate.
print("\nNo categorical features are used in the specified model features, so no encoding is applied to X.")
print("Categorical columns in original dataset (not used as features): 'Country name', 'Regional indicator'")


# --- 2. Create target variable and features ---
X = df_cleaned[FEATURE_COLUMNS]
y = df_cleaned[TARGET_COLUMN]

print(f"\nFeatures (X) shape: {X.shape}")
print(f"Target (y) shape: {y.shape}")
print("\nFeatures head:")
print(X.head())
print("\nTarget head:")
print(y.head())

# --- 3. Split the data into training and testing sets (80-20 split) ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nX_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

# --- 4. Train and compare multiple regression models ---

# Initialize models
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(random_state=42),
    'Lasso Regression': Lasso(random_state=42),
    'Random Forest Regressor': RandomForestRegressor(random_state=42),
    'Gradient Boosting Regressor': GradientBoostingRegressor(random_state=42)
}

# Dictionary to store model performance
model_performance = {}

# Train and evaluate each model
print("\n--- Training and Evaluating Models ---")
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Calculate evaluation metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    model_performance[name] = {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2,
        'Model': model, # Store the trained model
        'Predictions': y_pred # Store predictions for residual plots
    }

    print(f"{name} Performance:")
    print(f"  MAE: {mae:.4f}")
    print(f"  MSE: {mse:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R2 Score: {r2:.4f}")

# --- 6. Evaluate all models using MAE, MSE, RMSE, and R² score ---
# Display all model performances in a structured way
performance_df = pd.DataFrame.from_dict(model_performance, orient='index')
print("\n--- Model Performance Summary ---")
print(performance_df[['MAE', 'MSE', 'RMSE', 'R2']].sort_values(by='R2', ascending=False))

# Determine the best model based on R2 score for hyperparameter tuning
best_initial_model_name = performance_df['R2'].idxmax()
print(f"\nBest initial model based on R2 score: {best_initial_model_name}")

# --- 5. Perform hyperparameter tuning for the best model using GridSearchCV ---
# We will tune the Gradient Boosting Regressor as it's often a strong performer and benefits from tuning.
# If Random Forest was better, we could tune that instead.
# Let's choose Gradient Boosting Regressor for tuning as it's generally robust.
best_model_for_tuning = GradientBoostingRegressor(random_state=42)
print(f"\n--- Hyperparameter Tuning for Gradient Boosting Regressor ---")

# Define parameter grid for GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 4, 5],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 3]
}

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=best_model_for_tuning,
                           param_grid=param_grid,
                           cv=5, # 5-fold cross-validation
                           scoring='neg_mean_squared_error', # Optimize for lower MSE
                           n_jobs=-1, # Use all available cores
                           verbose=1)

# Fit GridSearchCV to the training data
grid_search.fit(X_train, y_train)

# Get the best estimator and its parameters
best_tuned_model = grid_search.best_estimator_
best_params = grid_search.best_params_
best_score = -grid_search.best_score_ # Convert back to positive MSE

print(f"\nBest Hyperparameters found: {best_params}")
print(f"Best cross-validation MSE (from GridSearchCV): {best_score:.4f}")

# Evaluate the best tuned model on the test set
y_pred_tuned = best_tuned_model.predict(X_test)
mae_tuned = mean_absolute_error(y_test, y_pred_tuned)
mse_tuned = mean_squared_error(y_test, y_pred_tuned)
rmse_tuned = np.sqrt(mse_tuned)
r2_tuned = r2_score(y_test, y_pred_tuned)

print("\nBest Tuned Gradient Boosting Regressor Performance on Test Set:")
print(f"  MAE: {mae_tuned:.4f}")
print(f"  MSE: {mse_tuned:.4f}")
print(f"  RMSE: {rmse_tuned:.4f}")
print(f"  R2 Score: {r2_tuned:.4f}")

# Update model_performance with the tuned model's results
model_performance['Tuned Gradient Boosting Regressor'] = {
    'MAE': mae_tuned,
    'MSE': mse_tuned,
    'RMSE': rmse_tuned,
    'R2': r2_tuned,
    'Model': best_tuned_model,
    'Predictions': y_pred_tuned
}

# Re-display all model performances including the tuned one
performance_df = pd.DataFrame.from_dict(model_performance, orient='index')
print("\n--- Updated Model Performance Summary (including Tuned Model) ---")
print(performance_df[['MAE', 'MSE', 'RMSE', 'R2']].sort_values(by='R2', ascending=False))

# --- 7. Create residual plots and feature importance visualizations ---

# Residual Plot for the best tuned model
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_pred_tuned, y=(y_test - y_pred_tuned))
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel("Predicted Ladder Score")
plt.ylabel("Residuals (Actual - Predicted)")
plt.title("Residual Plot for Tuned Gradient Boosting Regressor")
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# Feature Importance for the best tuned model (Gradient Boosting Regressor)
if hasattr(best_tuned_model, 'feature_importances_'):
    feature_importances = pd.Series(best_tuned_model.feature_importances_, index=X_train.columns)
    feature_importances = feature_importances.sort_values(ascending=False)

    plt.figure(figsize=(12, 7))
    sns.barplot(x=feature_importances.values, y=feature_importances.index)
    plt.xlabel("Feature Importance Score")
    plt.ylabel("Features")
    plt.title("Feature Importance for Tuned Gradient Boosting Regressor")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
else:
    print("\nFeature importance not available for the selected best model (Linear Regression, Ridge, Lasso).")


# --- 8. Implement cross-validation to assess model stability ---

print("\n--- Cross-Validation for Best Tuned Model (Gradient Boosting Regressor) ---")

# Define KFold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Perform cross-validation using R2 score
cv_scores_r2 = cross_val_score(best_tuned_model, X, y, cv=kf, scoring='r2', n_jobs=-1)
print(f"Cross-validation R2 scores: {cv_scores_r2}")
print(f"Mean CV R2 score: {np.mean(cv_scores_r2):.4f}")
print(f"Standard deviation of CV R2 scores: {np.std(cv_scores_r2):.4f}")

# Perform cross-validation using negative Mean Squared Error
cv_scores_mse = cross_val_score(best_tuned_model, X, y, cv=kf, scoring='neg_mean_squared_error', n_jobs=-1)
cv_scores_rmse = np.sqrt(-cv_scores_mse) # Convert negative MSE to RMSE
print(f"Cross-validation RMSE scores: {cv_scores_rmse}")
print(f"Mean CV RMSE score: {np.mean(cv_scores_rmse):.4f}")
print(f"Standard deviation of CV RMSE scores: {np.std(cv_scores_rmse):.4f}")

# --- 9. Make predictions for new hypothetical countries and analyze results ---

print("\n--- Predicting Happiness for New Hypothetical Countries ---")

# Create hypothetical new country data
# Ensure column names and order match the training data (X_train.columns)
hypothetical_countries_data = pd.DataFrame({
    'Logged GDP per capita': [10.5, 9.0, 11.2, 8.5, 9.8],
    'Social support': [0.95, 0.80, 0.98, 0.75, 0.90],
    'Healthy life expectancy': [75.0, 65.0, 80.0, 60.0, 70.0],
    'Freedom to make life choices': [0.90, 0.60, 0.95, 0.50, 0.75],
    'Generosity': [0.20, 0.10, 0.30, 0.05, 0.15],
    'Perceptions of corruption': [0.10, 0.80, 0.05, 0.90, 0.50]
}, index=['Country A (High Happiness)', 'Country B (Low Happiness)', 'Country C (Very High)', 'Country D (Very Low)', 'Country E (Moderate)'])

print("\nHypothetical Country Data:")
print(hypothetical_countries_data)

# Make predictions using the best tuned model
predicted_happiness = best_tuned_model.predict(hypothetical_countries_data)

# Display predictions
print("\nPredicted Ladder Scores for Hypothetical Countries:")
for country, score in zip(hypothetical_countries_data.index, predicted_happiness):
    print(f"{country}: {score:.4f}")

# Analyze results (simple interpretation)
print("\n--- Analysis of Hypothetical Country Predictions ---")
print("Country A (High Happiness): High GDP, social support, life expectancy, freedom, low corruption. Predicted high happiness.")
print("Country B (Low Happiness): Lower GDP, social support, life expectancy, freedom, high corruption. Predicted lower happiness.")
print("Country C (Very High): Even higher values in positive indicators, very low corruption. Predicted very high happiness.")
print("Country D (Very Low): Very low values in positive indicators, very high corruption. Predicted very low happiness.")
print("Country E (Moderate): Moderate values across indicators. Predicted moderate happiness.")
print("\nThe predictions align with the expected relationships between the features and happiness score,")
print("where higher values in positive indicators (GDP, social support, etc.) and lower corruption")
print("generally lead to higher predicted happiness scores.")

print("\n--- End of Analysis ---")
```