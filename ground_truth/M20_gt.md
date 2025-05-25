```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- 1. Load the dataset and parse 'date' ---
# Load the synthetic_covid19_data.csv dataset
try:
    df = pd.read_csv('synthetic_covid19_data.csv', parse_dates=['date'])
except FileNotFoundError:
    print("Error: 'synthetic_covid19_data.csv' not found. Please ensure the dataset is in the same directory.")
    exit()

# --- 2. Select features and target variable ---
# Define the list of features to be used for prediction
features = [
    'new_cases', 'hosp_patients', 'icu_patients', 'reproduction_rate',
    'stringency_index', 'people_fully_vaccinated', 'median_age', 'continent'
]
# Define the target variable
target = 'new_deaths'

# Create a DataFrame containing only the selected features and the target
# Using .copy() to avoid SettingWithCopyWarning later
df_selected = df[features + [target]].copy()

# --- 3. Handle missing values in selected features and target by imputing with the median ---
# Identify numerical columns that might have missing values and need imputation
# 'continent' is categorical and will be handled by one-hot encoding, not median imputation
numerical_cols_for_imputation = [col for col in features if col != 'continent'] + [target]

# Impute missing values in each identified numerical column with its median
for col in numerical_cols_for_imputation:
    if df_selected[col].isnull().any():
        median_val = df_selected[col].median()
        df_selected[col].fillna(median_val, inplace=True)
        # print(f"Imputed missing values in '{col}' with median: {median_val}") # Uncomment for debugging

# --- 4. Encode the 'continent' column using one-hot encoding and add it to the features ---
# Perform one-hot encoding on the 'continent' column.
# drop_first=True is used to avoid multicollinearity by dropping the first category.
df_encoded = pd.get_dummies(df_selected, columns=['continent'], drop_first=True)

# Separate the features (X) and the target variable (y)
X = df_encoded.drop(columns=[target])
y = df_encoded[target]

# --- 5. Scale all numerical features using StandardScaler ---
# Scaling is performed after splitting to prevent data leakage from the test set into the training process.

# --- 6. Split the data into training (80%) and testing (20%) sets ---
# Use train_test_split to divide the dataset. random_state ensures reproducibility.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize StandardScaler
scaler = StandardScaler()

# Fit the scaler on the training data and transform both training and testing data
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- 7. Train a Linear Regression model and a Random Forest Regressor model ---

# Initialize and train the Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)

# Initialize and train the Random Forest Regressor model
# n_estimators=100 specifies the number of trees in the forest.
# random_state ensures reproducibility of the random forest.
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# --- 8. Evaluate both models using Mean Absolute Error (MAE), Mean Squared Error (MSE), and R-squared (R²) on the test set ---

# Make predictions on the scaled test set using the Linear Regression model
y_pred_lr = lr_model.predict(X_test_scaled)

# Make predictions on the scaled test set using the Random Forest Regressor model
y_pred_rf = rf_model.predict(X_test_scaled)

# Calculate evaluation metrics for Linear Regression
mae_lr = mean_absolute_error(y_test, y_pred_lr)
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

# Calculate evaluation metrics for Random Forest Regressor
mae_rf = mean_absolute_error(y_test, y_pred_rf)
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

# --- 9. Print the evaluation metrics for both models ---
print("--- Model Evaluation Results ---")

print("\nLinear Regression Model:")
print(f"  Mean Absolute Error (MAE): {mae_lr:.2f}")
print(f"  Mean Squared Error (MSE): {mse_lr:.2f}")
print(f"  R-squared (R²): {r2_lr:.2f}")

print("\nRandom Forest Regressor Model:")
print(f"  Mean Absolute Error (MAE): {mae_rf:.2f}")
print(f"  Mean Squared Error (MSE): {mse_rf:.2f}")
print(f"  R-squared (R²): {r2_rf:.2f}")
```