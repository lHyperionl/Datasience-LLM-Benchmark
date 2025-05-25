```python
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pmdarima as pm
import warnings

# Suppress warnings for cleaner output, especially from auto_arima and statsmodels convergence
warnings.filterwarnings("ignore")

# --- 1. Load data, parse 'date', filter for 'Germany', and set 'date' as index ---

# Define the path to the dataset
file_path = 'synthetic_covid19_data.csv'

# Load the dataset
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found. Please ensure it's in the correct directory.")
    exit()

# Parse the 'date' column to datetime objects
df['date'] = pd.to_datetime(df['date'])

# Filter the DataFrame for 'Germany'
germany_df = df[df['location'] == 'Germany'].copy()

# Set 'date' as the DataFrame index
germany_df.set_index('date', inplace=True)

# Sort the index to ensure chronological order, which is crucial for time series
germany_df.sort_index(inplace=True)

# --- 2. Use 'new_cases' as the target variable. Consider 'stringency_index' and 'people_fully_vaccinated' as exogenous variables. Handle missing values. ---

# Define the target variable and exogenous variables
target_variable = 'new_cases'
exog_variables = ['stringency_index', 'people_fully_vaccinated']

# Select only the necessary columns for forecasting
data = germany_df[[target_variable] + exog_variables]

# Handle missing values:
# First, forward fill any NaNs (propagates last valid observation forward)
data.ffill(inplace=True)
# Then, backward fill any remaining NaNs (e.g., NaNs at the very beginning of the series)
data.bfill(inplace=True)

# As a robust fallback, if any NaNs still exist (e.g., if a column was entirely NaN),
# interpolate linearly and then ffill/bfill again.
if data.isnull().sum().sum() > 0:
    print("Warning: Some NaNs still remain after ffill and bfill. Attempting interpolation.")
    data.interpolate(method='linear', inplace=True)
    data.ffill(inplace=True) # Final ffill after interpolation
    data.bfill(inplace=True) # Final bfill after interpolation

# Final check for NaNs and drop rows if any persist (less ideal for TS, but a safeguard)
if data.isnull().sum().sum() > 0:
    print("Critical: NaNs still present after all imputation attempts. Dropping rows with NaNs.")
    data.dropna(inplace=True)
    if data.empty:
        print("Error: Dataframe is empty after dropping NaNs. Cannot proceed with forecasting.")
        exit()

# Separate the target variable (y) and exogenous variables (X)
y = data[target_variable]
X = data[exog_variables]

# --- 3. Split data into training and testing sets (e.g., last 30 days for testing). ---

# Define the number of days for the test set
test_days = 30

# Check if there's enough data to create the specified test set
if len(data) < test_days:
    print(f"Error: Not enough data points ({len(data)}) for Germany to create a test set of {test_days} days.")
    print("Please reduce 'test_days' or ensure the dataset for Germany is sufficiently large.")
    exit()

# Determine the split point for training and testing
split_point = len(data) - test_days

# Split the target variable
y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]

# Split the exogenous variables
X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]

# Ensure X_test has the same number of rows as y_test for consistent forecasting
if len(X_test) != len(y_test):
    print("Error: Mismatch in length between X_test and y_test after splitting. This should not happen if data is aligned.")
    exit()

# --- 4. Implement a SARIMA model. Determine appropriate (p,d,q)(P,D,Q,s) orders (e.g., using auto_arima). ---

print("Searching for optimal SARIMA orders using auto_arima. This may take a moment...")
# Use pmdarima's auto_arima to automatically determine the best SARIMA orders
# 'm=7' specifies a weekly seasonality
auto_model = pm.auto_arima(y_train,
                           exog=X_train,
                           start_p=1, start_q=1,
                           test='adf',       # Use Augmented Dickey-Fuller test to determine 'd'
                           max_p=5, max_q=5, # Maximum non-seasonal orders to consider
                           m=7,              # Seasonal period (7 for weekly seasonality)
                           start_P=0, start_Q=0, # Starting seasonal orders
                           max_P=2, max_Q=2, # Maximum seasonal orders to consider
                           seasonal=True,    # Enable seasonality
                           d=None, D=None,   # Let auto_arima determine 'd' and 'D'
                           trace=False,      # Set to True to see the search process output
                           error_action='ignore', # Ignore errors during model fitting
                           suppress_warnings=True, # Suppress warnings from statsmodels
                           stepwise=True,    # Use the stepwise algorithm for faster search
                           n_jobs=-1)        # Use all available CPU cores for parallel processing

# Extract the best non-seasonal and seasonal orders found by auto_arima
order = auto_model.order
seasonal_order = auto_model.seasonal_order
print(f"Optimal SARIMA orders found: Non-seasonal {order}, Seasonal {seasonal_order}")

# --- 5. Train the SARIMA model on the training set, including exogenous variables. ---

print("Training SARIMA model with optimal orders...")
# Initialize and fit the SARIMAX model from statsmodels
sarima_model = SARIMAX(y_train,
                       exog=X_train,
                       order=order,
                       seasonal_order=seasonal_order,
                       enforce_stationarity=False, # Set to False if auto_arima determines d/D > 0
                       enforce_invertibility=False) # Set to False if auto_arima determines d/D > 0

sarima_results = sarima_model.fit(disp=False) # disp=False suppresses convergence output during fitting
print("SARIMA model training complete.")
# Uncomment the line below to print a detailed summary of the fitted model
# print(sarima_results.summary())

# --- 6. Forecast 'new_cases' on the test set. ---

print("Generating forecasts for the test period...")
# Generate forecasts for the test period using the fitted model
# The 'exog' parameter must be provided for the forecast period (X_test)
forecast_steps = len(y_test)
forecast_obj = sarima_results.get_forecast(steps=forecast_steps, exog=X_test)
forecast_values = forecast_obj.predicted_mean

# Ensure the forecast values have the same index as the actual test data for easy comparison and plotting
forecast_values.index = y_test.index

# Handle potential negative forecasts, which are not logical for 'new_cases'
forecast_values[forecast_values < 0] = 0

# --- 7. Evaluate the forecast using Root Mean Squared Error (RMSE). ---

# Calculate the Root Mean Squared Error (RMSE) between actual and forecasted values
rmse = np.sqrt(mean_squared_error(y_test, forecast_values))
print(f"\nRoot Mean Squared Error (RMSE) on the test set: {rmse:.2f}")

# --- 8. Plot the actual vs. forecasted values for the test period. ---

plt.figure(figsize=(16, 8))

# Plot training data
plt.plot(y_train.index, y_train, label='Training Data (Actual)', color='blue', alpha=0.7)

# Plot actual test data
plt.plot(y_test.index, y_test, label='Actual Test Data', color='green', linewidth=2)

# Plot forecasted values
plt.plot(forecast_values.index, forecast_values, label='SARIMA Forecast', color='red', linestyle='--', linewidth=2)

plt.title(f'SARIMA Forecast of New Cases for Germany (RMSE: {rmse:.2f})', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('New Cases', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout() # Adjusts plot to prevent labels from overlapping
plt.show()

print("\nTime series forecasting process complete.")
```