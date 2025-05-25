```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
import pmdarima as pm
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
import statsmodels.formula.api as smf
import ruptures as rpt
from lifelines import KaplanMeierFitter, CoxPHFitter
import warnings

# Suppress warnings for cleaner output, especially from statsmodels and pmdarima
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# --- 1. Load the dataset and aggregate sales data by year ---
try:
    df = pd.read_csv('video_games_sales.csv')
except FileNotFoundError:
    print("Error: video_games_sales.csv not found. Please ensure the file is in the current directory.")
    exit()

# Clean 'Year_of_Release' column: convert to numeric, coerce errors to NaN, then drop NaNs
df['Year_of_Release'] = pd.to_numeric(df['Year_of_Release'], errors='coerce')
df.dropna(subset=['Year_of_Release', 'Global_Sales'], inplace=True)
df['Year_of_Release'] = df['Year_of_Release'].astype(int)

# Aggregate Global_Sales by Year_of_Release
yearly_sales = df.groupby('Year_of_Release')['Global_Sales'].sum().reset_index()
yearly_sales.set_index('Year_of_Release', inplace=True)
# Convert to datetime index for time series analysis, ensuring yearly frequency
yearly_sales.index = pd.to_datetime(yearly_sales.index, format='%Y')
yearly_sales = yearly_sales.asfreq('YS')

print("--- Aggregated Yearly Sales ---")
print(yearly_sales.head())
print("\n")

# --- 2. Implement ARIMA modeling to forecast future global sales trends ---
print("--- ARIMA Modeling ---")
# Use auto_arima to find the best ARIMA model automatically
with warnings.catch_warnings():
    warnings.filterwarnings("ignore") # Suppress auto_arima internal warnings
    arima_model = pm.auto_arima(yearly_sales['Global_Sales'],
                                start_p=1, start_q=1,
                                test='adf',       # Use ADF test to determine differencing order
                                max_p=5, max_q=5, # Maximum p and q
                                m=1,              # No seasonality for yearly data (m=1)
                                d=None,           # Let model determine 'd'
                                seasonal=False,   # No seasonal component for yearly data
                                trace=False,      # Do not print results per iteration
                                error_action='ignore',
                                suppress_warnings=True,
                                stepwise=True)

print(f"Best ARIMA model: {arima_model.order}")

# Forecast future sales (e.g., next 5 years)
n_periods = 5
forecast_arima, conf_int_arima = arima_model.predict(n_periods=n_periods, return_conf_int=True)

# Create a DataFrame for the forecast with confidence intervals
forecast_index = pd.date_range(start=yearly_sales.index[-1] + pd.DateOffset(years=1), periods=n_periods, freq='YS')
forecast_df_arima = pd.DataFrame({'Forecast': forecast_arima,
                                  'Lower_CI': conf_int_arima[:, 0],
                                  'Upper_CI': conf_int_arima[:, 1]},
                                 index=forecast_index)

print("\nARIMA Forecast (next 5 years):")
print(forecast_df_arima)

# Plot ARIMA forecast
plt.figure(figsize=(12, 6))
plt.plot(yearly_sales['Global_Sales'], label='Historical Sales')
plt.plot(forecast_df_arima['Forecast'], label='ARIMA Forecast', color='red')
plt.fill_between(forecast_df_arima.index, forecast_df_arima['Lower_CI'], forecast_df_arima['Upper_CI'], color='pink', alpha=0.3, label='95% Confidence Interval')
plt.title('Global Video Game Sales Forecast with ARIMA')
plt.xlabel('Year')
plt.ylabel('Global Sales (Millions)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('arima_forecast.png')
# plt.show() # Uncomment to display plots

# --- 3. Perform seasonal decomposition of the time series data ---
print("\n--- Seasonal Decomposition ---")
# For yearly data, traditional seasonality (e.g., monthly/quarterly cycles) is not applicable.
# However, we can still decompose into trend, residual. If a long-term cycle is suspected (e.g., 5-10 years),
# a period can be specified. Let's try a period of 5 years if enough data points exist.
min_period_for_decomposition = 2 * 5 # Need at least two cycles for meaningful decomposition
if len(yearly_sales) >= min_period_for_decomposition:
    period_val = 5 # Assuming a 5-year cycle for demonstration
    try:
        # Use additive model as sales are likely to increase/decrease by an absolute amount
        decomposition = seasonal_decompose(yearly_sales['Global_Sales'], model='additive', period=period_val)
        fig = decomposition.plot()
        fig.set_size_inches(10, 8)
        fig.suptitle(f'Seasonal Decomposition of Global Sales (Period={period_val})', y=1.02)
        plt.tight_layout(rect=[0, 0.03, 1, 0.98])
        plt.savefig('seasonal_decomposition.png')
        # plt.show()
        print(f"Seasonal decomposition performed with period={period_val}.")
    except Exception as e:
        print(f"Could not perform seasonal decomposition with period={period_val}. Error: {e}")
        print("This often happens if the time series is too short for the specified period or period is too large.")
else:
    print(f"Time series is too short ({len(yearly_sales)} points) for seasonal decomposition with a period of 5 years (requires at least {min_period_for_decomposition} points).")

# --- 4. Build a multivariate regression model with interaction terms ---
print("\n--- Multivariate Regression with Interaction Terms ---")
# Select relevant columns for the model and drop rows with any NaN in them
df_reg = df[['Year_of_Release', 'Genre', 'Platform', 'Publisher', 'Global_Sales']].copy()
df_reg.dropna(inplace=True)

# Limit number of categories for Platform and Publisher to avoid too many features
# This is crucial for model stability and interpretability with many interaction terms.
top_platforms = df_reg['Platform'].value_counts().nlargest(20).index
top_publishers = df_reg['Publisher'].value_counts().nlargest(50).index

df_reg_filtered = df_reg[df_reg['Platform'].isin(top_platforms) & df_reg['Publisher'].isin(top_publishers)].copy()

# Create dummy variables for categorical features
df_dummies = pd.get_dummies(df_reg_filtered, columns=['Genre', 'Platform', 'Publisher'], drop_first=True)

# Build the formula string for statsmodels.OLS
formula = 'Global_Sales ~ Year_of_Release'

# Add dummy variables for Genre, Platform, Publisher
genre_cols = [col for col in df_dummies.columns if 'Genre_' in col]
platform_cols = [col for col in df_dummies.columns if 'Platform_' in col]
publisher_cols = [col for col in df_dummies.columns if 'Publisher_' in col]

formula += ' + ' + ' + '.join(genre_cols)
formula += ' + ' + ' + '.join(platform_cols)
formula += ' + ' + ' + '.join(publisher_cols)

# Add interaction terms (Year_of_Release with top 3 genres/platforms/publishers)
# This can still create many features, but limits the explosion.
top_3_genres = df_reg_filtered['Genre'].value_counts().nlargest(3).index.tolist()
top_3_platforms = df_reg_filtered['Platform'].value_counts().nlargest(3).index.tolist()
top_3_publishers = df_reg_filtered['Publisher'].value_counts().nlargest(3).index.tolist()

for genre in top_3_genres:
    if f'Genre_{genre}' in df_dummies.columns: # Check if dummy column exists
        formula += f' + Year_of_Release:Q("Genre_{genre}")' # Q() handles special characters in column names

for platform in top_3_platforms:
    if f'Platform_{platform}' in df_dummies.columns:
        formula += f' + Year_of_Release:Q("Platform_{platform}")'

for publisher in top_3_publishers:
    if f'Publisher_{publisher}' in df_dummies.columns:
        formula += f' + Year_of_Release:Q("Publisher_{publisher}")'

# Fit the OLS model using statsmodels.formula.api for easy formula handling
try:
    model_ols = smf.ols(formula=formula, data=df_dummies).fit()
    print(model_ols.summary())

    # Confidence Intervals for coefficients
    print("\nMultivariate Regression Coefficients Confidence Intervals:")
    print(model_ols.conf_int())

    # Prediction Intervals for new data (example: first row of filtered data)
    # statsmodels.OLSResults.get_prediction provides prediction intervals
    example_row_for_pred = df_dummies.iloc[[0]].copy()
    predictions_ols = model_ols.get_prediction(example_row_for_pred)
    pred_summary = predictions_ols.summary_frame(alpha=0.05) # 95% prediction interval
    print("\nMultivariate Regression Prediction Interval for an example data point:")
    print(pred_summary)

except Exception as e:
    print(f"Error building multivariate regression model: {e}")
    print("This might be due to too many features, perfect multicollinearity, or issues with formula parsing.")
    print("Consider further reducing the number of categories or interaction terms if this persists.")


# --- 5. Implement polynomial regression with regularization (Ridge and Lasso) ---
print("\n--- Polynomial Regression with Regularization ---")
# Use 'Year_of_Release' as the feature for polynomial regression
X_poly = df_reg_filtered[['Year_of_Release']]
y_poly = df_reg_filtered['Global_Sales']

# Create polynomial features (e.g., degree 2)
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly_transformed = poly.fit_transform(X_poly)

# Scale features for regularization (important for Ridge/Lasso)
scaler_poly = StandardScaler()
X_poly_scaled = scaler_poly.fit_transform(X_poly_transformed)

# Use TimeSeriesSplit for validation consistent with the prompt
tscv_poly = TimeSeriesSplit(n_splits=3) # Use 3 splits for demonstration

# Store scores for each fold
ridge_rmse_scores = []
lasso_rmse_scores = []

for fold, (train_index, test_index) in enumerate(tscv_poly.split(X_poly_scaled)):
    X_train_poly, X_test_poly = X_poly_scaled[train_index], X_poly_scaled[test_index]
    y_train_poly, y_test_poly = y_poly.iloc[train_index], y_poly.iloc[test_index]

    # Ridge Regression
    ridge_model = Ridge(alpha=1.0) # alpha is the regularization strength
    ridge_model.fit(X_train_poly, y_train_poly)
    y_pred_ridge = ridge_model.predict(X_test_poly)
    rmse_ridge = np.sqrt(mean_squared_error(y_test_poly, y_pred_ridge))
    ridge_rmse_scores.append(rmse_ridge)

    # Lasso Regression
    lasso_model = Lasso(alpha=0.1, max_iter=10000) # alpha is the regularization strength
    lasso_model.fit(X_train_poly, y_train_poly)
    y_pred_lasso = lasso_model.predict(X_test_poly)
    rmse_lasso = np.sqrt(mean_squared_error(y_test_poly, y_pred_lasso))
    lasso_rmse_scores.append(rmse_lasso)

    print(f"  Fold {fold+1}: Ridge RMSE={rmse_ridge:.3f}, Lasso RMSE={rmse_lasso:.3f}")

print(f"Average Ridge RMSE across {len(ridge_rmse_scores)} folds: {np.mean(ridge_rmse_scores):.3f}")
print(f"Average Lasso RMSE across {len(lasso_rmse_scores)} folds: {np.mean(lasso_rmse_scores):.3f}")

# Confidence/Prediction Intervals for Ridge/Lasso (approximate via residuals)
# sklearn models do not directly provide CIs/PIs. We approximate PIs based on residuals.
# This is a simplification and assumes homoscedasticity and normality of residuals.
# For a more robust approach, bootstrapping would be needed.
if len(y_test_poly) > 0:
    residuals_ridge = y_test_poly - y_pred_ridge
    std_err_ridge = np.std(residuals_ridge)
    # For 95% PI, use ~1.96 * std_err (for normal distribution)
    y_pred_ridge_lower = y_pred_ridge - 1.96 * std_err_ridge
    y_pred_ridge_upper = y_pred_ridge + 1.96 * std_err_ridge

    residuals_lasso = y_test_poly - y_pred_lasso
    std_err_lasso = np.std(residuals_lasso)
    y_pred_lasso_lower = y_pred_lasso - 1.96 * std_err_lasso
    y_pred_lasso_upper = y_pred_lasso + 1.96 * std_err_lasso

    print(f"\n  Approximate Ridge PI for first test point: [{y_pred_ridge_lower[0]:.3f}, {y_pred_ridge_upper[0]:.3f}]")
    print(f"  Approximate Lasso PI for first test point: [{y_pred_lasso_lower[0]:.3f}, {y_pred_lasso_upper[0]:.3f}]")
else:
    print("\nNot enough test data to calculate approximate PIs for polynomial regression.")

# --- 6. Use Bayesian regression with PyMC3 ---
print("\n--- Bayesian Regression with PyMC3 ---")
# PyMC3 can be slow with many features. Let's use a simpler model with fewer features.
# Use 'Year_of_Release', 'Genre', 'Platform' (top few)
df_bayesian = df_reg_filtered[['Year_of_Release', 'Genre', 'Platform', 'Global_Sales']].copy()

# Limit categories for Bayesian model to speed up sampling
top_genres_bayesian = df_bayesian['Genre'].value_counts().nlargest(5).index
top_platforms_bayesian = df_bayesian['Platform'].value_counts().nlargest(5).index

df_bayesian_filtered = df_bayesian[
    df_bayesian['Genre'].isin(top_genres_bayesian) &
    df_bayesian['Platform'].isin(top_platforms_bayesian)
].copy()

# Convert categorical features to numerical codes for PyMC3
df_bayesian_filtered['Genre_code'] = df_bayesian_filtered['Genre'].astype('category').cat.codes
df_bayesian_filtered['Platform_code'] = df_bayesian_filtered['Platform'].astype('category').cat.codes

# Scale numerical features for better MCMC sampling
scaler_year_bayesian = StandardScaler()
df_bayesian_filtered['Year_of_Release_scaled'] = scaler_year_bayesian.fit_transform(df_bayesian_filtered[['Year_of_Release']])
scaler_sales_bayesian = StandardScaler()
df_bayesian_filtered['Global_Sales_scaled'] = scaler_sales_bayesian.fit_transform(df_bayesian_filtered[['Global_Sales']])

X_bayesian = df_bayesian_filtered[['Year_of_Release_scaled', 'Genre_code', 'Platform_code']].values
y_bayesian = df_bayesian_filtered['Global_Sales_scaled'].values

try:
    import pymc3 as pm3
    import arviz as az

    # Define the Bayesian model
    with pm3.Model() as bayesian_model:
        # Priors for regression coefficients
        beta_year = pm3.Normal('beta_year', mu=0, sigma=1)
        # Use a shared prior for categorical effects for regularization
        beta_genre = pm3.Normal('beta_genre', mu=0, sigma=1, shape=len(df_bayesian_filtered['Genre'].cat.categories))
        beta_platform = pm3.Normal('beta_platform', mu=0, sigma=1, shape=len(df_bayesian_filtered['Platform'].cat.categories))
        intercept = pm3.Normal('intercept', mu=0, sigma=1)

        # Linear model
        mu = intercept + beta_year * X_bayesian[:, 0] + \
             beta_genre[X_bayesian[:, 1].astype(int)] + \
             beta_platform[X_bayesian[:, 2].astype(int)]

        # Likelihood (observed data)
        sigma = pm3.HalfNormal('sigma', sigma=1) # Standard deviation must be positive
        Y_obs = pm3.Normal('Y_obs', mu=mu, sigma=sigma, observed=y_bayesian)

        # Sample from the posterior
        # Reduced draws and cores for faster execution in a general solution
        trace = pm3.sample(1000, tune=500, cores=1, return_inferencedata=True, random_seed=42)

    print("\nBayesian Model Summary:")
    print(az.summary(trace, var_names=['beta_year', 'beta_genre', 'beta_platform', 'intercept', 'sigma']))

    # Posterior predictive checks and prediction intervals
    with bayesian_model:
        # Sample from the posterior predictive distribution
        ppc = pm3.sample_posterior_predictive(trace, var_names=['Y_obs'], samples=500, random_seed=42)

    # Convert scaled predictions back to original scale
    ppc_original_scale = scaler_sales_bayesian.inverse_transform(ppc['Y_obs'])

    # Calculate prediction intervals (e.g., 95% credible interval)
    lower_bound = np.percentile(ppc_original_scale, 2.5, axis=0)
    upper_bound = np.percentile(ppc_original_scale, 97.5, axis=0)
    median_pred = np.percentile(ppc_original_scale, 50, axis=0)

    print("\nBayesian Regression Prediction Intervals (first 5 data points):")
    for i in range(min(5, len(df_bayesian_filtered))):
        print(f"  Actual: {df_bayesian_filtered['Global_Sales'].iloc[i]:.3f}, Predicted Median: {median_pred[i]:.3f}, PI: [{lower_bound[i]:.3f}, {upper_bound[i]:.3f}]")

except ImportError:
    print("PyMC3 or ArviZ not installed. Skipping Bayesian Regression.")
    print("Please install with: pip install pymc3 arviz")
except Exception as e:
    print(f"Error during PyMC3 Bayesian Regression: {e}")
    print("This might be due to issues with Theano compilation, model complexity, or data size.")


# --- 7. Perform change point detection ---
print("\n--- Change Point Detection ---")
# Use the aggregated yearly sales data for change point detection
signal = yearly_sales['Global_Sales'].values

# Detect change points using Pelt algorithm
# 'model' can be 'l1', 'l2', 'rbf'. 'rbf' is often good for changes in mean/variance.
algo = rpt.Pelt(model="rbf").fit(signal)
# 'pen' is the penalty value. A higher penalty means fewer change points.
# A common heuristic for penalty is 2 * log(n) for BIC-like behavior.
n_samples_signal = len(signal)
penalty_val = 2 * np.log(n_samples_signal) # Example penalty
result = algo.predict(pen=penalty_val)

# The last element of result is always the end of the series, so remove it if it's the last index
if result and result[-1] == len(signal):
    result = result[:-1]

print(f"Detected change points at indices: {result}")
# Map indices back to years
change_point_years = [yearly_sales.index[i].year for i in result]
print(f"Detected change points in years: {change_point_years}")

# Plot change points
plt.figure(figsize=(12, 6))
plt.plot(yearly_sales.index, yearly_sales['Global_Sales'], label='Global Sales')
for cp_idx in result:
    plt.axvline(yearly_sales.index[cp_idx], color='red', linestyle='--', label='Change Point' if cp_idx == result[0] else "")
plt.title('Change Point Detection in Global Video Game Sales')
plt.xlabel('Year')
plt.ylabel('Global Sales (Millions)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('change_points.png')
# plt.show()


# --- 8. Implement survival analysis to model 'game longevity' ---
print("\n--- Survival Analysis (Game Longevity) ---")
# Define 'longevity' as the difference between the maximum year a game appeared
# and its release year. Assume all games are 'observed' (event=1) at the end of their longevity.
# This is a simplified definition as we don't have explicit 'death' events (e.g., sales dropping to zero).

# Calculate max year and min year (release year) for each unique game
game_max_year = df.groupby('Name')['Year_of_Release'].max().reset_index()
game_min_year = df.groupby('Name')['Year_of_Release'].min().reset_index()

# Merge to get release year and last observed year for each game
game_longevity_df = pd.merge(game_min_year, game_max_year, on='Name', suffixes=('_release', '_last_observed'))
# Longevity is the number of years the game was observed, inclusive of release year
game_longevity_df['Longevity_Years'] = game_longevity_df['Year_of_Release_last_observed'] - game_longevity_df['Year_of_Release_release'] + 1

# Filter out games with 0 or negative longevity (e.g., data errors or single-year entries)
game_longevity_df = game_longevity_df[game_longevity_df['Longevity_Years'] > 0].copy()

# Add a dummy 'event' column, assuming all games are observed (event=1)
game_longevity_df['Observed_Event'] = 1

print(f"Sample of game longevity data ({len(game_longevity_df)} games):")
print(game_longevity_df.head())

# Kaplan-Meier Fitter for survival function
kmf = KaplanMeierFitter()
kmf.fit(game_longevity_df['Longevity_Years'], event_observed=game_longevity_df['Observed_Event'])

plt.figure(figsize=(10, 6))
kmf.plot_survival_function()
plt.title('Kaplan-Meier Survival Function for Game Longevity')
plt.xlabel('Years Since Release')
plt.ylabel('Probability of Survival (Still Observed)')
plt.grid(True)
plt.tight_layout()
plt.savefig('kaplan_meier.png')
# plt.show()

print("\nKaplan-Meier Survival Function Summary (first 5 years):")
print(kmf.survival_function_.head())

# Cox Proportional Hazards Model (to include covariates)
# Need to merge longevity data with original game attributes (Genre, Platform, Global_Sales)
# Aggregate attributes for each unique game (e.g., mode for categorical, sum for sales)
df_game_attributes = df.groupby('Name').agg({
    'Genre': lambda x: x.mode()[0] if not x.mode().empty else np.nan,
    'Platform': lambda x: x.mode()[0] if not x.mode().empty else np.nan,
    'Publisher': lambda x: x.mode()[0] if not x.mode().empty else np.nan,
    'Global_Sales': 'sum' # Total sales over its observed lifespan
}).reset_index()

# Merge longevity with attributes
cox_data = pd.merge(game_longevity_df, df_game_attributes, on='Name', how='left')
cox_data.dropna(subset=['Genre', 'Platform', 'Publisher'], inplace=True)

# Limit categories for Cox model to manage complexity
top_genres_cox = cox_data['Genre'].value_counts().nlargest(10).index
top_platforms_cox = cox_data['Platform'].value_counts().nlargest(10).index

cox_data_filtered = cox_data[
    cox_data['Genre'].isin(top_genres_cox) &
    cox_data['Platform'].isin(top_platforms_cox)
].copy()

# One-hot encode categorical features for Cox model
cox_data_encoded = pd.get_dummies(cox_data_filtered, columns=['Genre', 'Platform'], drop_first=True)

# Select features for Cox model (duration, event, and covariates)
cox_features = ['Longevity_Years', 'Observed_Event', 'Global_Sales'] + \
               [col for col in cox_data_encoded.columns if 'Genre_' in col or 'Platform_' in col]

cox_data_final = cox_data_encoded[cox_features].copy()

# Fit Cox Proportional Hazards model
cph = CoxPHFitter()
try:
    cph.fit(cox_data_final, duration_col='Longevity_Years', event_col='Observed_Event')
    print("\nCox Proportional Hazards Model Summary:")
    cph.print_summary()

    # Confidence Intervals for Cox model coefficients
    print("\nCox Model Coefficients Confidence Intervals:")
    print(cph.confidence_intervals_)

    # Prediction for a new game (example: median sales, specific genre/platform)
    example_cox_data = pd.DataFrame(np.zeros((1, len(cox_data_final.columns))), columns=cox_data_final.columns)
    example_cox_data['Global_Sales'] = cox_data_final['Global_Sales'].median()
    # Set one dummy variable for each category to 1 for a representative prediction
    if 'Genre_Action' in example_cox_data.columns: example_cox_data['Genre_Action'] = 1
    if 'Platform_PS2' in example_cox_data.columns: example_cox_data['Platform_PS2'] = 1

    # Drop duration and event columns for prediction
    example_cox_data_for_pred = example_cox_data.drop(columns=['Longevity_Years', 'Observed_Event'])

    # Predict survival function for the example game
    survival_prediction = cph.predict_survival_function(example_cox_data_for_pred)
    plt.figure(figsize=(10, 6))
    survival_prediction.plot()
    plt.title('Predicted Survival Function for an Example Game (Cox Model)')
    plt.xlabel('Years Since Release')
    plt.ylabel('Probability of Survival')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('cox_survival_prediction.png')
    # plt.show()
    print("\nPredicted survival function for an example game plotted.")

except Exception as e:
    print(f"Error fitting Cox Proportional Hazards model: {e}")
    print("This might be due to insufficient data, issues with feature encoding, or model convergence.")


# --- 10. Validate models using time series cross-validation ---
print("\n--- Time Series Cross-Validation ---")

# ARIMA Time Series Cross-Validation (Manual Rolling Forecast)
print("\nARIMA Time Series Cross-Validation (Manual Rolling Forecast):")
# Define a rolling window for evaluation
# Start training from a reasonable point, e.g., first 70% of data
train_size_arima_cv = int(len(yearly_sales) * 0.7)
n_forecast_steps_cv = 1 # Forecast one step ahead

arima_rmse_scores_cv = []

# Loop through the time series, expanding the training window
for i in range(train_size_arima_cv, len(yearly_sales) - n_forecast_steps_cv + 1):
    train_data_cv = yearly_sales['Global_Sales'].iloc[:i]
    test_data_cv = yearly_sales['Global_Sales'].iloc[i:i+n_forecast_steps_cv]

    if len(test_data_cv) == 0:
        continue

    try:
        # Fit ARIMA model on training data using the best order found earlier
        model_fold_arima = pm.ARIMA(order=arima_model.order,
                                    seasonal_order=arima_model.seasonal_order,
                                    suppress_warnings=True)
        model_fold_arima.fit(train_data_cv)
        forecast_fold_arima = model_fold_arima.predict(n_periods=n_forecast_steps_cv)

        rmse_fold_arima = np.sqrt(mean_squared_error(test_data_cv, forecast_fold_arima))
        arima_rmse_scores_cv.append(rmse_fold_arima)
    except Exception as e:
        print(f"  Error in ARIMA TS-CV fold {i - train_size_arima_cv + 1}: {e}")
        continue

if arima_rmse_scores_cv:
    print(f"Average ARIMA RMSE across {len(arima_rmse_scores_cv)} folds: {np.mean(arima_rmse_scores_cv):.3f}")
else:
    print("No successful ARIMA TS-CV folds.")


# Regression Models Time Series Cross-Validation (using TimeSeriesSplit)
print("\nRegression Models Time Series Cross-Validation (TimeSeriesSplit):")

# Prepare data for TimeSeriesSplit. Sort by Year_of_Release.
df_dummies_sorted = df_dummies.sort_values(by='Year_of_Release')
X_multi_sorted = df_dummies_sorted.drop('Global_Sales', axis=1)
y_multi_sorted = df_dummies_sorted['Global_Sales']

tscv_reg = TimeSeriesSplit(n_splits=5) # 5 splits for regression models

ols_rmse_scores_cv = []
ridge_rmse_scores_cv = []
lasso_rmse_scores_cv = []

for fold, (train_index, test_index) in enumerate(tscv_reg.split(X_multi_sorted)):
    X_train_multi, X_test_multi = X_multi_sorted.iloc[train_index], X_multi_sorted.iloc[test_index]
    y_train_multi, y_test_multi = y_multi_sorted.iloc[train_index], y_multi_sorted.iloc[test_index]

    # Multivariate Regression (using sklearn's LinearRegression for simplicity in TS-CV)
    model_ols_cv = LinearRegression()
    model_ols_cv.fit(X_train_multi, y_train_multi)
    y_pred_ols_cv = model_ols_cv.predict(X_test_multi)
    rmse_ols_cv = np.sqrt(mean_squared_error(y_test_multi, y_pred_ols_cv))
    ols_rmse_scores_cv.append(rmse_ols_cv)

    # Polynomial Regression (Ridge and Lasso)
    # Re-transform and scale for each fold to avoid data leakage
    X_train_poly_cv = poly.fit_transform(X_train_multi[['Year_of_Release']])
    X_test_poly_cv = poly.transform(X_test_multi[['Year_of_Release']]) # Use transform, not fit_transform

    scaler_poly_cv = StandardScaler()
    X_train_poly_scaled_cv = scaler_poly_cv.fit_transform(X_train_poly_cv)
    X_test_poly_scaled_cv = scaler_poly_cv.transform(X_test_poly_cv)

    # Ridge
    ridge_model_cv = Ridge(alpha=1.0)
    ridge_model_cv.fit(X_train_poly_scaled_cv, y_train_multi) # Use y_train_multi for consistency
    y_pred_ridge_cv = ridge_model_cv.predict(X_test_poly_scaled_cv)
    rmse_ridge_cv = np.sqrt(mean_squared_error(y_test_multi, y_pred_ridge_cv))
    ridge_rmse_scores_cv.append(rmse_ridge_cv)

    # Lasso
    lasso_model_cv = Lasso(alpha=0.1, max_iter=10000)
    lasso_model_cv.fit(X_train_poly_scaled_cv, y_train_multi)
    y_pred_lasso_cv = lasso_model_cv.predict(X_test_poly_scaled_cv)
    rmse_lasso_cv = np.sqrt(mean_squared_error(y_test_multi, y_pred_lasso_cv))
    lasso_rmse_scores_cv.append(rmse_lasso_cv)

    print(f"  Fold {fold+1}: OLS RMSE={rmse_ols_cv:.3f}, Ridge RMSE={rmse_ridge_cv:.3f}, Lasso RMSE={rmse_lasso_cv:.3f}")

print(f"Average OLS RMSE across {len(ols_rmse_scores_cv)} folds: {np.mean(ols_rmse_scores_cv):.3f}")
print(f"Average Ridge RMSE across {len(ridge_rmse_scores_cv)} folds: {np.mean(ridge_rmse_scores_cv):.3f}")
print(f"Average Lasso RMSE across {len(lasso_rmse_scores_cv)} folds: {np.mean(lasso_rmse_scores_cv):.3f}")

print("\n--- All tasks completed. Check generated plots. ---")
```