```python
import pandas as pd
import numpy as np
import pymc3 as pm
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from scipy.stats import ttest_ind
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Suppress future warnings from PyMC3/ArviZ and other libraries
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Set a random seed for reproducibility
np.random.seed(42)
# Increase PyMC3 data buffer size for potentially large models
pm.set_data_buffer_size(1000000000)

# --- 1. Load and Preprocess the Dataset ---
print("--- 1. Loading and Preprocessing Data ---")
try:
    df = pd.read_csv('world-happiness-report-2021.csv')
except FileNotFoundError:
    print("Error: 'world-happiness-report-2021.csv' not found. Please ensure the file is in the current directory.")
    exit()

# Rename columns for easier access (remove spaces and special characters)
df.columns = df.columns.str.replace(' ', '_').str.replace('[^A-Za-z0-9_]+', '', regex=True)

# Identify key variables for analysis
happiness_score = 'Ladder_score'
gdp_per_capita = 'Logged_GDP_per_capita'
social_support = 'Social_support'
healthy_life_expectancy = 'Healthy_life_expectancy'
freedom = 'Freedom_to_make_life_choices'
generosity = 'Generosity'
corruption = 'Perceptions_of_corruption'
regional_indicator = 'Regional_indicator'

# Select relevant columns for modeling
cols_to_use = [
    happiness_score, gdp_per_capita, social_support,
    healthy_life_expectancy, freedom, generosity, corruption,
    regional_indicator, 'Country_name'
]
df_model = df[cols_to_use].copy()

# Comprehensive Missing Value Handling
# Separate numerical and categorical columns
numerical_cols = [col for col in df_model.columns if df_model[col].dtype in ['float64', 'int64'] and col != happiness_score]
categorical_cols = [col for col in df_model.columns if df_model[col].dtype == 'object' and col != 'Country_name']

# Create preprocessing pipelines for numerical and categorical features
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')), # Median imputation for numerical
    ('scaler', StandardScaler()) # Standardize numerical features
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')), # Mode imputation for categorical
    ('onehot', OneHotEncoder(handle_unknown='ignore')) # One-hot encode categorical features
])

# Create a preprocessor using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ],
    remainder='passthrough' # Keep other columns (like Country_name, Ladder_score)
)

# Apply preprocessing
# Fit and transform numerical and categorical features
df_processed_features = preprocessor.fit_transform(df_model)

# Get feature names after one-hot encoding
ohe_feature_names = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_cols)
processed_feature_names = numerical_cols + list(ohe_feature_names)

# Create a DataFrame with processed features
# The remainder='passthrough' puts non-transformed columns at the end.
# We need to correctly re-assign them.
df_processed = pd.DataFrame(df_processed_features, columns=processed_feature_names + ['Country_name_temp', 'Ladder_score_temp'])
# Re-assign original Ladder_score and Country_name
df_processed[happiness_score] = df_model[happiness_score].values
df_processed['Country_name'] = df_model['Country_name'].values

# Drop temporary columns created by remainder='passthrough'
df_processed = df_processed.drop(columns=['Country_name_temp', 'Ladder_score_temp'])

# Handle potential missing values in the target variable (Ladder_score)
df_processed.dropna(subset=[happiness_score], inplace=True)

print(f"Original data shape: {df.shape}")
print(f"Processed data shape: {df_processed.shape}")
print(f"Missing values after processing:\n{df_processed.isnull().sum().sum()}")

# Map regional indicator to integer IDs for PyMC3 random effects
regions = df_processed[regional_indicator].unique()
region_id_map = {region: i for i, region in enumerate(regions)}
df_processed['region_id'] = df_processed[regional_indicator].map(region_id_map)

# Define predictors for models (these are the standardized numerical features)
predictors = [gdp_per_capita, social_support, healthy_life_expectancy,
              freedom, generosity, corruption]

# Extract processed numerical features and target for PyMC3
y_processed = df_processed[happiness_score].values
region_ids = df_processed['region_id'].values
n_regions = len(regions)
n_countries = len(df_processed)

# --- 2. Hierarchical Bayesian Model (PyMC3) ---
print("\n--- 2. Hierarchical Bayesian Model ---")
# Define informative priors based on general happiness research:
# GDP, social support, life expectancy, freedom, generosity are generally positive.
# Corruption is generally negative.
# Priors are set to be slightly informative, centered around plausible effects for standardized data.
# For scaled predictors, a coefficient of 0.5 means a 1-std increase in predictor
# leads to 0.5-std increase in happiness (assuming happiness is also scaled, which it isn't here,
# but the relative magnitudes are what matter for interpretation of standardized coefficients).
with pm.Model() as hierarchical_model:
    # Hyperpriors for regional effects
    mu_a = pm.Normal('mu_a', mu=0., sd=1.) # Mean of regional intercepts
    sigma_a = pm.HalfNormal('sigma_a', sd=1.) # Std of regional intercepts

    # Regional intercepts (random effects)
    a = pm.Normal('a', mu=mu_a, sd=sigma_a, shape=n_regions)

    # Fixed effects (population-level parameters) with informative priors
    beta_gdp = pm.Normal('beta_gdp', mu=0.5, sd=0.2) # Logged_GDP_per_capita: Positive effect
    beta_social = pm.Normal('beta_social', mu=0.7, sd=0.2) # Social_support: Strong positive effect
    beta_health = pm.Normal('beta_health', mu=0.4, sd=0.2) # Healthy_life_expectancy: Positive effect
    beta_freedom = pm.Normal('beta_freedom', mu=0.3, sd=0.2) # Freedom_to_make_life_choices: Positive effect
    beta_generosity = pm.Normal('beta_generosity', mu=0.1, sd=0.1) # Generosity: Small positive effect
    beta_corruption = pm.Normal('beta_corruption', mu=-0.2, sd=0.1) # Perceptions_of_corruption: Negative effect

    # Intercept (overall mean happiness for standardized predictors)
    intercept = pm.Normal('intercept', mu=y_processed.mean(), sd=y_processed.std())

    # Expected value of happiness (linear model with random effects)
    mu_happiness_pred = (intercept + a[region_ids] +
                         beta_gdp * df_processed[gdp_per_capita].values +
                         beta_social * df_processed[social_support].values +
                         beta_health * df_processed[healthy_life_expectancy].values +
                         beta_freedom * df_processed[freedom].values +
                         beta_generosity * df_processed[generosity].values +
                         beta_corruption * df_processed[corruption].values)

    # Likelihood (observed happiness scores)
    sigma = pm.HalfNormal('sigma', sd=1.) # Residual standard deviation
    happiness_likelihood = pm.Normal('happiness_likelihood', mu=mu_happiness_pred, sd=sigma, observed=y_processed)

    # Sample from the posterior
    trace_hierarchical = pm.sample(2000, tune=1000, cores=2, return_inferencedata=True, random_seed=42)

print("Hierarchical Model Summary:")
print(az.summary(trace_hierarchical, var_names=['beta_gdp', 'beta_social', 'beta_health',
                                                'beta_freedom', 'beta_generosity', 'beta_corruption',
                                                'mu_a', 'sigma_a', 'sigma', 'intercept']))

# Posterior Predictive Checks for Hierarchical Model
print("\n--- Posterior Predictive Checks for Hierarchical Model ---")
with hierarchical_model:
    ppc_hierarchical = pm.sample_posterior_predictive(trace_hierarchical, samples=500)

fig, ax = plt.subplots(figsize=(10, 6))
az.plot_ppc(az.from_pymc3(posterior_predictive=ppc_hierarchical, model=hierarchical_model), ax=ax, data_pairs={"happiness_likelihood": "happiness_likelihood"})
ax.set_title("Posterior Predictive Check for Hierarchical Model")
plt.tight_layout()
plt.savefig('hierarchical_ppc.png')
plt.close()


# --- 3. Structural Equation Models (SEM) - Approximated with PyMC3 ---
# This section approximates SEM by modeling a system of interconnected Bayesian regressions.
# Example pathways for mediation analysis:
# 1. GDP -> Social Support (Path A)
# 2. Social Support -> Happiness (Path B)
# 3. GDP -> Happiness (Direct Path C')
print("\n--- 3. Structural Equation Models (Approximated with PyMC3) ---")

# Model 1: GDP -> Social Support (Path A)
with pm.Model() as sem_gdp_social:
    # Priors
    alpha_social = pm.Normal('alpha_social', mu=0, sd=1)
    beta_gdp_to_social = pm.Normal('beta_gdp_to_social', mu=0.5, sd=0.2) # Expect positive effect
    sigma_social = pm.HalfNormal('sigma_social', sd=1)

    # Expected Social Support
    mu_social = alpha_social + beta_gdp_to_social * df_processed[gdp_per_capita].values

    # Likelihood
    social_likelihood = pm.Normal('social_likelihood', mu=mu_social, sd=sigma_social, observed=df_processed[social_support].values)

    trace_gdp_social = pm.sample(1000, tune=500, cores=2, return_inferencedata=True, random_seed=42)

print("\nSEM Path 1: GDP -> Social Support Summary")
print(az.summary(trace_gdp_social, var_names=['beta_gdp_to_social']))

# Model 2: Social Support + GDP -> Happiness (Paths B and C')
with pm.Model() as sem_social_gdp_happiness:
    # Priors
    alpha_happiness = pm.Normal('alpha_happiness', mu=0, sd=1)
    beta_social_to_happiness = pm.Normal('beta_social_to_happiness', mu=0.7, sd=0.2) # Expect strong positive effect
    beta_gdp_to_happiness_direct = pm.Normal('beta_gdp_to_happiness_direct', mu=0.3, sd=0.2) # Expect positive direct effect
    sigma_happiness = pm.HalfNormal('sigma_happiness', sd=1)

    # Expected Happiness
    mu_happiness_pred = (alpha_happiness +
                         beta_social_to_happiness * df_processed[social_support].values +
                         beta_gdp_to_happiness_direct * df_processed[gdp_per_capita].values)

    # Likelihood
    happiness_likelihood = pm.Normal('happiness_likelihood', mu=mu_happiness_pred, sd=sigma_happiness, observed=y_processed)

    trace_social_gdp_happiness = pm.sample(1000, tune=500, cores=2, return_inferencedata=True, random_seed=42)

print("\nSEM Path 2: Social Support + GDP -> Happiness Summary")
print(az.summary(trace_social_gdp_happiness, var_names=['beta_social_to_happiness', 'beta_gdp_to_happiness_direct']))


# --- 4. Causal Inference: Instrumental Variables (IV) and Propensity Score Matching (PSM) ---
print("\n--- 4. Causal Inference: Instrumental Variables (IV) and Propensity Score Matching (PSM) ---")

# Instrumental Variables (IV) using 2SLS (Two-Stage Least Squares) with statsmodels
# This dataset is cross-sectional, so a true, valid instrument is not available.
# We will simulate a plausible instrument for demonstration purposes.
# Let's assume 'Historical_Colonial_Influence' (simulated) as an instrument for GDP.
# It should affect GDP but not directly happiness, except through GDP.
print("\n--- Instrumental Variables (2SLS) ---")
# Simulate a binary instrument: 1 for historically influenced, 0 otherwise.
# Make it correlated with GDP (e.g., higher GDP for instrument=1).
# Create a copy to avoid modifying the original df_processed for other analyses
df_iv = df_processed.copy()
df_iv['simulated_instrument'] = np.random.choice([0, 1], size=len(df_iv), p=[0.6, 0.4])
# Adjust GDP based on the simulated instrument to create correlation
df_iv.loc[df_iv['simulated_instrument'] == 1, gdp_per_capita] += np.random.normal(0.5, 0.2, size=(df_iv['simulated_instrument'] == 1).sum())
df_iv.loc[df_iv['simulated_instrument'] == 0, gdp_per_capita] += np.random.normal(-0.5, 0.2, size=(df_iv['simulated_instrument'] == 0).sum())
# Re-standardize GDP after simulation to maintain scale
df_iv[gdp_per_capita] = StandardScaler().fit_transform(df_iv[[gdp_per_capita]])

# Exogenous covariates (controls)
exog_controls = [social_support, healthy_life_expectancy, freedom, generosity, corruption]

# Stage 1: Regress endogenous variable (GDP) on instrument and other exogenous covariates
formula_stage1 = f"{gdp_per_capita} ~ simulated_instrument + {' + '.join(exog_controls)}"
model_stage1 = smf.ols(formula_stage1, data=df_iv).fit()
df_iv['gdp_predicted'] = model_stage1.predict(df_iv)

# Stage 2: Regress outcome (Happiness) on predicted endogenous variable and exogenous covariates
formula_stage2 = f"{happiness_score} ~ gdp_predicted + {' + '.join(exog_controls)}"
model_stage2 = smf.ols(formula_stage2, data=df_iv).fit()

print("IV (2SLS) Results:")
print(model_stage2.summary().tables[1]) # Coefficients table

# Propensity Score Matching (PSM)
print("\n--- Propensity Score Matching (PSM) ---")
# Define "treatment" as high GDP (e.g., top 50% of countries by GDP)
df_psm = df_processed.copy() # Use a copy for PSM
df_psm['treatment'] = (df_psm[gdp_per_capita] > df_psm[gdp_per_capita].median()).astype(int)

# Covariates for propensity score model (variables that influence both treatment and outcome)
ps_covariates = [social_support, healthy_life_expectancy, freedom, generosity, corruption]

# Logistic regression to estimate propensity scores
X_ps = df_psm[ps_covariates]
y_ps = df_psm['treatment']

# Add a constant for statsmodels logistic regression
X_ps = sm.add_constant(X_ps)

ps_model = sm.Logit(y_ps, X_ps).fit(disp=0) # disp=0 to suppress convergence messages
df_psm['propensity_score'] = ps_model.predict(X_ps)

# Matching (Nearest Neighbor Matching)
treated_df = df_psm[df_psm['treatment'] == 1].copy()
control_df = df_psm[df_psm['treatment'] == 0].copy()

# Use NearestNeighbors to find matches
nn = NearestNeighbors(n_neighbors=1, algorithm='ball_tree')
nn.fit(control_df[['propensity_score']])
distances, indices = nn.kneighbors(treated_df[['propensity_score']])

# Create matched control group
matched_control_indices = control_df.index[indices.flatten()]
matched_control_df = df_psm.loc[matched_control_indices]

# Combine treated and matched control groups
matched_df = pd.concat([treated_df, matched_control_df])

# Calculate ATE on matched sample
ate_matched = matched_df.groupby('treatment')[happiness_score].mean()
ate_effect = ate_matched[1] - ate_matched[0]

print(f"Propensity Score Matching - Average Treatment Effect (High GDP vs Low GDP): {ate_effect:.4f}")

# Perform t-test on matched samples for significance
t_stat, p_value = ttest_ind(treated_df[happiness_score], matched_control_df[happiness_score])
print(f"T-test on matched samples: t-statistic = {t_stat:.4f}, p-value = {p_value:.4f}")
if p_value < 0.05:
    print("The difference in happiness between high and low GDP groups (after matching) is statistically significant.")
else:
    print("The difference in happiness between high and low GDP groups (after matching) is NOT statistically significant.")


# --- 5. Bayesian Regression with Informative Priors (Example) ---
# The hierarchical model already demonstrates this. Here's a simpler, non-hierarchical example.
print("\n--- 5. Bayesian Regression with Informative Priors (Example) ---")
with pm.Model() as bayesian_regression_informative:
    # Informative priors for coefficients
    # GDP: Positive, centered at 0.5, sd 0.2
    beta_gdp = pm.Normal('beta_gdp', mu=0.5, sd=0.2)
    # Social Support: Positive, centered at 0.7, sd 0.2
    beta_social = pm.Normal('beta_social', mu=0.7, sd=0.2)
    # Intercept: Centered around mean of happiness score
    intercept = pm.Normal('intercept', mu=y_processed.mean(), sd=y_processed.std())

    # Expected value
    mu_pred = (intercept +
               beta_gdp * df_processed[gdp_per_capita].values +
               beta_social * df_processed[social_support].values)

    # Likelihood
    sigma = pm.HalfNormal('sigma', sd=1)
    happiness_likelihood = pm.Normal('happiness_likelihood', mu=mu_pred, sd=sigma, observed=y_processed)

    trace_informative = pm.sample(1000, tune=500, cores=2, return_inferencedata=True, random_seed=42)

print("Bayesian Regression with Informative Priors Summary:")
print(az.summary(trace_informative, var_names=['beta_gdp', 'beta_social', 'intercept', 'sigma']))


# --- 6. Gaussian Processes for Non-parametric Modeling ---
print("\n--- 6. Gaussian Processes for Non-parametric Modeling ---")
# Model happiness as a non-linear function of GDP using a GP.
# This captures complex, non-linear relationships without specifying a functional form.

X_gdp_gp = df_processed[gdp_per_capita].values[:, None] # Needs to be 2D for GP
y_happiness_gp = y_processed

with pm.Model() as gp_model:
    # Squared Exponential kernel (common choice)
    # Lengthscale (ls): how quickly the function changes.
    # Amplitude (eta): overall magnitude of the function.
    ls = pm.Gamma("ls", alpha=2, beta=1)
    eta = pm.HalfCauchy("eta", beta=1)
    cov = eta**2 * pm.gp.cov.ExpQuad(1, ls)

    # Mean function (e.g., a constant mean or linear trend)
    mean_func = pm.gp.mean.Constant(pm.Normal("c", mu=y_happiness_gp.mean(), sd=y_happiness_gp.std()))

    # Gaussian Process
    gp = pm.gp.Latent(cov_func=cov, mean_func=mean_func)

    # Conditional distribution of the GP at observed points
    f = gp.prior("f", X=X_gdp_gp)

    # Likelihood (add noise to the GP output)
    sigma = pm.HalfNormal("sigma", sd=0.5)
    y_obs = pm.Normal("y_obs", mu=f, sd=sigma, observed=y_happiness_gp)

    trace_gp = pm.sample(1000, tune=500, cores=2, return_inferencedata=True, random_seed=42)

print("Gaussian Process Model Summary:")
print(az.summary(trace_gp, var_names=['ls', 'eta', 'c', 'sigma']))

# Plot GP predictions
print("\n--- Plotting Gaussian Process Predictions ---")
# Create new points for prediction
X_new = np.linspace(X_gdp_gp.min(), X_gdp_gp.max(), 100)[:, None]

with gp_model:
    # Predict f (latent function) at new points
    f_pred = gp.conditional("f_pred", X_new)
    # Sample from the posterior predictive distribution
    gp_posterior_predictive = pm.sample_posterior_predictive(trace_gp, var_names=["f_pred", "y_obs"], samples=500)

# Plot the GP fit
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(X_gdp_gp, y_happiness_gp, "o", markersize=4, alpha=0.5, label="Observed data")
az.plot_gp_posterior(gp_posterior_predictive, X_new, ax=ax, plot_kwargs={"color": "red", "alpha": 0.8})
ax.set_xlabel(gdp_per_capita + " (Standardized)")
ax.set_ylabel(happiness_score)
ax.set_title("Gaussian Process Regression: Happiness vs. GDP")
plt.tight_layout()
plt.savefig('gp_happiness_gdp.png')
plt.close()


# --- 7. Mediation Analysis ---
print("\n--- 7. Mediation Analysis ---")
# Using the results from the approximated SEM models (sem_gdp_social and sem_social_gdp_happiness)
# We want to understand how Social Support mediates the GDP-Happiness relationship.

# Total effect of GDP on Happiness = Direct Effect + Indirect Effect
# Indirect Effect = (GDP -> Social Support) * (Social Support -> Happiness)

# Extract posterior samples for relevant parameters
beta_gdp_to_social_samples = trace_gdp_social.posterior['beta_gdp_to_social'].values.flatten()
beta_social_to_happiness_samples = trace_social_gdp_happiness.posterior['beta_social_to_happiness'].values.flatten()
beta_gdp_to_happiness_direct_samples = trace_social_gdp_happiness.posterior['beta_gdp_to_happiness_direct'].values.flatten()

# Calculate indirect effect samples
indirect_effect_samples = beta_gdp_to_social_samples * beta_social_to_happiness_samples

# Calculate total effect samples
total_effect_samples = beta_gdp_to_happiness_direct_samples + indirect_effect_samples

print("\nMediation Analysis Results (Posterior Means and 95% HDI):")
print(f"Direct Effect (GDP -> Happiness): {np.mean(beta_gdp_to_happiness_direct_samples):.4f} "
      f"(95% HDI: {az.hdi(beta_gdp_to_happiness_direct_samples, hdi_prob=0.95)})")
print(f"Indirect Effect (GDP -> Social Support -> Happiness): {np.mean(indirect_effect_samples):.4f} "
      f"(95% HDI: {az.hdi(indirect_effect_samples, hdi_prob=0.95)})")
print(f"Total Effect (GDP -> Happiness): {np.mean(total_effect_samples):.4f} "
      f"(95% HDI: {az.hdi(total_effect_samples, hdi_prob=0.95)})")

# Plot distributions of effects
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
sns.histplot(beta_gdp_to_happiness_direct_samples, kde=True, ax=axes[0], color='skyblue')
axes[0].set_title('Direct Effect (GDP -> Happiness)')
axes[0].set_xlabel('Effect Size')

sns.histplot(indirect_effect_samples, kde=True, ax=axes[1], color='lightcoral')
axes[1].set_title('Indirect Effect (GDP -> Social Support -> Happiness)')
axes[1].set_xlabel('Effect Size')

sns.histplot(total_effect_samples, kde=True, ax=axes[2], color='lightgreen')
axes[2].set_title('Total Effect (GDP -> Happiness)')
axes[2].set_xlabel('Effect Size')

plt.tight_layout()
plt.savefig('mediation_effects.png')
plt.close()


# --- 8. Difference-in-Differences (DiD) Analysis (Simulated Time Component) ---
print("\n--- 8. Difference-in-Differences (DiD) Analysis ---")
# Simulate a time component and a policy intervention.
# Assume a policy was implemented in some countries (treatment group) between 'pre' and 'post' periods.
# We need to create a synthetic panel dataset.

# Create a copy of the processed data for simulation
df_did = df_processed.copy()

# Simulate 'pre' and 'post' periods
# Duplicate each row to represent two time points
df_did_pre = df_did.copy()
df_did_post = df_did.copy()

df_did_pre['time'] = 0 # Pre-period
df_did_post['time'] = 1 # Post-period

# Combine into a single panel dataset
df_panel = pd.concat([df_did_pre, df_did_post], ignore_index=True)

# Simulate a 'treatment' group (e.g., top 30% of countries by GDP as of 2021)
# These countries receive a 'policy' in the post-period.
gdp_threshold = df_panel[gdp_per_capita].quantile(0.7)
treated_countries = df_panel[df_panel[gdp_per_capita] >= gdp_threshold]['Country_name'].unique()

df_panel['treated'] = df_panel['Country_name'].isin(treated_countries).astype(int)

# Simulate the policy effect: increase happiness in treated group in post-period
# Add some noise to make it realistic
policy_effect = 0.5 # Simulated increase in happiness due to policy
df_panel.loc[(df_panel['treated'] == 1) & (df_panel['time'] == 1), happiness_score] += policy_effect + np.random.normal(0, 0.1, size=(df_panel['treated'] == 1) & (df_panel['time'] == 1)).sum()
# Add some random noise to control group as well, but no systematic increase
df_panel.loc[(df_panel['treated'] == 0) & (df_panel['time'] == 1), happiness_score] += np.random.normal(0, 0.05, size=(df_panel['treated'] == 0) & (df_panel['time'] == 1)).sum()

# DiD Regression: Happiness ~ treated * time + treated + time + controls
# The coefficient of 'treated:time' interaction term is the DiD estimator.
did_formula = f"{happiness_score} ~ treated * time + {' + '.join(predictors)}"
did_model = smf.ols(did_formula, data=df_panel).fit()

print("\nDifference-in-Differences (DiD) Regression Results:")
print(did_model.summary().tables[1]) # Coefficients table

# The key coefficient is 'treated:time'
did_estimator = did_model.params['treated:time']
print(f"\nDiD Estimator (Effect of Policy): {did_estimator:.4f}")


# --- 9. Bayesian Model Comparison (WAIC, LOO-CV) ---
print("\n--- 9. Bayesian Model Comparison (WAIC, LOO-CV) ---")
# Compare the Hierarchical Model with a non-hierarchical model using the same predictors.

with pm.Model() as non_hierarchical_model:
    # Fixed effects (population-level parameters) with informative priors
    beta_gdp = pm.Normal('beta_gdp', mu=0.5, sd=0.2)
    beta_social = pm.Normal('beta_social', mu=0.7, sd=0.2)
    beta_health = pm.Normal('beta_health', mu=0.4, sd=0.2)
    beta_freedom = pm.Normal('beta_freedom', mu=0.3, sd=0.2)
    beta_generosity = pm.Normal('beta_generosity', mu=0.1, sd=0.1)
    beta_corruption = pm.Normal('beta_corruption', mu=-0.2, sd=0.1)

    intercept = pm.Normal('intercept', mu=y_processed.mean(), sd=y_processed.std())

    # Expected value of happiness (linear model without random effects)
    mu_happiness_pred = (intercept +
                         beta_gdp * df_processed[gdp_per_capita].values +
                         beta_social * df_processed[social_support].values +
                         beta_health * df_processed[healthy_life_expectancy].values +
                         beta_freedom * df_processed[freedom].values +
                         beta_generosity * df_processed[generosity].values +
                         beta_corruption * df_processed[corruption].values)

    sigma = pm.HalfNormal('sigma', sd=1.)
    happiness_likelihood = pm.Normal('happiness_likelihood', mu=mu_happiness_pred, sd=sigma, observed=y_processed)

    trace_non_hierarchical = pm.sample(2000, tune=1000, cores=2, return_inferencedata=True, random_seed=42)

print("\nNon-Hierarchical Model Summary:")
print(az.summary(trace_non_hierarchical, var_names=['beta_gdp', 'beta_social', 'beta_health',
                                                    'beta_freedom', 'beta_generosity', 'beta_corruption',
                                                    'sigma', 'intercept']))

# Compare models using ArviZ
compare_df = az.compare({
    "hierarchical_model": trace_hierarchical,
    "non_hierarchical_model": trace_non_hierarchical
}, ic="waic", scale="log") # Use log scale for WAIC/LOO for better interpretation of differences

print("\nBayesian Model Comparison (WAIC & LOO-CV):")
print(compare_df)

# Plot comparison
az.plot_compare(compare_df, insample_dev=False) # insample_dev=False to plot LOO/WAIC
plt.title("Model Comparison (WAIC/LOO-CV)")
plt.tight_layout()
plt.savefig('model_comparison.png')
plt.close()

print("\nBayes Factors are not directly computed here due to complexity, but WAIC/LOO-CV provide robust model comparison.")


# --- 10. Credible Intervals, Posterior Predictive Checks, and Sensitivity Analysis ---
print("\n--- 10. Credible Intervals, Posterior Predictive Checks, and Sensitivity Analysis ---")

# Credible Intervals (already shown in az.summary output)
print("\nCredible Intervals (95% HDI) for Hierarchical Model parameters:")
print(az.summary(trace_hierarchical, hdi_prob=0.95, var_names=['beta_gdp', 'beta_social', 'beta_health',
                                                              'beta_freedom', 'beta_generosity', 'beta_corruption',
                                                              'mu_a', 'sigma_a', 'sigma', 'intercept']))

# Posterior Predictive Checks (PPC) (already done for Hierarchical Model)
# See 'hierarchical_ppc.png'

# Sensitivity Analysis: Varying Priors
# Let's re-run the simple Bayesian regression with less informative (wider) priors
print("\n--- Sensitivity Analysis: Wider Priors ---")
with pm.Model() as bayesian_regression_wider_priors:
    # Wider priors for coefficients
    beta_gdp = pm.Normal('beta_gdp', mu=0, sd=1) # Wider SD
    beta_social = pm.Normal('beta_social', mu=0, sd=1) # Wider SD
    intercept = pm.Normal('intercept', mu=y_processed.mean(), sd=y_processed.std()*2) # Wider SD

    mu_pred = (intercept +
               beta_gdp * df_processed[gdp_per_capita].values +
               beta_social * df_processed[social_support].values)

    sigma = pm.HalfNormal('sigma', sd=2) # Wider SD
    happiness_likelihood = pm.Normal('happiness_likelihood', mu=mu_pred, sd=sigma, observed=y_processed)

    trace_wider_priors = pm.sample(1000, tune=500, cores=2, return_inferencedata=True, random_seed=42)

print("Bayesian Regression with Wider Priors Summary:")
print(az.summary(trace_wider_priors, var_names=['beta_gdp', 'beta_social', 'intercept', 'sigma']))

# Compare posteriors from informative vs. wider priors
print("\nComparing Posteriors (Informative vs. Wider Priors):")
# Plotting both traces on the same plot for comparison
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
az.plot_posterior(trace_informative, var_names=['beta_gdp'], ax=axes[0], color='skyblue', hdi_prob=0.95, point_estimate='mean', textsize=10)
az.plot_posterior(trace_wider_priors, var_names=['beta_gdp'], ax=axes[0], color='lightcoral', hdi_prob=0.95, point_estimate='mean', textsize=10)
axes[0].set_title('Posterior of beta_gdp')
axes[0].legend(['Informative Priors', 'Wider Priors'])

az.plot_posterior(trace_informative, var_names=['beta_social'], ax=axes[1], color='skyblue', hdi_prob=0.95, point_estimate='mean', textsize=10)
az.plot_posterior(trace_wider_priors, var_names=['beta_social'], ax=axes[1], color='lightcoral', hdi_prob=0.95, point_estimate='mean', textsize=10)
axes[1].set_title('Posterior of beta_social')
axes[1].legend(['Informative Priors', 'Wider Priors'])

plt.suptitle("Posterior Comparison: Informative vs. Wider Priors", y=1.02)
plt.tight_layout()
plt.savefig('prior_sensitivity_comparison.png')
plt.close()

print("\n--- All analyses complete. Check generated PNG files for plots. ---")
```