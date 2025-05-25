```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import proportional_hazards_test
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from scipy.stats import ttest_ind, chi2_contingency
import statsmodels.api as sm
import pymc3 as pm
import arviz as az
import warnings

# Suppress warnings for cleaner output, especially from PyMC3 and lifelines
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# --- 0. Data Loading and Initial Preprocessing ---
# Load the dataset
try:
    df = pd.read_csv('titanic.csv')
except FileNotFoundError:
    print("Error: titanic.csv not found. Please ensure the file is in the current directory.")
    exit()

# Drop irrelevant columns for this analysis
df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

# Impute missing 'Age' with the median
imputer_age = SimpleImputer(strategy='median')
df['Age'] = imputer_age.fit_transform(df[['Age']])

# Impute missing 'Embarked' with the mode
imputer_embarked = SimpleImputer(strategy='most_frequent')
df['Embarked'] = imputer_embarked.fit_transform(df[['Embarked']])

# Feature Engineering: FamilySize and IsAlone
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

# Convert 'Sex' to numerical (0 for female, 1 for male)
df['Sex'] = df['Sex'].map({'female': 0, 'male': 1})

# One-hot encode 'Embarked' and 'Pclass' for modeling
# Pclass is treated as categorical for some models (e.g., CoxPH, Bayesian)
df_encoded = pd.get_dummies(df, columns=['Embarked', 'Pclass'], drop_first=True)

# Define T (time) and E (event) for survival analysis
# As per prompt, voyage is time-to-event, survival is event indicator.
# Since no specific time of death is given, we assume a fixed time T=1 for all,
# and the event E is death (Survived=0).
# This effectively treats the Cox model as a logistic regression for binary outcomes.
T = np.ones(len(df_encoded)) # Fixed time for all observations
E = 1 - df_encoded['Survived'] # Event is death (0=survived, 1=died)

# Prepare the dataframe for lifelines models
# Drop 'Survived' as it's used to define E
df_lifelines = df_encoded.drop('Survived', axis=1)
df_lifelines['T'] = T
df_lifelines['E'] = E

# --- 1. Kaplan-Meier Survival Analysis ---
print("--- 1. Kaplan-Meier Survival Analysis ---")

kmf = KaplanMeierFitter()
kmf.fit(T, event_observed=E, label="Overall Survival")

plt.figure(figsize=(10, 6))
kmf.plot_survival_function()
plt.title('Kaplan-Meier Survival Curve (Overall)')
plt.xlabel('Time (Fixed Voyage Duration)')
plt.ylabel('Survival Probability')
plt.grid(True)
plt.show()

# Kaplan-Meier by Sex
plt.figure(figsize=(10, 6))
for sex_val in df['Sex'].unique():
    # Use original df for grouping, then get corresponding indices from df_encoded
    subset_indices = df_encoded[df['Sex'] == sex_val].index
    kmf.fit(T[subset_indices], event_observed=E[subset_indices], label=f'Sex: {"Female" if sex_val == 0 else "Male"}')
    kmf.plot_survival_function()
plt.title('Kaplan-Meier Survival Curves by Sex')
plt.xlabel('Time (Fixed Voyage Duration)')
plt.ylabel('Survival Probability')
plt.grid(True)
plt.show()

# Kaplan-Meier by Pclass (using original Pclass for grouping)
plt.figure(figsize=(10, 6))
for pclass_val in sorted(df['Pclass'].unique()):
    # Use original df for grouping, then get corresponding indices from df_encoded
    subset_indices = df_encoded[df['Pclass'] == pclass_val].index
    kmf.fit(T[subset_indices], event_observed=E[subset_indices], label=f'Pclass: {pclass_val}')
    kmf.plot_survival_function()
plt.title('Kaplan-Meier Survival Curves by Pclass')
plt.xlabel('Time (Fixed Voyage Duration)')
plt.ylabel('Survival Probability')
plt.grid(True)
plt.show()

# --- 2. Cox Proportional Hazards Regression Models ---
print("\n--- 2. Cox Proportional Hazards Regression Models ---")

# Define features for Cox model
# Exclude 'Pclass_3' as it's the reference category after one-hot encoding with drop_first=True
cox_features = [col for col in df_lifelines.columns if col not in ['T', 'E', 'Pclass_3']]
# Ensure Pclass_1 and Pclass_2 are included if they exist
if 'Pclass_1' in df_lifelines.columns:
    cox_features.append('Pclass_1')
if 'Pclass_2' in df_lifelines.columns:
    cox_features.append('Pclass_2')
cox_features = list(set(cox_features)) # Remove duplicates if any

# Base Cox model
cph = CoxPHFitter()
cph.fit(df_lifelines[cox_features + ['T', 'E']], duration_col='T', event_col='E')
print("\nBase Cox Proportional Hazards Model Summary:")
cph.print_summary()

# Model Diagnostics: Proportional Hazards Assumption Test
print("\nProportional Hazards Assumption Test (Base Model):")
# Note: When 'T' (duration) is constant for all observations, the proportional hazards
# assumption is trivially met or becomes irrelevant, as the model effectively reduces
# to a logistic regression. The `check_assumptions` method might issue warnings or
# behave unexpectedly in such cases, as it's designed for varying durations.
try:
    cph.check_assumptions(df_lifelines[cox_features + ['T', 'E']], p_value_threshold=0.05)
except Exception as e:
    print(f"Warning: Proportional Hazards Assumption Test might be problematic with constant 'T'. Error: {e}")
    print("This is expected if 'T' is constant, as the test relies on varying time points.")

# Cox model with Interaction Effects (e.g., Age * Sex, Pclass_1 * Sex)
df_lifelines_interaction = df_lifelines.copy()
df_lifelines_interaction['Age_x_Sex'] = df_lifelines_interaction['Age'] * df_lifelines_interaction['Sex']
df_lifelines_interaction['Pclass1_x_Sex'] = df_lifelines_interaction['Pclass_1'] * df_lifelines_interaction['Sex']

interaction_features = cox_features + ['Age_x_Sex', 'Pclass1_x_Sex']
# Ensure no duplicates and all features exist in the dataframe
interaction_features = [f for f in interaction_features if f in df_lifelines_interaction.columns]
interaction_features = list(set(interaction_features))

cph_interaction = CoxPHFitter()
cph_interaction.fit(df_lifelines_interaction[interaction_features + ['T', 'E']], duration_col='T', event_col='E')
print("\nCox Proportional Hazards Model with Interaction Effects Summary:")
cph_interaction.print_summary()

# Model Diagnostics for Interaction Model
print("\nProportional Hazards Assumption Test (Interaction Model):")
try:
    cph_interaction.check_assumptions(df_lifelines_interaction[interaction_features + ['T', 'E']], p_value_threshold=0.05)
except Exception as e:
    print(f"Warning: Proportional Hazards Assumption Test might be problematic with constant 'T'. Error: {e}")
    print("This is expected if 'T' is constant, as the test relies on varying time points.")

# Compare models using Concordance Index (C-index)
print(f"\nC-index for Base Cox Model: {cph.concordance_index_:.4f}")
print(f"C-index for Interaction Cox Model: {cph_interaction.concordance_index_:.4f}")

# --- 3. Propensity Score Matching ---
print("\n--- 3. Propensity Score Matching ---")

# Define treatment: Pclass 1 vs Pclass 3 (excluding Pclass 2 for clearer contrast)
df_psm = df.copy() # Use original df for Pclass definition
df_psm = df_psm[df_psm['Pclass'].isin([1, 3])].reset_index(drop=True)

# Define treatment variable: 1 if Pclass=1, 0 if Pclass=3
df_psm['Treatment'] = (df_psm['Pclass'] == 1).astype(int)

# Covariates for propensity score model (excluding Pclass and Survived)
ps_covariates = ['Age', 'Sex', 'Fare', 'SibSp', 'Parch', 'FamilySize', 'IsAlone']
# One-hot encode Embarked for PSM covariates
df_psm = pd.get_dummies(df_psm, columns=['Embarked'], drop_first=True)
ps_covariates.extend([col for col in df_psm.columns if 'Embarked_' in col])

# Ensure all PSM covariates are numeric and handle any remaining NaNs if any
for col in ps_covariates:
    if df_psm[col].dtype == 'object':
        df_psm[col] = pd.to_numeric(df_psm[col], errors='coerce')
    if df_psm[col].isnull().any():
        df_psm[col] = SimpleImputer(strategy='median').fit_transform(df_psm[[col]])[:,0]

# Standardize covariates for logistic regression
scaler_psm = StandardScaler()
df_psm[ps_covariates] = scaler_psm.fit_transform(df_psm[ps_covariates])

# Estimate propensity scores using Logistic Regression
X_psm = df_psm[ps_covariates]
y_psm = df_psm['Treatment']

log_reg_psm = LogisticRegression(solver='liblinear', random_state=42)
log_reg_psm.fit(X_psm, y_psm)
df_psm['PropensityScore'] = log_reg_psm.predict_proba(X_psm)[:, 1]

# Nearest Neighbor Matching (1-to-1 matching without replacement)
treated = df_psm[df_psm['Treatment'] == 1].copy()
control = df_psm[df_psm['Treatment'] == 0].copy()

# Use NearestNeighbors to find matches
nn = NearestNeighbors(n_neighbors=1, algorithm='kd_tree')
nn.fit(control[['PropensityScore']])
distances, indices = nn.kneighbors(treated[['PropensityScore']])

# Create matched dataframe
matched_control_indices = indices.flatten()
matched_treated = treated.reset_index(drop=True)
matched_control = control.iloc[matched_control_indices].reset_index(drop=True)

df_matched = pd.concat([matched_treated, matched_control], axis=0)
print(f"\nNumber of matched pairs: {len(matched_treated)}")

# Check balance after matching (Standardized Mean Difference - SMD)
print("\nCovariate Balance Check (SMD) after Matching:")
for cov in ps_covariates:
    mean_treated = matched_treated[cov].mean()
    mean_control = matched_control[cov].mean()
    std_treated = matched_treated[cov].std()
    std_control = matched_control[cov].std()
    # Handle cases where std is zero to avoid division by zero
    if std_treated == 0 and std_control == 0:
        smd = 0.0
    else:
        smd = (mean_treated - mean_control) / np.sqrt((std_treated**2 + std_control**2) / 2)
    print(f"  {cov}: SMD = {smd:.4f}")
print("SMD values close to 0 (e.g., < 0.1 or < 0.25) indicate good balance.")

# Analyze treatment effect on survival in matched groups
# Survival rate for treated (Pclass=1) vs control (Pclass=3)
survival_treated_matched = matched_treated['Survived'].mean()
survival_control_matched = matched_control['Survived'].mean()

print(f"\nSurvival Rate (Matched Pclass 1): {survival_treated_matched:.4f}")
print(f"Survival Rate (Matched Pclass 3): {survival_control_matched:.4f}")

# Perform a chi-squared test for survival difference in matched groups (binary outcome)
contingency_table = pd.crosstab(df_matched['Treatment'], df_matched['Survived'])
chi2, p_val, _, _ = chi2_contingency(contingency_table)
print(f"\nChi-squared test for survival difference in matched groups:")
print(f"  Chi2 statistic: {chi2:.4f}")
print(f"  P-value: {p_val:.4f}")
if p_val < 0.05:
    print("  Conclusion: Statistically significant difference in survival between matched Pclass 1 and Pclass 3.")
else:
    print("  Conclusion: No statistically significant difference in survival between matched Pclass 1 and Pclass 3.")

# --- 4. Bootstrap Resampling for Confidence Intervals ---
print("\n--- 4. Bootstrap Resampling for Confidence Intervals ---")

n_bootstraps = 500
bootstrapped_coeffs = []
bootstrapped_c_indices = []

# Use the base Cox model features for bootstrapping
# Ensure the dataframe used for bootstrapping has 'T' and 'E'
df_boot = df_lifelines[cox_features + ['T', 'E']]

print(f"Performing {n_bootstraps} bootstrap iterations for CoxPH model...")
for i in range(n_bootstraps):
    # Sample with replacement
    sample_df = df_boot.sample(n=len(df_boot), replace=True, random_state=i)
    
    try:
        cph_boot = CoxPHFitter()
        cph_boot.fit(sample_df, duration_col='T', event_col='E')
        
        # Store coefficients
        coeffs = cph_boot.summary['coef'].to_dict()
        bootstrapped_coeffs.append(coeffs)
        
        # Store C-index
        bootstrapped_c_indices.append(cph_boot.concordance_index_)
    except Exception as e:
        # Handle cases where a bootstrap sample might lead to convergence issues
        # print(f"Bootstrap iteration {i} failed: {e}")
        continue

# Convert list of dicts to DataFrame for easier analysis
coeffs_df = pd.DataFrame(bootstrapped_coeffs)

print("\nBootstrapped CoxPH Coefficients (Mean and 95% CI):")
for col in coeffs_df.columns:
    mean_coeff = coeffs_df[col].mean()
    lower_ci = coeffs_df[col].quantile(0.025)
    upper_ci = coeffs_df[col].quantile(0.975)
    print(f"  {col}: Mean = {mean_coeff:.4f}, 95% CI = [{lower_ci:.4f}, {upper_ci:.4f}]")

mean_c_index = np.mean(bootstrapped_c_indices)
lower_c_index = np.percentile(bootstrapped_c_indices, 2.5)
upper_c_index = np.percentile(bootstrapped_c_indices, 97.5)
print(f"\nBootstrapped C-index: Mean = {mean_c_index:.4f}, 95% CI = [{lower_c_index:.4f}, {upper_c_index:.4f}]")

# --- 5. Bayesian Analysis using PyMC3 ---
print("\n--- 5. Bayesian Analysis using PyMC3 ---")

# Prepare data for Bayesian Logistic Regression
# Target variable: Survived (0 or 1)
# Predictors: Age, Sex, Fare, Pclass_1, Pclass_2, FamilySize, IsAlone, Embarked_Q, Embarked_S
# Use the df_encoded dataframe which has Survived and one-hot encoded features
bayesian_df = df_encoded.copy()

# Select features for Bayesian model
bayesian_features = ['Age', 'Sex', 'Fare', 'FamilySize', 'IsAlone', 'Pclass_1', 'Pclass_2', 'Embarked_Q', 'Embarked_S']
# Ensure all features exist and are numeric
bayesian_features = [f for f in bayesian_features if f in bayesian_df.columns]
for col in bayesian_features:
    if bayesian_df[col].dtype == 'object':
        bayesian_df[col] = pd.to_numeric(bayesian_df[col], errors='coerce')
    if bayesian_df[col].isnull().any():
        bayesian_df[col] = SimpleImputer(strategy='median').fit_transform(bayesian_df[[col]])[:,0]

# Standardize numerical features for better MCMC sampling
scaler_bayesian = StandardScaler()
bayesian_df[bayesian_features] = scaler_bayesian.fit_transform(bayesian_df[bayesian_features])

X_bayesian = bayesian_df[bayesian_features].values
y_bayesian = bayesian_df['Survived'].values

# Define the Bayesian Logistic Regression model
with pm.Model() as titanic_model:
    # Priors for the regression coefficients
    # Using Normal priors, centered at 0 with a reasonable standard deviation (e.g., 5)
    
    # Intercept prior
    intercept = pm.Normal('intercept', mu=0, sd=5)
    
    # Priors for coefficients of each feature
    coeffs = {}
    for feature in bayesian_features:
        coeffs[feature] = pm.Normal(f'beta_{feature}', mu=0, sd=5)
    
    # Linear model (dot product of features and coefficients)
    linear_predictor = intercept
    for i, feature_name in enumerate(bayesian_features):
        linear_predictor += coeffs[feature_name] * X_bayesian[:, i]
    
    # Link function (logistic/sigmoid)
    p = pm.Deterministic('p', pm.invlogit(linear_predictor))
    
    # Likelihood (Bernoulli distribution for binary outcome)
    Y_obs = pm.Bernoulli('Y_obs', p=p, observed=y_bayesian)

# Sample from the posterior distribution
print("\nSampling from Bayesian model posterior (this may take a few minutes)...")
with titanic_model:
    # Use NUTS sampler, tune for 1000 iterations, sample 2000 iterations
    # Increase draws and tune for more stable results. cores=2 for parallel chains.
    trace = pm.sample(2000, tune=1000, cores=2, random_seed=42, return_inferencedata=True)

print("\nBayesian Model Summary:")
# Filter variables to show only coefficients and intercept
var_names_to_show = [v for v in trace.posterior.data_vars if 'beta_' in v or 'intercept' in v]
print(az.summary(trace, var_names=var_names_to_show))

# Plot posterior distributions and trace plots
print("\nPlotting Bayesian model trace and posterior distributions...")
az.plot_trace(trace, var_names=var_names_to_show)
plt.tight_layout()
plt.show()

az.plot_posterior(trace, var_names=var_names_to_show, kind='hist')
plt.tight_layout()
plt.show()

# --- 6. Model Validation and Sensitivity Analysis ---
print("\n--- 6. Model Validation and Sensitivity Analysis ---")

# 6.1 Imputation Strategy Sensitivity Analysis (Age)
print("\n--- Imputation Strategy Sensitivity Analysis (Age) ---")

# Re-run CoxPH with mean imputation for Age to compare with median imputation
df_mean_imputed = pd.read_csv('titanic.csv')
df_mean_imputed = df_mean_imputed.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
df_mean_imputed['Age'] = SimpleImputer(strategy='mean').fit_transform(df_mean_imputed[['Age']])
df_mean_imputed['Embarked'] = SimpleImputer(strategy='most_frequent').fit_transform(df_mean_imputed[['Embarked']])
df_mean_imputed['FamilySize'] = df_mean_imputed['SibSp'] + df_mean_imputed['Parch'] + 1
df_mean_imputed['IsAlone'] = (df_mean_imputed['FamilySize'] == 1).astype(int)
df_mean_imputed['Sex'] = df_mean_imputed['Sex'].map({'female': 0, 'male': 1})
df_mean_imputed_encoded = pd.get_dummies(df_mean_imputed, columns=['Embarked', 'Pclass'], drop_first=True)

T_mean = np.ones(len(df_mean_imputed_encoded))
E_mean = 1 - df_mean_imputed_encoded['Survived']
df_lifelines_mean = df_mean_imputed_encoded.drop('Survived', axis=1)
df_lifelines_mean['T'] = T_mean
df_lifelines_mean['E'] = E_mean

# Use the same features as the base Cox model
cox_features_mean = [col for col in df_lifelines_mean.columns if col not in ['T', 'E', 'Pclass_3']]
if 'Pclass_1' in df_lifelines_mean.columns:
    cox_features_mean.append('Pclass_1')
if 'Pclass_2' in df_lifelines_mean.columns:
    cox_features_mean.append('Pclass_2')
cox_features_mean = list(set(cox_features_mean))

cph_mean_imputed = CoxPHFitter()
cph_mean_imputed.fit(df_lifelines_mean[cox_features_mean + ['T', 'E']], duration_col='T', event_col='E')

print("\nCoxPH Model Summary with Mean Imputation for Age:")
cph_mean_imputed.print_summary()
print(f"C-index with Mean Imputation: {cph_mean_imputed.concordance_index_:.4f}")

print("\nComparison of Coefficients (Median vs Mean Imputation):")
print("Median Imputation Coefficients (from base model):")
print(cph.summary['coef'])
print("\nMean Imputation Coefficients:")
print(cph_mean_imputed.summary['coef'])
print("\nDifferences in coefficients indicate sensitivity to imputation strategy. Small differences suggest robustness.")

# 6.2 Model Specification Sensitivity (Conceptual)
print("\n--- Model Specification Sensitivity (Conceptual) ---")
print("To perform this, one would build multiple CoxPH models with different sets of covariates")
print("or different interaction terms, and compare their C-indices, AIC/BIC (if available),")
print("and coefficient stability. For example, removing 'Fare' or 'FamilySize' and observing changes.")
print("The interaction model built in Section 2 is an example of varying model specification.")

# 6.3 Propensity Score Matching Balance Check (already done in section 3)
print("\nPropensity Score Matching Balance Check was performed in Section 3.")
print("SMD values close to 0 (e.g., < 0.1 or < 0.25) indicate good balance, suggesting successful matching.")

# 6.4 Bayesian Prior Sensitivity (Conceptual)
print("\n--- Bayesian Prior Sensitivity (Conceptual) ---")
print("To perform sensitivity analysis on priors, one would re-run the Bayesian model")
print("with different prior specifications (e.g., wider or narrower standard deviations for Normal priors).")
print("Then, compare the posterior distributions and conclusions (e.g., mean, credible intervals).")
print("If posteriors change significantly, the model is sensitive to prior choice, suggesting less data information relative to prior.")

print("\n--- End of Analysis ---")
```