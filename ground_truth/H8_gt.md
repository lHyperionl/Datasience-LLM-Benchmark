```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

# Scikit-learn for imputation, preprocessing, models, and anomaly detection
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.experimental import enable_iterative_imputer # Required for IterativeImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, classification_report, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

# Statsmodels for advanced statistical models and diagnostics
import statsmodels.api as sm
from statsmodels.formula.api import ols, glm
from statsmodels.multivariate.manova import MANOVA
from statsmodels.graphics.gofplots import qqplot
from statsmodels.graphics.regressionplots import plot_leverage_resid2, influence_plot
from statsmodels.robust.robust_linear_model import RLM
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod import families

# PyGAM for Generalized Additive Models
from pygam import LogisticGAM, s, f

# PyMC3 for Bayesian modeling
import pymc3 as pm
import arviz as az

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# --- 1. Load Data ---
print("--- 1. Loading Data ---")
try:
    df = pd.read_csv('water_potability.csv')
    print("Dataset loaded successfully.")
    print(f"Initial shape: {df.shape}")
    print("Missing values before imputation:\n", df.isnull().sum())
except FileNotFoundError:
    print("Error: water_potability.csv not found. Please ensure the file is in the current directory.")
    exit()

# Separate target variable
X = df.drop('Potability', axis=1)
y = df['Potability']

# --- 2. Implement multiple missing data imputation techniques ---
print("\n--- 2. Missing Data Imputation Techniques ---")

# Create copies for different imputation methods
df_knn_imputed = df.copy()
df_iterative_imputed = df.copy()

# Identify columns with missing values
missing_cols = df.columns[df.isnull().any()].tolist()
print(f"Columns with missing values: {missing_cols}")

# 2.1 KNN Imputation
print("\n2.1 Performing KNN Imputation...")
knn_imputer = KNNImputer(n_neighbors=5)
df_knn_imputed[missing_cols] = knn_imputer.fit_transform(df_knn_imputed[missing_cols])
print("KNN Imputation complete. Missing values:\n", df_knn_imputed.isnull().sum())

# 2.2 Iterative Imputation (MICE-like)
print("\n2.2 Performing Iterative Imputation (MICE-like)...")
# IterativeImputer uses BayesianRidge by default, which is suitable for MICE-like imputation
iterative_imputer = IterativeImputer(max_iter=10, random_state=42)
df_iterative_imputed[missing_cols] = iterative_imputer.fit_transform(df_iterative_imputed[missing_cols])
print("Iterative Imputation complete. Missing values:\n", df_iterative_imputed.isnull().sum())

# 2.3 Compare effectiveness (basic descriptive statistics)
print("\n2.3 Comparing Imputation Effectiveness (Descriptive Statistics of Imputed Columns):")
for col in missing_cols:
    print(f"\n--- Column: {col} ---")
    print("Original (before imputation, NaN excluded):\n", df[col].describe())
    print("KNN Imputed:\n", df_knn_imputed[col].describe())
    print("Iterative Imputed:\n", df_iterative_imputed[col].describe())

# For subsequent analysis, we'll use the Iterative Imputed dataset as it's generally robust
df_imputed = df_iterative_imputed.copy()
X_imputed = df_imputed.drop('Potability', axis=1)
y_imputed = df_imputed['Potability']
print("\nSelected Iterative Imputed dataset for further analysis.")

# --- 3. Data Preprocessing (Scaling) ---
print("\n--- 3. Data Preprocessing (Scaling) ---")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)
X_scaled_df = pd.DataFrame(X_scaled, columns=X_imputed.columns)
print("Features scaled using StandardScaler.")

# Split data for modeling
X_train, X_test, y_train, y_test = train_test_split(X_scaled_df, y_imputed, test_size=0.3, random_state=42, stratify=y_imputed)
print(f"Train set shape: {X_train.shape}, Test set shape: {X_test.shape}")

# --- 4. Principal Component Analysis (PCA) and Factor Analysis ---
print("\n--- 4. Dimensionality Reduction: PCA and Factor Analysis ---")

# 4.1 Principal Component Analysis (PCA)
print("\n4.1 Performing PCA...")
pca = PCA(n_components=0.95) # Retain 95% of variance
X_pca = pca.fit_transform(X_scaled_df)
print(f"Number of components selected by PCA (95% variance): {pca.n_components_}")
print(f"Explained variance ratio per component:\n {pca.explained_variance_ratio_}")
print(f"Total explained variance: {pca.explained_variance_ratio_.sum():.2f}")

plt.figure(figsize=(10, 6))
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('PCA: Cumulative Explained Variance')
plt.grid(True)
plt.show()

# 4.2 Factor Analysis
print("\n4.2 Performing Factor Analysis...")
# Factor analysis assumes underlying latent factors. Number of components needs to be specified.
# Let's try to extract 3 factors as an example.
n_factors = 3
fa = FactorAnalysis(n_components=n_factors, random_state=42)
X_fa = fa.fit_transform(X_scaled_df)
print(f"Factor Analysis performed with {n_factors} components.")
print("Factor Loadings (first 5 features):\n", pd.DataFrame(fa.components_, columns=X_scaled_df.columns).iloc[:, :5])

# --- 5. Build Multivariate Statistical Models ---
print("\n--- 5. Building Multivariate Statistical Models ---")

# Add constant for statsmodels models
X_train_sm = sm.add_constant(X_train)
X_test_sm = sm.add_constant(X_test)

# 5.1 Logistic Regression with Regularization
print("\n5.1 Logistic Regression with Regularization (L1 and L2) ---")
# L1 Regularization (Lasso)
log_reg_l1 = LogisticRegression(penalty='l1', solver='liblinear', C=0.1, random_state=42)
log_reg_l1.fit(X_train, y_train)
y_pred_l1 = log_reg_l1.predict(X_test)
y_prob_l1 = log_reg_l1.predict_proba(X_test)[:, 1]

print("\nLogistic Regression (L1 Regularization):")
print(f"ROC AUC: {roc_auc_score(y_test, y_prob_l1):.4f}")
print("Classification Report:\n", classification_report(y_test, y_pred_l1))
print("Coefficients (L1):\n", pd.Series(log_reg_l1.coef_[0], index=X_train.columns))

# L2 Regularization (Ridge)
log_reg_l2 = LogisticRegression(penalty='l2', solver='liblinear', C=0.1, random_state=42)
log_reg_l2.fit(X_train, y_train)
y_pred_l2 = log_reg_l2.predict(X_test)
y_prob_l2 = log_reg_l2.predict_proba(X_test)[:, 1]

print("\nLogistic Regression (L2 Regularization):")
print(f"ROC AUC: {roc_auc_score(y_test, y_prob_l2):.4f}")
print("Classification Report:\n", classification_report(y_test, y_pred_l2))
print("Coefficients (L2):\n", pd.Series(log_reg_l2.coef_[0], index=X_train.columns))

# 5.2 Generalized Additive Models (GAM)
print("\n--- 5.2 Generalized Additive Models (GAM) ---")
# Define GAM formula using smooth terms (s) for continuous features and factor terms (f) for categorical (if any)
# All features are continuous here, so use 's' for all.
gam_formula = y_train.name + ' ~ ' + ' + '.join([f's({col})' for col in X_train.columns])

# For LogisticGAM, the target must be 0 or 1
gam = LogisticGAM().fit(X_train, y_train)
y_prob_gam = gam.predict_proba(X_test)
y_pred_gam = (y_prob_gam > 0.5).astype(int)

print("\nGeneralized Additive Model (LogisticGAM):")
print(f"ROC AUC: {roc_auc_score(y_test, y_prob_gam):.4f}")
print("Classification Report:\n", classification_report(y_test, y_pred_gam))

# Plot partial dependencies for GAM
plt.figure(figsize=(15, 10))
for i, feature in enumerate(X_train.columns):
    plt.subplot(3, 3, i + 1)
    XX = gam.generate_X_grid(term=i)
    pdep, confi = gam.partial_dependence(term=i, X=XX, width=0.95)
    plt.plot(XX[:, i], pdep)
    plt.plot(XX[:, i], confi, c='r', ls='--')
    plt.title(f'Partial Dependence for {feature}')
plt.tight_layout()
plt.suptitle('GAM Partial Dependence Plots', y=1.02, fontsize=16)
plt.show()

# 5.3 Bayesian Logistic Regression using PyMC3
print("\n--- 5.3 Bayesian Logistic Regression using PyMC3 ---")
# For simplicity and speed, we'll use a subset of features for PyMC3
# and a smaller number of draws.
X_pymc3 = X_scaled_df[['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity']]
y_pymc3 = y_imputed

# Split data for PyMC3 model
X_train_pymc3, X_test_pymc3, y_train_pymc3, y_test_pymc3 = train_test_split(
    X_pymc3, y_pymc3, test_size=0.3, random_state=42, stratify=y_pymc3
)

with pm.Model() as bayesian_logistic_model:
    # Priors for the model parameters
    # Coefficients for each feature
    betas = pm.Normal('betas', mu=0, sigma=10, shape=X_train_pymc3.shape[1])
    # Intercept
    alpha = pm.Normal('alpha', mu=0, sigma=10)

    # Linear model
    mu = alpha + pm.math.dot(X_train_pymc3, betas)

    # Logistic link function
    p = pm.Deterministic('p', pm.math.sigmoid(mu))

    # Likelihood (Bernoulli distribution for binary outcome)
    Y_obs = pm.Bernoulli('Y_obs', p=p, observed=y_train_pymc3)

    # Sample from the posterior
    trace = pm.sample(2000, tune=1000, cores=1, random_seed=42, return_inferencedata=True) # Reduced draws for speed

print("\nBayesian Logistic Regression (PyMC3) sampling complete.")

# Plot trace and summary
print("\nPyMC3 Trace Plot:")
az.plot_trace(trace)
plt.tight_layout()
plt.show()

print("\nPyMC3 Posterior Summary:")
print(az.summary(trace, var_names=['betas', 'alpha'], round_to=2))

# Predict on test set using posterior predictive samples
with bayesian_logistic_model:
    pm.set_data({"X_train": X_test_pymc3, "Y_obs": y_test_pymc3}) # Temporarily set data for prediction
    ppc = pm.sample_posterior_predictive(trace, var_names=['p'], samples=1000) # Use 'p' for probabilities

# Calculate mean probabilities and ROC AUC
y_prob_bayesian = ppc['p'].mean(axis=0)
y_pred_bayesian = (y_prob_bayesian > 0.5).astype(int)

print("\nBayesian Logistic Regression (PyMC3) Performance:")
print(f"ROC AUC: {roc_auc_score(y_test_pymc3, y_prob_bayesian):.4f}")
print("Classification Report:\n", classification_report(y_test_pymc3, y_pred_bayesian))

# 5.4 Robust Statistical Methods (RLM for demonstration on a continuous variable)
print("\n--- 5.4 Robust Statistical Methods (RLM) ---")
# Since 'Potability' is binary, RLM (Robust Linear Model) is not directly for classification.
# We'll demonstrate RLM by modeling a continuous variable (e.g., 'ph') against others
# to show its capability in handling outliers in a regression context.
# We'll use the original (imputed) features for this.
X_rlm = df_imputed.drop(['Potability', 'ph'], axis=1)
y_rlm = df_imputed['ph']

# Scale X_rlm for RLM
scaler_rlm = StandardScaler()
X_rlm_scaled = scaler_rlm.fit_transform(X_rlm)
X_rlm_scaled_df = pd.DataFrame(X_rlm_scaled, columns=X_rlm.columns)
X_rlm_scaled_df = sm.add_constant(X_rlm_scaled_df)

print("\nModeling 'ph' using Robust Linear Model (RLM):")
rlm_model = RLM(y_rlm, X_rlm_scaled_df, M=sm.robust.norms.HuberT()) # HuberT norm for robustness
rlm_results = rlm_model.fit()
print(rlm_results.summary())

# Compare with OLS
ols_model = sm.OLS(y_rlm, X_rlm_scaled_df)
ols_results = ols_model.fit()
print("\nModeling 'ph' using Ordinary Least Squares (OLS) for comparison:")
print(ols_results.summary())
print("\nRLM provides robust estimates less sensitive to outliers compared to OLS.")

# --- 6. Model Diagnostics ---
print("\n--- 6. Model Diagnostics (for Logistic Regression L2) ---")

# For logistic regression, residuals are not normally distributed.
# We'll use deviance residuals or Pearson residuals.
# Let's use statsmodels GLM for diagnostics as it provides more direct access to diagnostics.
# Re-fit Logistic Regression using statsmodels GLM
formula = 'Potability ~ ' + ' + '.join(X_train.columns)
data_train_sm = pd.concat([y_train, X_train], axis=1)
glm_model = GLM(data_train_sm['Potability'], sm.add_constant(data_train_sm.drop('Potability', axis=1)),
                family=families.Binomial())
glm_results = glm_model.fit()

print("\nGLM (Logistic Regression) Summary for Diagnostics:")
print(glm_results.summary())

# 6.1 Residual Analysis
print("\n6.1 Residual Analysis:")
# Deviance residuals
deviance_residuals = glm_results.resid_deviance
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(glm_results.fittedvalues, deviance_residuals, alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Fitted Probabilities')
plt.ylabel('Deviance Residuals')
plt.title('Deviance Residuals vs. Fitted Probabilities')

# Q-Q plot of deviance residuals (for normality assumption check, though not strictly normal for GLMs)
plt.subplot(1, 2, 2)
qqplot(deviance_residuals, line='s', ax=plt.gca())
plt.title('Q-Q Plot of Deviance Residuals')
plt.tight_layout()
plt.show()

# 6.2 Influence Measures
print("\n6.2 Influence Measures (Cook's Distance, Leverage):")
# Influence plot (Cook's distance, Leverage)
fig, ax = plt.subplots(figsize=(12, 8))
influence_plot(glm_results, ax=ax)
ax.set_title("Influence Plot (Cook's Distance vs. Leverage)")
plt.show()

# Identify highly influential points (e.g., Cook's distance > 4/n)
cooks_d = glm_results.get_influence().cooks_distance[0]
n_obs = len(data_train_sm)
influential_points = data_train_sm.index[cooks_d > 4/n_obs]
print(f"Number of influential points (Cook's D > 4/n): {len(influential_points)}")
if not influential_points.empty:
    print("Indices of influential points:\n", influential_points.tolist())

# 6.3 Goodness-of-Fit Tests (for Logistic Regression)
print("\n6.3 Goodness-of-Fit Tests:")
# Hosmer-Lemeshow Test (requires custom implementation or specific package)
# For simplicity, we'll rely on ROC AUC and classification report as primary GOF for classification.
# Deviance and AIC/BIC are also reported in GLM summary.
print(f"Model Deviance: {glm_results.deviance:.2f}")
print(f"Model AIC: {glm_results.aic:.2f}")
print(f"Model BIC: {glm_results.bic:.2f}")
print("Lower Deviance, AIC, BIC generally indicate better fit.")

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob_l2) # Using L2 Logistic Regression probabilities
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'Logistic Regression (AUC = {roc_auc_score(y_test, y_prob_l2):.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='No Skill')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid(True)
plt.show()

# --- 7. Advanced Hypothesis Testing ---
print("\n--- 7. Advanced Hypothesis Testing ---")

# 7.1 Multivariate ANOVA (MANOVA)
print("\n7.1 Multivariate ANOVA (MANOVA) ---")
# MANOVA tests if group means differ across multiple dependent variables simultaneously.
# Here, 'Potability' is the grouping variable, and other features are dependent variables.
# MANOVA requires no missing values, so use df_imputed.
# Ensure all columns are numeric for MANOVA.
manova_data = df_imputed.copy()
manova_data['Potability'] = manova_data['Potability'].astype('category') # Ensure Potability is categorical

# Construct formula: dependent_vars ~ grouping_var
# Use all features except Potability as dependent variables
dependent_vars = ' + '.join(manova_data.drop('Potability', axis=1).columns)
manova_formula = f'{dependent_vars} ~ Potability'

try:
    mc = MANOVA.from_formula(manova_formula, data=manova_data)
    print(mc.mv_test())
    print("\nInterpretation: Look at p-values for Wilks' Lambda, Pillai's Trace, Hotelling-Lawley Trace, Roy's Largest Root.")
    print("A significant p-value (e.g., < 0.05) suggests that the group means (potable vs. non-potable) differ significantly across the set of water quality parameters.")
except Exception as e:
    print(f"Error during MANOVA: {e}")
    print("MANOVA might fail if there are singular matrices (e.g., highly correlated features or too few observations per group).")

# 7.2 Permutation Tests
print("\n--- 7.2 Permutation Tests ---")
# Example: Test if 'ph' mean differs significantly between potable and non-potable water.
group1 = df_imputed[df_imputed['Potability'] == 0]['ph'].dropna()
group2 = df_imputed[df_imputed['Potability'] == 1]['ph'].dropna()

observed_diff = np.mean(group1) - np.mean(group2)
n_permutations = 1000
permutation_diffs = []

combined_ph = np.concatenate([group1, group2])
n1 = len(group1)

for _ in range(n_permutations):
    np.random.shuffle(combined_ph)
    perm_group1 = combined_ph[:n1]
    perm_group2 = combined_ph[n1:]
    permutation_diffs.append(np.mean(perm_group1) - np.mean(perm_group2))

p_value_permutation = (np.sum(np.abs(permutation_diffs) >= np.abs(observed_diff))) / n_permutations

print(f"\nPermutation Test for 'ph' mean difference (Potable vs. Non-Potable):")
print(f"Observed mean difference: {observed_diff:.4f}")
print(f"P-value (two-tailed): {p_value_permutation:.4f}")
if p_value_permutation < 0.05:
    print("Conclusion: The difference in 'ph' means is statistically significant (p < 0.05).")
else:
    print("Conclusion: The difference in 'ph' means is not statistically significant (p >= 0.05).")

plt.figure(figsize=(8, 6))
sns.histplot(permutation_diffs, kde=True, bins=30)
plt.axvline(observed_diff, color='red', linestyle='--', label='Observed Difference')
plt.title('Permutation Distribution of Mean Differences for pH')
plt.xlabel('Mean Difference (Non-Potable - Potable)')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# 7.3 Bootstrap Confidence Intervals
print("\n--- 7.3 Bootstrap Confidence Intervals ---")
# Example: Bootstrap CI for the mean of 'Hardness'
data_hardness = df_imputed['Hardness'].dropna()
n_bootstraps = 5000
bootstrap_means = []

for _ in range(n_bootstraps):
    sample = np.random.choice(data_hardness, size=len(data_hardness), replace=True)
    bootstrap_means.append(np.mean(sample))

# Calculate 95% confidence interval
lower_bound = np.percentile(bootstrap_means, 2.5)
upper_bound = np.percentile(bootstrap_means, 97.5)

print(f"\nBootstrap 95% Confidence Interval for Mean 'Hardness':")
print(f"Original Mean: {np.mean(data_hardness):.4f}")
print(f"Bootstrap CI: ({lower_bound:.4f}, {upper_bound:.4f})")

plt.figure(figsize=(8, 6))
sns.histplot(bootstrap_means, kde=True, bins=50)
plt.axvline(np.mean(data_hardness), color='red', linestyle='--', label='Original Mean')
plt.axvline(lower_bound, color='green', linestyle=':', label='2.5th Percentile')
plt.axvline(upper_bound, color='green', linestyle=':', label='97.5th Percentile')
plt.title('Bootstrap Distribution of Mean Hardness')
plt.xlabel('Mean Hardness')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# --- 8. Anomaly Detection Algorithms ---
print("\n--- 8. Anomaly Detection Algorithms ---")
# Use the scaled data for anomaly detection
X_anomaly = X_scaled_df.copy()

# 8.1 Isolation Forest
print("\n8.1 Isolation Forest ---")
iso_forest = IsolationForest(random_state=42, contamination='auto') # 'auto' estimates proportion of outliers
iso_forest_preds = iso_forest.fit_predict(X_anomaly) # -1 for outliers, 1 for inliers
iso_forest_scores = iso_forest.decision_function(X_anomaly)

n_outliers_iso = np.sum(iso_forest_preds == -1)
print(f"Number of anomalies detected by Isolation Forest: {n_outliers_iso}")

# Visualize anomaly scores
plt.figure(figsize=(10, 6))
sns.histplot(iso_forest_scores, bins=50, kde=True)
plt.title('Isolation Forest Anomaly Scores Distribution')
plt.xlabel('Anomaly Score')
plt.ylabel('Frequency')
plt.show()

# 8.2 Local Outlier Factor (LOF)
print("\n8.2 Local Outlier Factor (LOF) ---")
lof = LocalOutlierFactor(n_neighbors=20, contamination='auto')
lof_preds = lof.fit_predict(X_anomaly) # -1 for outliers, 1 for inliers
lof_scores = lof.negative_outlier_factor_ # LOF scores are negative, lower is more anomalous

n_outliers_lof = np.sum(lof_preds == -1)
print(f"Number of anomalies detected by LOF: {n_outliers_lof}")

# Visualize LOF scores
plt.figure(figsize=(10, 6))
sns.histplot(lof_scores, bins=50, kde=True)
plt.title('Local Outlier Factor (LOF) Scores Distribution')
plt.xlabel('LOF Score (lower is more anomalous)')
plt.ylabel('Frequency')
plt.show()

# 8.3 One-Class SVM
print("\n8.3 One-Class SVM ---")
# One-Class SVM is sensitive to scaling and kernel choice.
# Use a linear kernel for simplicity or RBF for non-linear boundaries.
# 'nu' parameter is an upper bound on the fraction of training errors and a lower bound of the fraction of support vectors.
# It's also an upper bound on the fraction of outliers.
oc_svm = OneClassSVM(kernel='rbf', nu=0.05) # nu is an estimate of the fraction of outliers
oc_svm.fit(X_anomaly)
oc_svm_preds = oc_svm.predict(X_anomaly) # -1 for outliers, 1 for inliers
oc_svm_scores = oc_svm.decision_function(X_anomaly)

n_outliers_ocsvm = np.sum(oc_svm_preds == -1)
print(f"Number of anomalies detected by One-Class SVM: {n_outliers_ocsvm}")

# Visualize One-Class SVM scores
plt.figure(figsize=(10, 6))
sns.histplot(oc_svm_scores, bins=50, kde=True)
plt.title('One-Class SVM Anomaly Scores Distribution')
plt.xlabel('Decision Function Score (lower is more anomalous)')
plt.ylabel('Frequency')
plt.show()

print("\n--- Anomaly Detection Summary ---")
print(f"Isolation Forest detected {n_outliers_iso} anomalies.")
print(f"Local Outlier Factor detected {n_outliers_lof} anomalies.")
print(f"One-Class SVM detected {n_outliers_ocsvm} anomalies.")

# --- 9. Advanced Statistical Visualizations (Dashboard-like) ---
print("\n--- 9. Advanced Statistical Visualizations (Dashboard-like) ---")

# 9.1 Pairplot with Potability (for initial EDA)
print("\n9.1 Pairplot of selected features by Potability:")
# Select a subset of features for readability in pairplot
selected_features = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Potability']
sns.pairplot(df_imputed[selected_features], hue='Potability', diag_kind='kde', palette='viridis')
plt.suptitle('Pairplot of Selected Water Quality Parameters by Potability', y=1.02)
plt.show()

# 9.2 Correlation Heatmap
print("\n9.2 Correlation Heatmap:")
plt.figure(figsize=(10, 8))
sns.heatmap(df_imputed.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix of Water Quality Parameters')
plt.show()

# 9.3 Box plots for each feature by Potability
print("\n9.3 Box Plots of Features by Potability:")
plt.figure(figsize=(18, 12))
for i, col in enumerate(X_imputed.columns):
    plt.subplot(3, 3, i + 1)
    sns.boxplot(x='Potability', y=col, data=df_imputed, palette='pastel')
    plt.title(f'{col} by Potability')
plt.tight_layout()
plt.suptitle('Distribution of Water Quality Parameters by Potability', y=1.02, fontsize=16)
plt.show()

print("\n--- Analysis Complete ---")
```