```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.feature_selection import SelectKBest, f_regression, RFE, mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split # Included for best practice, though not strictly used for all steps

# Suppress warnings for cleaner output, especially from sklearn and pandas
import warnings
warnings.filterwarnings('ignore')

# --- 1. Load and Preprocess the Dataset ---
# Load the dataset from the specified CSV file
try:
    df = pd.read_csv('world-happiness-report-2021.csv')
except FileNotFoundError:
    print("Error: 'world-happiness-report-2021.csv' not found. Please ensure the file is in the current directory.")
    exit()

# Identify relevant columns for the task to simplify the DataFrame
relevant_cols = [
    'Country name', 'Regional indicator', 'Ladder score', 'Logged GDP per capita',
    'Social support', 'Healthy life expectancy', 'Freedom to make life choices',
    'Generosity', 'Perceptions of corruption'
]
df_processed = df[relevant_cols].copy()

# Handle missing values: 'Perceptions of corruption' often has a few NaNs.
# Fill these missing values with the mean of the column.
if 'Perceptions of corruption' in df_processed.columns:
    df_processed['Perceptions of corruption'].fillna(df_processed['Perceptions of corruption'].mean(), inplace=True)

# Drop any remaining rows with NaNs in the selected core columns to ensure data integrity for calculations.
df_processed.dropna(inplace=True)

# --- 2. Create Engineered Features ---

# 'Happiness_Efficiency': Ladder score / GDP per capita
# This feature aims to capture how much happiness is generated per unit of economic output.
df_processed['Happiness_Efficiency'] = df_processed['Ladder score'] / df_processed['Logged GDP per capita']

# 'Social_Wellness_Index': combination of Social support and Healthy life expectancy
# This index combines two key social factors contributing to well-being.
df_processed['Social_Wellness_Index'] = df_processed['Social support'] + df_processed['Healthy life expectancy']

# 'Governance_Score': combination of Freedom and low Corruption
# 'Perceptions of corruption' is a score where higher values indicate more perceived corruption.
# To make 'low corruption' contribute positively, we use (1 - Perceptions of corruption).
# This score reflects the quality of governance and its impact on happiness.
df_processed['Governance_Score'] = df_processed['Freedom to make life choices'] + (1 - df_processed['Perceptions of corruption'])

# 'Regional_Happiness_Rank': rank within region based on 'Ladder score'
# This feature provides a country's happiness standing relative to its regional peers.
df_processed['Regional_Happiness_Rank'] = df_processed.groupby('Regional indicator')['Ladder score'].rank(ascending=False)

# 'GDP_vs_Regional_Average': difference from regional GDP mean
# This feature highlights how a country's economic output compares to its regional average.
df_processed['Regional_GDP_Mean'] = df_processed.groupby('Regional indicator')['Logged GDP per capita'].transform('mean')
df_processed['GDP_vs_Regional_Average'] = df_processed['Logged GDP per capita'] - df_processed['Regional_GDP_Mean']
# Drop the temporary column used for calculation
df_processed.drop('Regional_GDP_Mean', axis=1, inplace=True)

# --- 3. Apply polynomial features (degree 2) to key happiness factors ---
# These features capture non-linear relationships and interactions between core factors.
key_happiness_factors = [
    'Logged GDP per capita', 'Social support', 'Healthy life expectancy',
    'Freedom to make life choices', 'Generosity', 'Perceptions of corruption'
]

# Filter to ensure only existing and numeric columns are used
key_happiness_factors = [col for col in key_happiness_factors if col in df_processed.columns and pd.api.types.is_numeric_dtype(df_processed[col])]

if key_happiness_factors:
    poly = PolynomialFeatures(degree=2, include_bias=False)
    # Fit and transform the selected key happiness factors
    poly_features = poly.fit_transform(df_processed[key_happiness_factors])
    # Get the names for the new polynomial features
    poly_feature_names = poly.get_feature_names_out(key_happiness_factors)
    # Create a DataFrame for the new polynomial features
    df_poly = pd.DataFrame(poly_features, columns=poly_feature_names, index=df_processed.index)

    # Concatenate the new polynomial features back to the main DataFrame
    df_processed = pd.concat([df_processed, df_poly], axis=1)
else:
    print("Warning: No key happiness factors found for polynomial features. Skipping this step.")

# --- 8. Create interaction features between economic and social factors ---
# These features capture how economic and social factors might jointly influence happiness.
df_processed['GDP_x_SocialSupport'] = df_processed['Logged GDP per capita'] * df_processed['Social support']
df_processed['GDP_x_HealthyLifeExpectancy'] = df_processed['Logged GDP per capita'] * df_processed['Healthy life expectancy']

# --- Prepare data for scaling and dimensionality reduction ---
# Identify all numerical features, excluding identifiers and the target variable ('Ladder score').
numerical_features = df_processed.select_dtypes(include=np.number).columns.tolist()
if 'Ladder score' in numerical_features:
    numerical_features.remove('Ladder score') # 'Ladder score' is our target variable (y)

# Separate features (X) and target (y)
X = df_processed[numerical_features]
y = df_processed['Ladder score']

# Handle potential infinite values (e.g., from division by zero in engineered features)
# Replace inf/-inf with NaN, then fill NaNs with the mean of the respective column.
X.replace([np.inf, -np.inf], np.nan, inplace=True)
X.fillna(X.mean(), inplace=True) # Impute any NaNs that might have been created

# --- 4. Perform feature scaling using multiple methods ---
print("\n--- Feature Scaling ---")

# StandardScaler: Scales features to have zero mean and unit variance. Good for PCA.
scaler_standard = StandardScaler()
X_scaled_standard = scaler_standard.fit_transform(X)
df_scaled_standard = pd.DataFrame(X_scaled_standard, columns=X.columns, index=X.index)
print(f"StandardScaler applied. Shape: {df_scaled_standard.shape}")

# MinMaxScaler: Scales features to a fixed range, usually 0 to 1.
scaler_minmax = MinMaxScaler()
X_scaled_minmax = scaler_minmax.fit_transform(X)
df_scaled_minmax = pd.DataFrame(X_scaled_minmax, columns=X.columns, index=X.index)
print(f"MinMaxScaler applied. Shape: {df_scaled_minmax.shape}")

# RobustScaler: Scales features using statistics that are robust to outliers (median and IQR).
scaler_robust = RobustScaler()
X_scaled_robust = scaler_robust.fit_transform(X)
df_scaled_robust = pd.DataFrame(X_scaled_robust, columns=X.columns, index=X.index)
print(f"RobustScaler applied. Shape: {df_scaled_robust.shape}")

# For subsequent steps (PCA, t-SNE, Feature Selection), we will use the StandardScaler output.
X_scaled = df_scaled_standard.copy()

# --- 5. Apply PCA and t-SNE for dimensionality reduction and visualization ---
print("\n--- Dimensionality Reduction ---")

# PCA (Principal Component Analysis): Reduces dimensionality while preserving variance.
# We reduce to 2 components for easy visualization.
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df_pca = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'], index=X_scaled.index)
print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
print(f"Total explained variance by 2 components: {pca.explained_variance_ratio_.sum():.2f}")

# Visualization of PCA results
plt.figure(figsize=(10, 7))
sns.scatterplot(x='PC1', y='PC2', data=df_pca, hue=y, palette='viridis', s=50, alpha=0.8)
plt.title('PCA of Happiness Report Features (2 Components)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Ladder Score')
plt.grid(True)
plt.show()

# t-SNE (t-Distributed Stochastic Neighbor Embedding): Non-linear dimensionality reduction
# for visualizing high-dimensional data, preserving local structures.
# For efficiency, t-SNE is often applied on PCA-reduced data, especially for larger datasets.
# We'll use a reasonable number of PCA components as input to t-SNE.
n_components_tsne_input = min(50, X_scaled.shape[1]) # Use up to 50 components or max available
if n_components_tsne_input > 2:
    pca_for_tsne = PCA(n_components=n_components_tsne_input, random_state=42)
    X_for_tsne = pca_for_tsne.fit_transform(X_scaled)
else:
    X_for_tsne = X_scaled.values # If few features, use scaled data directly

# Adjust perplexity based on dataset size (should be less than n_samples - 1)
perplexity_val = min(30, len(X_for_tsne) - 1)
tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity_val, n_iter=1000)
X_tsne = tsne.fit_transform(X_for_tsne)
df_tsne = pd.DataFrame(data=X_tsne, columns=['TSNE1', 'TSNE2'], index=X_scaled.index)
print("t-SNE applied.")

# Visualization of t-SNE results
plt.figure(figsize=(10, 7))
sns.scatterplot(x='TSNE1', y='TSNE2', data=df_tsne, hue=y, palette='viridis', s=50, alpha=0.8)
plt.title('t-SNE of Happiness Report Features (2 Components)')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.colorbar(label='Ladder Score')
plt.grid(True)
plt.show()

# --- 6. Use SelectKBest and Recursive Feature Elimination for feature selection ---
print("\n--- Feature Selection ---")
k_best = 10 # Number of top features to select

# SelectKBest using f_regression: Scores features based on F-value for regression tasks.
selector_f_reg = SelectKBest(score_func=f_regression, k=k_best)
selector_f_reg.fit(X_scaled, y)
selected_features_f_reg = X_scaled.columns[selector_f_reg.get_support()].tolist()
print(f"SelectKBest (f_regression) - Top {k_best} features: {selected_features_f_reg}")

# SelectKBest using mutual_info_regression: Scores features based on mutual information,
# which can capture non-linear relationships.
selector_mi_reg = SelectKBest(score_func=mutual_info_regression, k=k_best)
selector_mi_reg.fit(X_scaled, y)
selected_features_mi_reg = X_scaled.columns[selector_mi_reg.get_support()].tolist()
print(f"SelectKBest (mutual_info_regression) - Top {k_best} features: {selected_features_mi_reg}")

# Recursive Feature Elimination (RFE): Selects features by recursively considering smaller
# and smaller sets of features. It trains a model and prunes features based on importance.
estimator = LinearRegression() # Using Linear Regression as the base estimator for RFE
rfe_selector = RFE(estimator=estimator, n_features_to_select=k_best, step=1)
rfe_selector.fit(X_scaled, y)
selected_features_rfe = X_scaled.columns[rfe_selector.get_support()].tolist()
print(f"RFE (LinearRegression) - Top {k_best} features: {selected_features_rfe}")

# --- 7. Implement correlation-based feature filtering and mutual information scoring ---

# Correlation-based feature filtering: Removes highly correlated features to reduce redundancy.
print("\n--- Correlation-based Feature Filtering ---")
correlation_matrix = X_scaled.corr().abs() # Calculate absolute correlation matrix
# Select upper triangle of correlation matrix to avoid duplicates and self-correlation
upper_tri = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
# Find features with correlation greater than a threshold (e.g., 0.9)
to_drop_highly_correlated = [column for column in upper_tri.columns if any(upper_tri[column] > 0.9)]
print(f"Features to drop due to high correlation (>0.9): {to_drop_highly_correlated}")

# Create a DataFrame without highly correlated features for potential further use
X_filtered_corr = X_scaled.drop(columns=to_drop_highly_correlated, errors='ignore')
print(f"Shape after correlation filtering: {X_filtered_corr.shape}")

# Mutual Information Scoring: (Already performed with SelectKBest, but displaying scores explicitly)
print("\n--- Mutual Information Scoring (Full List) ---")
mi_scores = mutual_info_regression(X_scaled, y, random_state=42)
mi_scores_df = pd.DataFrame({'Feature': X_scaled.columns, 'MI_Score': mi_scores})
mi_scores_df = mi_scores_df.sort_values(by='MI_Score', ascending=False)
print("Top 10 features by Mutual Information Score:")
print(mi_scores_df.head(10).to_string()) # Use to_string() for better console formatting

# --- 9. Validate feature importance using Random Forest and visualize results ---
print("\n--- Random Forest Feature Importance ---")

# Initialize and train a RandomForestRegressor model.
# n_estimators: number of trees in the forest.
# random_state: for reproducibility.
# n_jobs=-1: use all available CPU cores.
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_scaled, y) # Train on the full scaled dataset

# Get feature importances from the trained Random Forest model
feature_importances = rf_model.feature_importances_
# Create a DataFrame to store features and their importances
features_df = pd.DataFrame({'Feature': X_scaled.columns, 'Importance': feature_importances})
# Sort features by importance in descending order
features_df = features_df.sort_values(by='Importance', ascending=False)

print("\nTop 20 Features by Random Forest Importance:")
print(features_df.head(20).to_string())

# Visualize the top 20 feature importances
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=features_df.head(20), palette='viridis')
plt.title('Top 20 Feature Importances from Random Forest')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout() # Adjust layout to prevent labels from overlapping
plt.show()

print("\n--- Advanced Feature Engineering and Dimensionality Reduction Process Completed ---")
print(f"Final DataFrame shape after engineering: {df_processed.shape}")
print(f"Number of features used for modeling (after scaling): {X_scaled.shape[1]}")
```