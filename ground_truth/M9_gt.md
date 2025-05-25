```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split # Not directly used for selection, but good practice for model training

# Set plot style for better aesthetics
sns.set_style("whitegrid")

# --- 1. Load Data and Handle Missing Values ---
# Load the dataset from the current directory
try:
    df = pd.read_csv('water_potability.csv')
except FileNotFoundError:
    print("Error: water_potability.csv not found. Please ensure the file is in the current directory.")
    exit()

# Handle missing values: Impute with the mean for numerical columns
# The 'ph', 'Sulfate', and 'Trihalomethanes' columns are known to have missing values.
for col in ['ph', 'Sulfate', 'Trihalomethanes']:
    if df[col].isnull().any():
        df[col] = df[col].fillna(df[col].mean())

# Separate features (X) and target (y)
X = df.drop('Potability', axis=1)
y = df['Potability']

# --- 2. Feature Engineering ---
# Create a copy of the original features to add engineered features
X_fe = X.copy()

# Define a small epsilon to prevent division by zero in ratio calculations
epsilon = 1e-6

# 2.1. Create new meaningful features: Ratios between chemical parameters
X_fe['pH_Chloramines_Ratio'] = X_fe['ph'] / (X_fe['Chloramines'] + epsilon)
X_fe['Hardness_Solids_Ratio'] = X_fe['Hardness'] / (X_fe['Solids'] + epsilon)
X_fe['Sulfate_Chloramines_Ratio'] = X_fe['Sulfate'] / (X_fe['Chloramines'] + epsilon)
X_fe['Organic_carbon_Trihalomethanes_Ratio'] = X_fe['Organic_carbon'] / (X_fe['Trihalomethanes'] + epsilon)
X_fe['Sulfate_pH_Ratio'] = X_fe['Sulfate'] / (X_fe['ph'] + epsilon)
X_fe['Chloramines_Organic_carbon_Ratio'] = X_fe['Chloramines'] / (X_fe['Organic_carbon'] + epsilon)
X_fe['Solids_Turbidity_Ratio'] = X_fe['Solids'] / (X_fe['Turbidity'] + epsilon)

# 2.2. Create Interaction Terms
X_fe['pH_Chloramines_Interaction'] = X_fe['ph'] * X_fe['Chloramines']
X_fe['Hardness_Solids_Interaction'] = X_fe['Hardness'] * X_fe['Solids']
X_fe['Sulfate_Organic_carbon_Interaction'] = X_fe['Sulfate'] * X_fe['Organic_carbon']
X_fe['Trihalomethanes_Turbidity_Interaction'] = X_fe['Trihalomethanes'] * X_fe['Turbidity']
X_fe['Chloramines_Turbidity_Interaction'] = X_fe['Chloramines'] * X_fe['Turbidity']

# 2.3. Create Polynomial Features
# Select a subset of original features for polynomial expansion to avoid excessive dimensionality
poly_features_cols = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Organic_carbon', 'Trihalomethanes', 'Turbidity']
poly = PolynomialFeatures(degree=2, include_bias=False) # degree=2 for quadratic terms, no bias (intercept)

# Fit and transform the selected columns to create polynomial features
poly_temp_df = pd.DataFrame(poly.fit_transform(X_fe[poly_features_cols]),
                            columns=poly.get_feature_names_out(poly_features_cols),
                            index=X_fe.index)

# Add polynomial features to the engineered DataFrame, avoiding duplication of original columns
for col in poly_temp_df.columns:
    if col not in X_fe.columns: # Only add if it's a new feature (e.g., x1*x2, x1^2)
        X_fe[col] = poly_temp_df[col]

# Ensure all column names are strings for consistency, especially after polynomial features
X_fe.columns = X_fe.columns.astype(str)

# --- 3. Calculate Correlation Matrix ---
# Combine engineered features with the target variable for correlation calculation
df_corr = pd.concat([X_fe, y], axis=1)
correlation_matrix = df_corr.corr()

# 3.1. Identify the top 5 features most correlated with water potability
# Calculate absolute correlations with 'Potability' and sort them
potability_correlations = correlation_matrix['Potability'].abs().sort_values(ascending=False)
# Exclude 'Potability' itself from the top features list (it will have a correlation of 1 with itself)
top_5_correlated_features = potability_correlations[1:6].index.tolist() # [1:6] gets the top 5 after 'Potability'

print(f"Top 5 features most correlated with Potability:\n{top_5_correlated_features}\n")

# --- 4. Apply Feature Selection Techniques ---

# Standardize features before applying selection techniques.
# This is crucial for models sensitive to scale and generally good practice.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_fe)
X_scaled_df = pd.DataFrame(X_scaled, columns=X_fe.columns, index=X_fe.index)

# 4.1. SelectKBest (using f_classif for classification tasks)
k_best_features_count = 15 # Number of features to select
selector_kbest = SelectKBest(f_classif, k=k_best_features_count)
selector_kbest.fit(X_scaled_df, y)
selected_features_kbest = X_scaled_df.columns[selector_kbest.get_support()].tolist()

print(f"Features selected by SelectKBest (k={k_best_features_count}):\n{selected_features_kbest}\n")

# 4.2. RFE (Recursive Feature Elimination) with Random Forest Classifier
rfe_features_count = 15 # Number of features to select
# Initialize a RandomForestClassifier as the estimator for RFE
estimator = RandomForestClassifier(n_estimators=100, random_state=42)
selector_rfe = RFE(estimator=estimator, n_features_to_select=rfe_features_count, step=1)
selector_rfe.fit(X_scaled_df, y)
selected_features_rfe = X_scaled_df.columns[selector_rfe.support_].tolist()

print(f"Features selected by RFE (RandomForestClassifier, n={rfe_features_count}):\n{selected_features_rfe}\n")

# --- 5. Create Advanced Visualizations ---

# 5.1. Correlation Heatmap for all features (original + engineered)
plt.figure(figsize=(20, 18))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix of All Features (Original + Engineered)', fontsize=18)
plt.xticks(rotation=90, fontsize=8)
plt.yticks(rotation=0, fontsize=8)
plt.tight_layout()
plt.show()

# 5.2. Feature Importance Plots (from a RandomForestClassifier trained on all features)
# Train a RandomForestClassifier on the full scaled dataset to get comprehensive feature importances
rf_full = RandomForestClassifier(n_estimators=100, random_state=42)
rf_full.fit(X_scaled_df, y)
feature_importances = pd.Series(rf_full.feature_importances_, index=X_scaled_df.columns)
feature_importances_sorted = feature_importances.sort_values(ascending=False)

plt.figure(figsize=(14, 10))
sns.barplot(x=feature_importances_sorted.head(25).values, y=feature_importances_sorted.head(25).index, palette='viridis')
plt.title('Top 25 Feature Importances (from RandomForestClassifier)', fontsize=16)
plt.xlabel('Importance', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.tight_layout()
plt.show()

# 5.3. Pairwise Scatter Plots for Top Correlated Features
# Include 'Potability' in the list for hue
features_for_pairplot = top_5_correlated_features + ['Potability']
# Create a DataFrame containing only these features for the pairplot
# Ensure all selected features actually exist in the combined dataframe
valid_features_for_pairplot = [col for col in features_for_pairplot if col in df_corr.columns]

if len(valid_features_for_pairplot) > 1: # Ensure there are features to plot
    g = sns.pairplot(df_corr[valid_features_for_pairplot], hue='Potability', palette='viridis', diag_kind='kde', markers=["o", "s"])
    g.fig.suptitle('Pairwise Scatter Plots of Top 5 Correlated Features by Potability', y=1.02, fontsize=16) # Adjust title position
    plt.show()
else:
    print("Not enough valid features to create pairwise scatter plots for top correlated features.")

# --- 6. Compare Feature Sets Selected by Different Methods ---
# Convert lists to sets for easy comparison (intersection, difference)
set_kbest = set(selected_features_kbest)
set_rfe = set(selected_features_rfe)
set_top_corr = set(top_5_correlated_features)

# Overlap between SelectKBest and RFE selected features
overlap_kbest_rfe = set_kbest.intersection(set_rfe)
print(f"\nOverlap between SelectKBest and RFE selected features ({len(overlap_kbest_rfe)} features):\n{list(overlap_kbest_rfe)}\n")

# Features unique to SelectKBest
unique_to_kbest = set_kbest.difference(set_rfe)
print(f"Features unique to SelectKBest ({len(unique_to_kbest)} features):\n{list(unique_to_kbest)}\n")

# Features unique to RFE
unique_to_rfe = set_rfe.difference(set_kbest)
print(f"Features unique to RFE ({len(unique_to_rfe)} features):\n{list(unique_to_rfe)}\n")

# Overlap with top 5 correlated features
overlap_kbest_top_corr = set_kbest.intersection(set_top_corr)
print(f"Overlap between SelectKBest and Top 5 Correlated features ({len(overlap_kbest_top_corr)} features):\n{list(overlap_kbest_top_corr)}\n")

overlap_rfe_top_corr = set_rfe.intersection(set_top_corr)
print(f"Overlap between RFE and Top 5 Correlated features ({len(overlap_rfe_top_corr)} features):\n{list(overlap_rfe_top_corr)}\n")

# All unique features selected by either method
all_selected_features_union = list(set_kbest.union(set_rfe))
print(f"All unique features selected by either SelectKBest or RFE ({len(all_selected_features_union)} features):\n{all_selected_features_union}\n")

print("Feature engineering, correlation analysis, and feature selection complete.")
```