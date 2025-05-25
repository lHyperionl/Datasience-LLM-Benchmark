```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from collections import Counter

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

# --- 1. Load and Preprocess the Dataset ---
print("--- 1. Loading and Preprocessing Data ---")
try:
    df = pd.read_csv('diabetes.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: diabetes.csv not found. Please ensure the file is in the current directory.")
    exit()

# Identify columns where 0s represent missing values
# Pregnancies and Outcome can legitimately be 0.
# Glucose, BloodPressure, SkinThickness, Insulin, BMI cannot be 0.
cols_with_zero_as_missing = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

# Replace 0s with NaN in specified columns
for col in cols_with_zero_as_missing:
    df[col] = df[col].replace(0, np.nan)

# Impute missing values using the median strategy
imputer = SimpleImputer(strategy='median')
df[cols_with_zero_as_missing] = imputer.fit_transform(df[cols_with_zero_as_missing])
print("Missing values (0s) replaced with NaNs and then imputed with column medians.")

# --- 2. Create Engineered Features ---
print("\n--- 2. Creating Engineered Features ---")

# 'Metabolic_Score': Combination of Glucose, BMI, and Insulin
# Using a sum as a simple combination, as these are all related to metabolic health.
df['Metabolic_Score'] = df['Glucose'] + df['BMI'] + df['Insulin']
print("Created 'Metabolic_Score'.")

# 'Risk_Factor_Count': Count of high-risk factors
# Define thresholds for high-risk based on common medical guidelines
df['Risk_Factor_Count'] = (
    (df['Glucose'] > 140).astype(int) +  # Prediabetes/Diabetes threshold
    (df['BloodPressure'] > 90).astype(int) + # Hypertension threshold
    (df['BMI'] > 30).astype(int) +       # Obesity threshold
    (df['Age'] > 50).astype(int)         # Older age as a risk factor
)
print("Created 'Risk_Factor_Count'.")

# 'Glucose_per_Age'
df['Glucose_per_Age'] = df['Glucose'] / df['Age']
print("Created 'Glucose_per_Age'.")

# 'BMI_BloodPressure_interaction'
df['BMI_BloodPressure_interaction'] = df['BMI'] * df['BloodPressure']
print("Created 'BMI_BloodPressure_interaction'.")

# 'Pedigree_Age_product'
df['Pedigree_Age_product'] = df['DiabetesPedigreeFunction'] * df['Age']
print("Created 'Pedigree_Age_product'.")

print(f"Current number of features after engineering: {df.shape[1] - 1}") # -1 for target column

# --- 3. Apply Polynomial Features (degree 2) ---
print("\n--- 3. Applying Polynomial Features (degree 2) ---")
# Select numerical columns for polynomial features, excluding the target and newly engineered features
# that might already be interactions or sums.
# Let's apply to core numerical features.
poly_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'Age']
poly = PolynomialFeatures(degree=2, include_bias=False)

# Create a DataFrame for polynomial features
poly_features = poly.fit_transform(df[poly_cols])
poly_feature_names = poly.get_feature_names_out(poly_cols)
df_poly = pd.DataFrame(poly_features, columns=poly_feature_names, index=df.index)

# Drop original columns that are now represented by polynomial features to avoid redundancy
# Keep only the new polynomial features that are not already in the original dataframe
# This avoids duplicating original features if include_bias=False is used.
# We will merge the new polynomial features with the original dataframe, excluding the original poly_cols.
df = df.drop(columns=poly_cols)
df = pd.concat([df, df_poly], axis=1)
print(f"Applied polynomial features to {len(poly_cols)} columns. New total features: {df.shape[1] - 1}")

# Separate features (X) and target (y)
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# --- 4. Perform Feature Scaling using StandardScaler and MinMaxScaler ---
print("\n--- 4. Performing Feature Scaling ---")

# Identify numerical columns for scaling (all features except the target)
numerical_cols = X.columns

# StandardScaler
scaler_standard = StandardScaler()
X_scaled_standard = scaler_standard.fit_transform(X)
X_scaled_standard_df = pd.DataFrame(X_scaled_standard, columns=numerical_cols, index=X.index)
print("Features scaled using StandardScaler.")

# MinMaxScaler
scaler_minmax = MinMaxScaler()
X_scaled_minmax = scaler_minmax.fit_transform(X)
X_scaled_minmax_df = pd.DataFrame(X_scaled_minmax, columns=numerical_cols, index=X.index)
print("Features scaled using MinMaxScaler.")

# For subsequent steps, we'll primarily use the StandardScaler output as it's generally preferred for PCA and distance-based methods.
X_processed = X_scaled_standard_df

# --- 7. Implement correlation-based feature filtering ---
print("\n--- 7. Implementing Correlation-based Feature Filtering ---")
# Calculate the correlation matrix
corr_matrix = X_processed.corr().abs()

# Select upper triangle of correlation matrix
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

# Find features with correlation greater than 0.9
to_drop_highly_correlated = [column for column in upper_tri.columns if any(upper_tri[column] > 0.9)]

# Drop features
X_filtered_corr = X_processed.drop(columns=to_drop_highly_correlated)
print(f"Dropped {len(to_drop_highly_correlated)} highly correlated features (threshold > 0.9).")
print(f"Features remaining after correlation filtering: {X_filtered_corr.shape[1]}")

# --- 5. Apply Principal Component Analysis (PCA) and explain variance ratios ---
print("\n--- 5. Applying Principal Component Analysis (PCA) ---")
# Apply PCA on the correlation-filtered data
pca = PCA(n_components=0.95) # Retain components explaining 95% of variance
X_pca = pca.fit_transform(X_filtered_corr)

print(f"Original number of features: {X_filtered_corr.shape[1]}")
print(f"Number of components selected by PCA (explaining 95% variance): {pca.n_components_}")
print("Explained variance ratio per component:")
print(pca.explained_variance_ratio_)
print(f"Cumulative explained variance: {np.sum(pca.explained_variance_ratio_):.2f}")

# --- 6. Use SelectKBest and Recursive Feature Elimination for feature selection ---
print("\n--- 6. Applying SelectKBest and Recursive Feature Elimination ---")

# SelectKBest
# Using f_classif for classification tasks
k_best_selector = SelectKBest(score_func=f_classif, k=15) # Select top 15 features
X_kbest = k_best_selector.fit_transform(X_filtered_corr, y)
selected_features_kbest = X_filtered_corr.columns[k_best_selector.get_support()]
print(f"\nSelectKBest selected {len(selected_features_kbest)} features:")
print(selected_features_kbest.tolist())

# Recursive Feature Elimination (RFE)
# Using Logistic Regression as the estimator
rfe_estimator = LogisticRegression(solver='liblinear', random_state=42)
rfe_selector = RFE(estimator=rfe_estimator, n_features_to_select=10, step=1) # Select top 10 features
X_rfe = rfe_selector.fit_transform(X_filtered_corr, y)
selected_features_rfe = X_filtered_corr.columns[rfe_selector.get_support()]
print(f"\nRFE selected {len(selected_features_rfe)} features:")
print(selected_features_rfe.tolist())

# --- 8. Apply SMOTE for handling class imbalance ---
print("\n--- 8. Applying SMOTE for handling class imbalance ---")

# Use the correlation-filtered data for SMOTE and subsequent steps
# This ensures we are working with a refined feature set.
X_final = X_filtered_corr

# Split data into training and testing sets BEFORE SMOTE
X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42, stratify=y)

print(f"Original training set class distribution: {Counter(y_train)}")

smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print(f"Training set class distribution after SMOTE: {Counter(y_train_smote)}")

# --- 9. Evaluate feature importance using Random Forest and visualize top features ---
print("\n--- 9. Evaluating Feature Importance using Random Forest ---")

# Train a RandomForestClassifier on the SMOTE-transformed training data
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_smote, y_train_smote)

# Get feature importances
feature_importances = rf_model.feature_importances_
features = X_train_smote.columns

# Create a DataFrame for feature importances
importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

print("\nTop 15 Features by Importance:")
print(importance_df.head(15))

# Visualize top features
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=importance_df.head(15))
plt.title('Top 15 Feature Importances from Random Forest')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()

print("\n--- Advanced Feature Engineering and Dimensionality Reduction Complete ---")
```