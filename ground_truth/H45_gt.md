```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder

# Set a consistent style for plots
sns.set_theme(style="whitegrid")

# --- 1. Load the dataset ---
try:
    df = pd.read_csv('penguins.csv')
except FileNotFoundError:
    print("Error: penguins.csv not found. Please ensure the file is in the current directory.")
    exit()

# Create a copy to avoid SettingWithCopyWarning later
df_processed = df.copy()

# --- 2. Impute missing values ---

# Drop rows where 'island' is NA first, as specified
df_processed.dropna(subset=['island'], inplace=True)

# Impute numerical columns with their respective means
numerical_cols_to_impute = ['culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g']
for col in numerical_cols_to_impute:
    if df_processed[col].isnull().any():
        df_processed[col].fillna(df_processed[col].mean(), inplace=True)

# Impute 'sex' with its mode
if df_processed['sex'].isnull().any():
    # mode() can return multiple values if there's a tie, so take the first one
    mode_sex = df_processed['sex'].mode()[0]
    df_processed['sex'].fillna(mode_sex, inplace=True)

# --- 3. Define features X and target y ---

# Numerical features (already imputed)
numerical_features = df_processed[numerical_cols_to_impute]

# One-hot encode 'island' and 'sex'
# Ensure 'drop_first=True' is not used if all categories are needed for feature importance
# (though for models, it's often used to avoid multicollinearity)
# For feature importance, we want to see the importance of each dummy variable.
island_encoded = pd.get_dummies(df_processed['island'], prefix='island', dtype=int)
sex_encoded = pd.get_dummies(df_processed['sex'], prefix='sex', dtype=int)

# Concatenate all features to form X
X = pd.concat([numerical_features, island_encoded, sex_encoded], axis=1)

# Label encode 'species' for target y
le = LabelEncoder()
y = le.fit_transform(df_processed['species'])

# Get the original species names for plotting later
species_names = le.classes_

# --- 4. Train a GradientBoostingClassifier ---
gb_classifier = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb_classifier.fit(X, y)

# --- 5. Extract and print feature importances ---
feature_importances = gb_classifier.feature_importances_
feature_names = X.columns

# Create a DataFrame for better readability
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importances
})

# Sort by importance in descending order
importance_df = importance_df.sort_values(by='Importance', ascending=False)

print("--- Feature Importances from GradientBoostingClassifier ---")
print(importance_df)
print("\n")

# --- 6. For each species, create box plots to visualize the distribution of flipper_length_mm ---
# Ensure the 'species' column in df_processed is the original categorical one for plotting
# (not the label encoded 'y')
# And 'sex' is the imputed one.

print("--- Visualizing Flipper Length Distribution by Species and Sex ---")
plt.figure(figsize=(12, 7))
sns.boxplot(data=df_processed, x='species', y='flipper_length_mm', hue='sex', palette='viridis')
plt.title('Distribution of Flipper Length by Species and Sex')
plt.xlabel('Species')
plt.ylabel('Flipper Length (mm)')
plt.legend(title='Sex')
plt.tight_layout()
plt.show()

# Optional: Create separate plots for each species for more detailed view if needed
# for species in species_names:
#     plt.figure(figsize=(8, 6))
#     subset_df = df_processed[df_processed['species'] == species]
#     sns.boxplot(data=subset_df, x='sex', y='flipper_length_mm', palette='viridis')
#     plt.title(f'Flipper Length Distribution for {species} by Sex')
#     plt.xlabel('Sex')
#     plt.ylabel('Flipper Length (mm)')
#     plt.tight_layout()
#     plt.show()
```