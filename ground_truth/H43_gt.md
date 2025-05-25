```python
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# --- 1. Load the dataset ---
# Assume 'penguins.csv' is in the current directory
df = pd.read_csv('penguins.csv')

# --- 2. Define features X and target y ---
# Features to be used for classification
features = ['culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g', 'island', 'sex']
# Target variable
target = 'species'

# Create copies to avoid SettingWithCopyWarning
X = df[features].copy()
y = df[target].copy()

# Separate numerical and categorical feature names for easier processing
numerical_features = ['culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g']
categorical_features = ['island', 'sex']

# --- Handle missing values in the target variable (species) ---
# Rows with missing target values cannot be used for supervised learning.
if y.isnull().any():
    # Get indices of rows where 'species' is NA
    na_target_indices = y[y.isnull()].index
    # Drop these rows from both features (X) and target (y)
    X.drop(na_target_indices, inplace=True)
    y.drop(na_target_indices, inplace=True)
    # Reset index to ensure alignment and continuous indexing after dropping
    X.reset_index(drop=True, inplace=True)
    y.reset_index(drop=True, inplace=True)

# --- 3. Impute missing values in numerical features using their respective medians ---
for col in numerical_features:
    if X[col].isnull().any():
        median_val = X[col].median()
        X[col].fillna(median_val, inplace=True)

# --- 4. Impute missing values in categorical features using their respective modes ---

# Impute 'sex' using its mode
if X['sex'].isnull().any():
    # mode() returns a Series, take the first element (most frequent)
    mode_sex = X['sex'].mode()[0]
    X['sex'].fillna(mode_sex, inplace=True)

# Impute 'island' using its mode, with special handling for all-NA case
if X['island'].isnull().any():
    # Check if ALL 'island' values are NA. If so, mode() will return an empty Series.
    if X['island'].isnull().all():
        # If all values are NA, mode imputation is not possible.
        # As per requirement: "if mode imputation results in NA for island, drop those rows".
        # This means dropping all rows if 'island' column was entirely NA.
        X.dropna(subset=['island'], inplace=True)
        y = y.loc[X.index] # Align y with X after dropping
        # Reset index after dropping to maintain consistency
        X.reset_index(drop=True, inplace=True)
        y.reset_index(drop=True, inplace=True)
    else:
        # If not all values are NA, impute with the calculated mode
        mode_island = X['island'].mode()[0]
        X['island'].fillna(mode_island, inplace=True)
        # After imputation, if any NAs still exist (e.g., if mode itself was NaN, which is unlikely for strings),
        # drop those rows as per the requirement. This is a safeguard.
        if X['island'].isnull().any():
            X.dropna(subset=['island'], inplace=True)
            y = y.loc[X.index] # Align y with X after dropping
            # Reset index after dropping to maintain consistency
            X.reset_index(drop=True, inplace=True)
            y.reset_index(drop=True, inplace=True)

# --- 5. One-hot encode the categorical features 'island' and 'sex' ---
# pd.get_dummies automatically handles creating new columns and dropping original ones
X = pd.get_dummies(X, columns=categorical_features, drop_first=False)

# --- 6. Scale all numerical features using StandardScaler ---
scaler = StandardScaler()
# Apply scaling only to the numerical columns
X[numerical_features] = scaler.fit_transform(X[numerical_features])

# --- 7. Label encode the target 'species' ---
# RandomForestClassifier requires numerical target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# --- 8. Train a RandomForestClassifier model and evaluate using 5-fold cross-validated accuracy score ---
# Initialize the RandomForestClassifier with specified parameters
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Perform 5-fold cross-validation to evaluate the model's performance
# 'scoring='accuracy'' specifies that accuracy score should be used
cv_scores = cross_val_score(model, X, y_encoded, cv=5, scoring='accuracy')

# Print the mean accuracy across all folds
print(f"Mean 5-fold cross-validated accuracy: {cv_scores.mean():.4f}")
```