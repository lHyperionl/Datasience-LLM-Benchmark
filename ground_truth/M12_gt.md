```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

# Suppress warnings for cleaner output, especially from sklearn/pandas versions
import warnings
warnings.filterwarnings('ignore')

# --- 1. Load the dataset and handle missing values ---
# Load the dataset from the specified CSV file
try:
    df = pd.read_csv('video_games_sales.csv')
except FileNotFoundError:
    print("Error: 'video_games_sales.csv' not found. Please ensure the file is in the current directory.")
    exit()

print("--- Initial Data Loading and Missing Value Handling ---")
print("Original DataFrame Info:")
df.info()
print("\nMissing values before handling:")
print(df.isnull().sum())

# Handle missing values for 'Year' using median imputation
# 'Year' is a numerical column, median is robust to outliers
imputer_year = SimpleImputer(strategy='median')
df['Year'] = imputer_year.fit_transform(df[['Year']])
# Convert 'Year' to integer type after imputation
df['Year'] = df['Year'].astype(int)

# Handle missing values for 'Publisher' and 'Genre' using mode imputation
# These are categorical columns, mode is appropriate
imputer_categorical = SimpleImputer(strategy='most_frequent')
df['Publisher'] = imputer_categorical.fit_transform(df[['Publisher']]).ravel()
df['Genre'] = imputer_categorical.fit_transform(df[['Genre']]).ravel()

# Handle missing values for 'Name' (if any), though not critical for numerical analysis
df['Name'].fillna('Unknown Game', inplace=True)

print("\nMissing values after handling:")
print(df.isnull().sum())
print("\nDataFrame Info after handling missing values:")
df.info()
print("\nDataFrame head after missing value handling:")
print(df.head())

# --- 2. Create engineered features ---
print("\n--- Feature Engineering ---")

# Feature 1: 'Sales_Ratio_NA_EU' (NA_Sales/EU_Sales)
# Add a small epsilon (1e-6) to the denominator to prevent division by zero errors
df['Sales_Ratio_NA_EU'] = df['NA_Sales'] / (df['EU_Sales'] + 1e-6)

# Feature 2: 'Publisher_Avg_Sales' (average global sales per publisher)
# Use .transform() to broadcast the aggregated mean back to the original DataFrame's size
df['Publisher_Avg_Sales'] = df.groupby('Publisher')['Global_Sales'].transform('mean')

# Feature 3: 'Genre_Market_Share' (percentage of total sales by genre)
# Calculate total global sales across all games
total_global_sales = df['Global_Sales'].sum()
# Calculate sum of global sales per genre and then divide by total global sales
df['Genre_Market_Share'] = df.groupby('Genre')['Global_Sales'].transform('sum') / total_global_sales

# Feature 4: 'Platform_Popularity' (number of games per platform)
# Count the number of games associated with each platform
df['Platform_Popularity'] = df.groupby('Platform')['Name'].transform('count')

# Feature 5: 'Sales_Momentum' (difference between Global_Sales and median sales for that year)
# Calculate the median global sales for each year
median_sales_per_year = df.groupby('Year')['Global_Sales'].transform('median')
# Calculate the difference for each game
df['Sales_Momentum'] = df['Global_Sales'] - median_sales_per_year

print("\nDataFrame head with new engineered features:")
print(df[['Name', 'Global_Sales', 'Year', 'Publisher', 'Genre', 'Platform',
          'Sales_Ratio_NA_EU', 'Publisher_Avg_Sales', 'Genre_Market_Share',
          'Platform_Popularity', 'Sales_Momentum']].head())

# --- 3. Perform correlation analysis between all numerical features and visualize with a heatmap ---
print("\n--- Correlation Analysis ---")
# Select only numerical columns for correlation analysis
# Exclude 'Rank' as it's an identifier, not a true numerical feature for correlation
numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
if 'Rank' in numerical_cols:
    numerical_cols.remove('Rank')

# Calculate the correlation matrix
correlation_matrix = df[numerical_cols].corr()

# Plot the heatmap
plt.figure(figsize=(16, 12))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, cbar_kws={'label': 'Correlation Coefficient'})
plt.title('Correlation Matrix of Numerical Features', fontsize=16)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# --- 4. Use PCA to reduce dimensionality and explain variance ---
print("\n--- Principal Component Analysis (PCA) ---")
# Select numerical features for PCA. It's crucial to scale data before PCA.
# We'll use all numerical columns except 'Rank' for PCA.
pca_features = [col for col in numerical_cols if col not in ['Rank']]

# Drop any rows that might have NaN values in the selected PCA features (should be minimal after imputation)
df_pca_data = df[pca_features].dropna()

# Scale the data using StandardScaler
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_pca_data)

# Apply PCA
pca = PCA()
pca.fit(scaled_data)

# Explained variance ratio by each principal component
explained_variance_ratio = pca.explained_variance_ratio_
print("\nExplained Variance Ratio by Principal Components:")
# Display components explaining at least 1% of variance
for i, ratio in enumerate(explained_variance_ratio):
    if ratio > 0.01: # Display components that explain more than 1% variance
        print(f"  PC_{i+1}: {ratio:.4f}")

# Plot cumulative explained variance
plt.figure(figsize=(10, 6))
plt.plot(np.cumsum(explained_variance_ratio), marker='o', linestyle='--')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Cumulative Explained Variance by PCA Components', fontsize=14)
plt.grid(True)
plt.show()

# Transform data to principal components
principal_components = pca.transform(scaled_data)
# Create a DataFrame for the principal components
df_pca_components = pd.DataFrame(data=principal_components,
                                 columns=[f'PC_{i+1}' for i in range(principal_components.shape[1])])

print("\nFirst 5 Principal Components (transformed data):")
print(df_pca_components.head())

# --- 5. Apply polynomial features (degree 2) to sales columns and analyze their impact ---
print("\n--- Polynomial Feature Engineering (Degree 2) ---")
sales_cols = ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales']

# Ensure sales columns are numeric and handle any potential non-numeric values
for col in sales_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
# Drop rows where sales data might have become NaN during conversion (should be rare)
df.dropna(subset=sales_cols, inplace=True)

# Initialize PolynomialFeatures with degree 2 and no bias term
poly = PolynomialFeatures(degree=2, include_bias=False)
# Fit and transform the sales columns to create polynomial features
poly_features = poly.fit_transform(df[sales_cols])

# Get the names of the new polynomial features
poly_feature_names = poly.get_feature_names_out(sales_cols)
# Create a DataFrame for the polynomial features, preserving the original index
df_poly_features = pd.DataFrame(poly_features, columns=poly_feature_names, index=df.index)

# Add the new polynomial features to the main DataFrame
df = pd.concat([df, df_poly_features], axis=1)

print("\nDataFrame head with new polynomial features (first 5 columns):")
# Display a selection of the new polynomial features
print(df[poly_feature_names].head())

# --- 6. Create interaction features between Year and Genre ---
print("\n--- Interaction Feature Engineering (Year x Genre) ---")
# Convert 'Genre' to one-hot encoded features
# drop_first=True avoids multicollinearity if used in a linear model
genre_dummies = pd.get_dummies(df['Genre'], prefix='Genre', drop_first=True)

# Ensure 'Year' is numeric (it should be after initial imputation)
df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
# Drop rows where Year might have become NaN (should be rare)
df.dropna(subset=['Year'], inplace=True)

# Create interaction features by multiplying 'Year' with each genre dummy column
for col in genre_dummies.columns:
    df[f'Year_x_{col}'] = df['Year'] * genre_dummies[col]

# Add the genre dummy variables to the main DataFrame (they are used for interaction)
df = pd.concat([df, genre_dummies], axis=1)

print("\nDataFrame head with new interaction features (Year x Genre):")
# Display original Year, Genre, and a few example interaction columns
interaction_cols_display = [col for col in df.columns if 'Year_x_Genre_' in col]
print(df[['Year', 'Genre'] + interaction_cols_display[:5]].head()) # Display first 5 interaction columns

# --- 7. Identify and handle outliers using IQR method ---
print("\n--- Outlier Handling using IQR Method ---")
# Define columns to apply outlier handling to. These are typically numerical features.
outlier_cols = ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales',
                'Sales_Ratio_NA_EU', 'Publisher_Avg_Sales', 'Genre_Market_Share', 'Sales_Momentum']

# Ensure all outlier_cols are numeric before processing
for col in outlier_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop rows where these critical columns might have become NaN during type conversion
df.dropna(subset=outlier_cols, inplace=True)

print("\nOutlier handling statistics (before and after capping):")
for col in outlier_cols:
    # Calculate Q1 (25th percentile) and Q3 (75th percentile)
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    # Calculate the Interquartile Range (IQR)
    IQR = Q3 - Q1
    
    # Define lower and upper bounds for outlier detection
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Count outliers before capping
    outliers_count = df[(df[col] < lower_bound) | (df[col] > upper_bound)].shape[0]
    print(f"Column '{col}': {outliers_count} outliers detected.")

    # Cap outliers: values below lower_bound are set to lower_bound,
    # values above upper_bound are set to upper_bound.
    df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
    df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])

print("\nOutlier handling complete. Outliers in specified columns have been capped.")

# --- Final DataFrame Overview ---
print("\n--- Final DataFrame Overview ---")
print("Final DataFrame Info after all operations:")
df.info()
print("\nFinal DataFrame head:")
print(df.head())
```