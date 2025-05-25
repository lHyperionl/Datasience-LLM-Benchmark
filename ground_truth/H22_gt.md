```python
# Import necessary libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# --- 1. Load the dataset ---
# Assume the CSV file is in the current directory.
try:
    df = pd.read_csv('clash_royale_cards.csv')
    print("Dataset loaded successfully.")
    print(f"Dataset shape: {df.shape}")
    print("First 5 rows of the dataset:")
    print(df.head())
except FileNotFoundError:
    print("Error: 'clash_royale_cards.csv' not found. Please ensure the file is in the current directory.")
    # Exit the script if the file is not found, as subsequent steps depend on it.
    exit()

# --- 2. Select the features for clustering ---
features = ['elixirCost', 'Win Rate', 'Usage']

# Check if all required features exist in the DataFrame
missing_features = [f for f in features if f not in df.columns]
if missing_features:
    print(f"Error: Missing required features in the dataset: {missing_features}")
    print("Available columns:", df.columns.tolist())
    # Exit the script if required columns are missing.
    exit()

# Create a copy to avoid SettingWithCopyWarning when modifying the original DataFrame later
X = df[features].copy()

print(f"\nSelected features for clustering: {features}")
print("First 5 rows of selected features:")
print(X.head())

# --- 3. Standardize these features using StandardScaler ---
# StandardScaler transforms data to have a mean of 0 and a standard deviation of 1.
# This is crucial for distance-based algorithms like K-Means.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert scaled features back to a DataFrame for better readability (optional, but good for inspection)
X_scaled_df = pd.DataFrame(X_scaled, columns=features)
print("\nFirst 5 rows of standardized features:")
print(X_scaled_df.head())

# --- 4. Determine an appropriate number of clusters (K) using the Elbow Method ---
# The Elbow Method plots the Within-Cluster Sum of Squares (WCSS) against the number of clusters (K).
# The "elbow" point, where the rate of decrease in WCSS sharply changes, indicates the optimal K.
wcss = []
k_values = range(2, 8) # K from 2 to 7 as specified in the task

print("\nCalculating WCSS for K from 2 to 7 to determine optimal K using the Elbow Method...")
for k in k_values:
    # Initialize KMeans with n_init=10 to explicitly set the number of initializations
    # and suppress the warning in newer scikit-learn versions.
    # random_state ensures reproducibility.
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_) # inertia_ is the WCSS

# Plot the Elbow Method graph
plt.figure(figsize=(10, 6))
plt.plot(k_values, wcss, marker='o', linestyle='--')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
plt.xticks(k_values) # Ensure x-axis ticks correspond to K values
plt.grid(True)
plt.show()

# Based on the Elbow plot, visually inspect where the rate of decrease in WCSS sharply changes.
# For this dataset, a common 'elbow' point might be around K=3 or K=4.
# We will choose K=3 for demonstration purposes. Users should adjust this based on their observation of the plot.
chosen_k = 3
print(f"\nChosen number of clusters (K) based on Elbow Method observation: {chosen_k}")

# --- 5. Apply K-Means clustering with the chosen K ---
# Initialize and fit the K-Means model with the determined K.
kmeans_final = KMeans(n_clusters=chosen_k, random_state=42, n_init=10)
# Predict cluster labels for each data point and add them to the original DataFrame.
df['cluster'] = kmeans_final.fit_predict(X_scaled)

print(f"\nK-Means clustering applied with K={chosen_k}.")
print("Distribution of data points across clusters:")
print(df['cluster'].value_counts().sort_index())

# --- 6. Reduce the dimensionality of the standardized features to 2 components using PCA ---
# PCA (Principal Component Analysis) is used to reduce the number of dimensions
# while retaining as much variance as possible.
pca = PCA(n_components=2)
principal_components = pca.fit_transform(X_scaled)

# Create a DataFrame for the principal components for easier handling and plotting.
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])

print("\nDimensionality reduced to 2 components using PCA.")
print(f"Explained variance ratio of PC1: {pca.explained_variance_ratio_[0]*100:.2f}%")
print(f"Explained variance ratio of PC2: {pca.explained_variance_ratio_[1]*100:.2f}%")
print(f"Cumulative explained variance by 2 components: {sum(pca.explained_variance_ratio_)*100:.2f}%")

# --- 7. Create a scatter plot of the two principal components, coloring the points by their assigned cluster labels ---
# Merge the cluster labels with the PCA components DataFrame.
pca_df['cluster'] = df['cluster']

plt.figure(figsize=(10, 8))
# Create a scatter plot. 'c' parameter colors points by cluster label.
# 'cmap' sets the colormap, 's' sets marker size, 'alpha' sets transparency.
scatter = plt.scatter(pca_df['PC1'], pca_df['PC2'],
                      c=pca_df['cluster'],
                      cmap='viridis', # A good default colormap for distinct categories
                      s=50,
                      alpha=0.8)
plt.title(f'K-Means Clustering (K={chosen_k}) on PCA-reduced Features')
# Add explained variance to axis labels for better interpretation.
plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.2f}%)')
plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.2f}%)')
# Add a color bar to show which color corresponds to which cluster label.
plt.colorbar(scatter, label='Cluster Label')
plt.grid(True)
plt.show()

print("\nClustering visualization complete.")
```