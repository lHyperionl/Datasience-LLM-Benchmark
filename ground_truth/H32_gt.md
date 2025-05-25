```python
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# --- 1. Load Data ---
# Assume spotify_2023.csv is in the current directory
try:
    df = pd.read_csv('spotify_2023.csv')
except FileNotFoundError:
    print("Error: 'spotify_2023.csv' not found. Please ensure the CSV file is in the same directory as the script.")
    exit()

# --- 2. Select audio features and handle NaNs ---
# Define the list of audio features to be used
audio_features = [
    'danceability_%', 'energy_%', 'valence_%', 'acousticness_%',
    'instrumentalness_%', 'liveness_%', 'speechiness_%', 'bpm'
]

# Create a DataFrame with only the selected features
# Use .copy() to ensure we're working on a separate copy and avoid SettingWithCopyWarning
df_features = df[audio_features].copy()

# Drop rows with any NaN values in these selected features
# This ensures that all rows used for clustering have complete data
initial_rows = len(df_features)
df_features.dropna(inplace=True)
rows_after_nan_drop = len(df_features)
print(f"Original rows with selected features: {initial_rows}")
print(f"Rows after dropping NaNs: {rows_after_nan_drop}")
print(f"Number of rows dropped due to NaNs: {initial_rows - rows_after_nan_drop}\n")

# Store the original (unstandardized) data for mean calculation later
# This DataFrame will be used to attach cluster labels and calculate means
df_original_for_analysis = df_features.copy()

# --- 3. Standardize these features using StandardScaler ---
scaler = StandardScaler()
# Fit the scaler to the data and transform it
X_scaled = scaler.fit_transform(df_features)

# Convert the scaled array back to a DataFrame for better readability and future use
# (though for K-Means and PCA, the numpy array X_scaled is sufficient)
df_scaled = pd.DataFrame(X_scaled, columns=audio_features, index=df_features.index)

# --- 4. Apply K-Means clustering ---
# Initialize KMeans with K=4, specified random_state for reproducibility, and n_init='auto'
kmeans = KMeans(n_clusters=4, random_state=42, n_init='auto')

# Fit KMeans to the scaled data and predict cluster labels
cluster_labels = kmeans.fit_predict(X_scaled)

# Add the cluster labels to the original (unstandardized) DataFrame for analysis
df_original_for_analysis['cluster'] = cluster_labels

# --- 5. Reduce standardized features to 2 principal components using PCA ---
# Initialize PCA with 2 components and specified random_state for reproducibility
pca = PCA(n_components=2, random_state=42)

# Fit PCA to the scaled data and transform it to 2 principal components
principal_components = pca.fit_transform(X_scaled)

# Create a DataFrame for the principal components for easier plotting
df_pca = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])

# Add the cluster labels to the PCA DataFrame for coloring the plot
df_pca['cluster'] = cluster_labels

# --- 6. Create a scatter plot of the two principal components ---
plt.figure(figsize=(10, 7))
scatter = plt.scatter(
    df_pca['PC1'],
    df_pca['PC2'],
    c=df_pca['cluster'],  # Color points by their K-Means assigned cluster labels
    cmap='viridis',       # Colormap for clusters
    s=50,                 # Marker size
    alpha=0.7             # Transparency
)

# Add title and labels
plt.title('K-Means Clusters (K=4) on PCA-Reduced Spotify Audio Features')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

# Add a color bar to indicate cluster labels
cbar = plt.colorbar(scatter, ticks=range(4))
cbar.set_label('Cluster Label')

plt.grid(True)
plt.show()

# --- 7. For each cluster, calculate and display the mean values of the original (unstandardized) selected audio features ---
print("\n--- Mean values of original (unstandardized) audio features per cluster ---")
# Group the original data by cluster and calculate the mean for each audio feature
cluster_means = df_original_for_analysis.groupby('cluster')[audio_features].mean()

# Display the results
print(cluster_means)
```