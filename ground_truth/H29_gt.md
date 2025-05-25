```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import numpy as np

# --- 1. Load the dataset ---
try:
    df = pd.read_csv('heart.csv')
    print("Dataset 'heart.csv' loaded successfully.")
except FileNotFoundError:
    print("Error: 'heart.csv' not found. Please ensure the file is in the current directory.")
    exit() # Exit the script if the file is not found

# --- 2. Select the specified numerical features for clustering ---
selected_features = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
features_for_clustering = df[selected_features]
print(f"\nSelected features for clustering: {selected_features}")
print(f"Shape of selected features: {features_for_clustering.shape}")

# --- 3. Standardize these selected features using StandardScaler ---
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features_for_clustering)
print("\nFeatures standardized using StandardScaler.")

# Convert scaled features back to a DataFrame for clarity, though not strictly necessary for KMeans/PCA
scaled_features_df = pd.DataFrame(scaled_features, columns=selected_features)

# --- 4. Determine an appropriate number of clusters (K) for K-Means ---
# Trying values for K from 2 to 5.
# Use the elbow method (plotting Within-Cluster Sum of Squares - WCSS) and silhouette scores.

wcss = []
silhouette_scores = []
k_range = range(2, 6) # K from 2 to 5 (exclusive upper bound)

print("\nDetermining optimal K using Elbow Method (WCSS) and Silhouette Score:")
for k in k_range:
    # Initialize KMeans with n_init='auto' and random_state for reproducibility
    kmeans = KMeans(n_clusters=k, n_init='auto', random_state=42)
    kmeans.fit(scaled_features)
    wcss.append(kmeans.inertia_) # WCSS (Within-Cluster Sum of Squares)
    
    # Calculate Silhouette Score
    # Silhouette score requires at least 2 clusters, which is covered by k_range starting at 2
    score = silhouette_score(scaled_features, kmeans.labels_)
    silhouette_scores.append(score)
    print(f"  K={k}: WCSS = {kmeans.inertia_:.2f}, Silhouette Score = {score:.2f}")

# Plotting WCSS (Elbow Method)
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(k_range, wcss, marker='o', linestyle='-')
plt.title('Elbow Method for Optimal K (WCSS)')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
plt.xticks(k_range)
plt.grid(True)

# Plotting Silhouette Scores
plt.subplot(1, 2, 2)
plt.plot(k_range, silhouette_scores, marker='o', linestyle='-')
plt.title('Silhouette Scores for Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Silhouette Score')
plt.xticks(k_range)
plt.grid(True)

plt.tight_layout()
plt.show()

# Based on the plots, choose an appropriate K.
# For this dataset and K range, K=3 often represents a good balance,
# where the WCSS curve starts to flatten (the "elbow") and silhouette score is relatively high.
# In a real analysis, one would visually inspect these plots to make the most informed decision.
chosen_k = 3 # Example choice based on typical elbow/silhouette patterns for this range.
print(f"\nChosen K for K-Means clustering: {chosen_k} (based on visual inspection of Elbow and Silhouette plots).")

# --- 5. Apply K-Means clustering with the chosen K ---
kmeans_final = KMeans(n_clusters=chosen_k, n_init='auto', random_state=42)
cluster_labels = kmeans_final.fit_predict(scaled_features)
print(f"\nK-Means clustering applied with K={chosen_k}.")
print(f"Cluster labels generated for {len(cluster_labels)} data points.")

# --- 6. Reduce the dimensionality of the standardized features to 2 components using PCA ---
pca = PCA(n_components=2)
principal_components = pca.fit_transform(scaled_features)
print(f"\nDimensionality reduced to 2 components using PCA.")
print(f"Explained variance ratio by PC1: {pca.explained_variance_ratio_[0]:.2f}")
print(f"Explained variance ratio by PC2: {pca.explained_variance_ratio_[1]:.2f}")
print(f"Total explained variance by 2 PCs: {pca.explained_variance_ratio_.sum():.2f}")

# Create a DataFrame for the principal components and add cluster labels
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
pca_df['Cluster'] = cluster_labels

# --- 7. Create a scatter plot of the two principal components, coloring the points by their assigned K-Means cluster labels ---
plt.figure(figsize=(10, 7))

# Define a colormap to distinguish clusters visually
colors = plt.cm.viridis(np.linspace(0, 1, chosen_k))

# Plot each cluster separately to ensure distinct colors and labels in the legend
for i in range(chosen_k):
    cluster_data = pca_df[pca_df['Cluster'] == i]
    plt.scatter(cluster_data['PC1'], cluster_data['PC2'],
                color=colors[i], label=f'Cluster {i}',
                s=60, alpha=0.7, edgecolors='w', linewidth=0.5)

plt.title(f'K-Means Clusters (K={chosen_k}) Visualized with PCA')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Cluster')
plt.grid(True)
plt.show()

print("\nK-Means clustering and PCA visualization complete.")
```