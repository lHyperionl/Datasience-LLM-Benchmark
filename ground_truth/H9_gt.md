```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from scipy.spatial.distance import cdist
from scipy.cluster.hierarchy import dendrogram, linkage

from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, adjusted_rand_score
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import IsolationForest

import umap
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# --- Configuration ---
FILE_PATH = 'water_potability.csv'
RANDOM_STATE = 42
N_COMPONENTS_DR = 2 # For 2D visualizations
N_COMPONENTS_DR_3D = 3 # For 3D visualizations
K_RANGE = range(2, 10) # Range for optimal K determination
N_BOOTSTRAPS = 50 # For cluster stability analysis
N_CONSENSUS_RUNS = 50 # For consensus clustering

# --- 1. Data Loading and Initial Preprocessing ---
print("--- 1. Data Loading and Initial Preprocessing ---")
try:
    df = pd.read_csv(FILE_PATH)
    print(f"Dataset loaded successfully. Shape: {df.shape}")
except FileNotFoundError:
    print(f"Error: {FILE_PATH} not found. Please ensure the CSV file is in the current directory.")
    exit()

# Separate target variable if it exists, for later validation
if 'Potability' in df.columns:
    y = df['Potability']
    X = df.drop('Potability', axis=1)
else:
    print("Warning: 'Potability' column not found. Proceeding without external validation target.")
    y = None
    X = df.copy()

# Impute missing values using median strategy
imputer = SimpleImputer(strategy='median')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
print("Missing values imputed using median strategy.")

# --- 2. Advanced Feature Engineering ---
print("\n--- 2. Advanced Feature Engineering ---")

# Water Quality Indices (simplified for demonstration)
# These indices are illustrative and can be refined based on specific domain knowledge
# and regulatory standards (e.g., WHO, EPA).
# Normalization for index calculation (MinMaxScaler for 0-1 range)
scaler_minmax = MinMaxScaler()
X_normalized = pd.DataFrame(scaler_minmax.fit_transform(X_imputed), columns=X_imputed.columns)

# pH deviation from ideal (7.0-8.5)
X_imputed['pH_Deviation'] = np.abs(X_imputed['ph'] - 7.75) # Midpoint of 7.0-8.5
X_imputed['pH_Quality_Score'] = 1 - (X_imputed['pH_Deviation'] / X_imputed['pH_Deviation'].max())

# Hardness quality (ideal around 80-100 mg/L)
X_imputed['Hardness_Quality_Score'] = 1 - (np.abs(X_imputed['Hardness'] - 90) / X_imputed['Hardness'].max())

# Contamination Risk Score (higher values indicate higher risk)
# Assuming higher Chloramines, Trihalomethanes, Turbidity indicate higher risk
X_imputed['Contamination_Risk_Score'] = (
    X_normalized['Chloramines'] * 0.3 +
    X_normalized['Trihalomethanes'] * 0.4 +
    X_normalized['Turbidity'] * 0.3
)

# Chemical Balance Ratios
# Add a small epsilon to avoid division by zero
epsilon = 1e-6
X_imputed['Sulfate_Chloramines_Ratio'] = X_imputed['Sulfate'] / (X_imputed['Chloramines'] + epsilon)
X_imputed['OrganicCarbon_Trihalomethanes_Ratio'] = X_imputed['Organic_carbon'] / (X_imputed['Trihalomethanes'] + epsilon)
X_imputed['Solids_Conductivity_Ratio'] = X_imputed['Solids'] / (X_imputed['Conductivity'] + epsilon)

# Overall Water Quality Index (WQI) - simplified
# Lower values for contaminants are better, higher for beneficial (like pH in range)
# This is a very simplified WQI. A real WQI involves more complex weighting and sub-indices.
X_imputed['WQI_Overall'] = (
    X_imputed['pH_Quality_Score'] * 0.15 +
    X_imputed['Hardness_Quality_Score'] * 0.10 -
    X_normalized['Solids'] * 0.10 -
    X_normalized['Chloramines'] * 0.15 -
    X_normalized['Sulfate'] * 0.10 -
    X_normalized['Conductivity'] * 0.05 -
    X_normalized['Organic_carbon'] * 0.05 -
    X_normalized['Trihalomethanes'] * 0.15 -
    X_normalized['Turbidity'] * 0.15
)
# Scale WQI to be positive and interpretable (e.g., 0-1)
X_imputed['WQI_Overall'] = MinMaxScaler().fit_transform(X_imputed[['WQI_Overall']])

print(f"Engineered features added. New shape: {X_imputed.shape}")

# Temporal patterns: Not applicable for this static dataset.
# If temporal data were available, features like moving averages, seasonality,
# and trend components would be engineered.

# --- 3. Preprocessing and Scaling for Clustering ---
print("\n--- 3. Preprocessing and Scaling for Clustering ---")
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_imputed), columns=X_imputed.columns)
print("Data scaled using StandardScaler.")

# --- 4. Dimensionality Reduction for Visualization and Enhancement ---
print("\n--- 4. Dimensionality Reduction ---")
dr_results = {}

# PCA
pca = PCA(n_components=N_COMPONENTS_DR, random_state=RANDOM_STATE)
dr_results['PCA_2D'] = pca.fit_transform(X_scaled)
print(f"PCA (2D) explained variance ratio: {pca.explained_variance_ratio_.sum():.2f}")

pca_3d = PCA(n_components=N_COMPONENTS_DR_3D, random_state=RANDOM_STATE)
dr_results['PCA_3D'] = pca_3d.fit_transform(X_scaled)
print(f"PCA (3D) explained variance ratio: {pca_3d.explained_variance_ratio_.sum():.2f}")

# t-SNE (computationally intensive, consider running on PCA-reduced data for large datasets)
# For this dataset size, direct application is feasible.
tsne = TSNE(n_components=N_COMPONENTS_DR, random_state=RANDOM_STATE, perplexity=30, n_iter=1000)
dr_results['tSNE_2D'] = tsne.fit_transform(X_scaled)
print("t-SNE (2D) applied.")

# UMAP
reducer = umap.UMAP(n_components=N_COMPONENTS_DR, random_state=RANDOM_STATE)
dr_results['UMAP_2D'] = reducer.fit_transform(X_scaled)
print("UMAP (2D) applied.")

reducer_3d = umap.UMAP(n_components=N_COMPONENTS_DR_3D, random_state=RANDOM_STATE)
dr_results['UMAP_3D'] = reducer_3d.fit_transform(X_scaled)
print("UMAP (3D) applied.")

# Factor Analysis
# Factor Analysis assumes underlying latent factors. It can fail if n_components > n_features.
# Also, it's sensitive to multicollinearity.
try:
    fa = FactorAnalysis(n_components=N_COMPONENTS_DR, random_state=RANDOM_STATE)
    dr_results['FactorAnalysis_2D'] = fa.fit_transform(X_scaled)
    print("Factor Analysis (2D) applied.")
except ValueError as e:
    print(f"Factor Analysis failed: {e}. Skipping Factor Analysis.")
    dr_results['FactorAnalysis_2D'] = np.full((X_scaled.shape[0], N_COMPONENTS_DR), np.nan) # Placeholder

# --- 5. Determine Optimal Number of Clusters ---
print("\n--- 5. Determining Optimal Number of Clusters ---")
# Using K-Means for optimal K determination as it's widely used and interpretable.
# Metrics: Elbow (SSE), Silhouette, Davies-Bouldin, Calinski-Harabasz

sse = []
silhouette_scores = []
davies_bouldin_scores = []
calinski_harabasz_scores = []

for k in K_RANGE:
    kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    kmeans.fit(X_scaled)
    labels = kmeans.labels_

    sse.append(kmeans.inertia_)
    if k > 1: # Silhouette, DB, CH scores require at least 2 clusters
        silhouette_scores.append(silhouette_score(X_scaled, labels))
        davies_bouldin_scores.append(davies_bouldin_score(X_scaled, labels))
        calinski_harabasz_scores.append(calinski_harabasz_score(X_scaled, labels))
    else:
        silhouette_scores.append(np.nan)
        davies_bouldin_scores.append(np.nan)
        calinski_harabasz_scores.append(np.nan)

# Plotting metrics
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Optimal Number of Clusters Validation Metrics')

axes[0, 0].plot(K_RANGE, sse, marker='o')
axes[0, 0].set_title('Elbow Method (SSE)')
axes[0, 0].set_xlabel('Number of Clusters (K)')
axes[0, 0].set_ylabel('SSE')

axes[0, 1].plot(K_RANGE, silhouette_scores, marker='o')
axes[0, 1].set_title('Silhouette Score')
axes[0, 1].set_xlabel('Number of Clusters (K)')
axes[0, 1].set_ylabel('Score')

axes[1, 0].plot(K_RANGE, davies_bouldin_scores, marker='o')
axes[1, 0].set_title('Davies-Bouldin Index (Lower is Better)')
axes[1, 0].set_xlabel('Number of Clusters (K)')
axes[1, 0].set_ylabel('Index')

axes[1, 1].plot(K_RANGE, calinski_harabasz_scores, marker='o')
axes[1, 1].set_title('Calinski-Harabasz Index (Higher is Better)')
axes[1, 1].set_xlabel('Number of Clusters (K)')
axes[1, 1].set_ylabel('Index')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# Based on visual inspection of the plots, choose an optimal K.
# For demonstration, let's pick a K that often shows good balance, e.g., 3 or 4.
# In a real scenario, this would be a data scientist's decision.
optimal_k = 3
print(f"Selected optimal number of clusters (k) for subsequent analysis: {optimal_k}")

# --- 6. Multiple Clustering Algorithms ---
print("\n--- 6. Applying Multiple Clustering Algorithms ---")
cluster_labels = {}

# K-Means
kmeans = KMeans(n_clusters=optimal_k, random_state=RANDOM_STATE, n_init=10)
cluster_labels['KMeans'] = kmeans.fit_predict(X_scaled)
print("K-Means clustering applied.")

# Hierarchical Clustering (Agglomerative)
agg_clustering = AgglomerativeClustering(n_clusters=optimal_k)
cluster_labels['Hierarchical'] = agg_clustering.fit_predict(X_scaled)
print("Hierarchical clustering applied.")

# DBSCAN (requires careful parameter tuning)
# Estimate eps using k-distance graph (for min_samples = 2*n_features)
# For simplicity, let's use a fixed min_samples and estimate eps.
# min_samples = 2 * len(X_scaled.columns) # A common heuristic
min_samples_dbscan = 2 * len(X_scaled.columns) # A common heuristic
neighbors = NearestNeighbors(n_neighbors=min_samples_dbscan)
neighbors_fit = neighbors.fit(X_scaled)
distances, indices = neighbors_fit.kneighbors(X_scaled)
distances = np.sort(distances[:, min_samples_dbscan-1], axis=0)

plt.figure(figsize=(10, 6))
plt.plot(distances)
plt.title('K-distance Graph for DBSCAN Epsilon Estimation')
plt.xlabel('Points sorted by distance')
plt.ylabel(f'{min_samples_dbscan}-th Nearest Neighbor Distance')
plt.grid(True)
plt.show()

# Visually select an elbow point from the graph for eps.
# For demonstration, let's pick a value. This is highly data-dependent.
eps_dbscan = 2.5 # Example value, adjust based on the plot above
dbscan = DBSCAN(eps=eps_dbscan, min_samples=min_samples_dbscan)
cluster_labels['DBSCAN'] = dbscan.fit_predict(X_scaled)
print(f"DBSCAN applied with eps={eps_dbscan}, min_samples={min_samples_dbscan}.")
print(f"DBSCAN found {len(np.unique(cluster_labels['DBSCAN'])) - (1 if -1 in cluster_labels['DBSCAN'] else 0)} clusters and {np.sum(cluster_labels['DBSCAN'] == -1)} noise points.")

# Gaussian Mixture Models (GMM)
gmm = GaussianMixture(n_components=optimal_k, random_state=RANDOM_STATE)
gmm.fit(X_scaled)
cluster_labels['GMM'] = gmm.predict(X_scaled)
print("Gaussian Mixture Models applied.")

# Spectral Clustering
# Spectral clustering can be computationally expensive for large datasets.
# affinity='nearest_neighbors' is often preferred for non-linear structures.
try:
    spectral = SpectralClustering(n_clusters=optimal_k, assign_labels='discretize',
                                  random_state=RANDOM_STATE, affinity='nearest_neighbors', n_neighbors=10)
    cluster_labels['Spectral'] = spectral.fit_predict(X_scaled)
    print("Spectral Clustering applied.")
except Exception as e:
    print(f"Spectral Clustering failed: {e}. Skipping Spectral Clustering.")
    cluster_labels['Spectral'] = np.full(X_scaled.shape[0], -1) # Placeholder for failure

# Add cluster labels to the original (imputed) DataFrame for analysis
df_clustered = X_imputed.copy()
for algo, labels in cluster_labels.items():
    df_clustered[f'Cluster_{algo}'] = labels

# Add Potability column back for validation
if y is not None:
    df_clustered['Potability'] = y

# --- 7. Cluster Stability Analysis ---
print("\n--- 7. Cluster Stability Analysis (using K-Means) ---")

# Bootstrap Resampling
print("Performing Bootstrap Resampling...")
ari_scores = []
for i in range(N_BOOTSTRAPS):
    # Resample with replacement
    resampled_indices = np.random.choice(X_scaled.index, size=len(X_scaled), replace=True)
    X_resampled = X_scaled.iloc[resampled_indices]

    # Fit KMeans
    kmeans_resampled = KMeans(n_clusters=optimal_k, random_state=i, n_init=10)
    labels_resampled = kmeans_resampled.fit_predict(X_resampled)

    # To compare, we need to map clusters. This is a simplified approach.
    # A more robust approach would use a cluster matching algorithm (e.g., Hungarian algorithm).
    # For simplicity, we'll compare the resampled labels to the original KMeans labels.
    # This is an approximation as cluster labels are arbitrary.
    # A better metric for stability is to check if pairs of points stay together.
    # Let's use a pairwise co-occurrence matrix approach for stability.

    # Simplified stability: run KMeans on full data, then on bootstrapped data,
    # and compare how many points change clusters.
    # This is not ARI. ARI requires comparing two sets of labels on the same data.
    # For stability, we want to see if the *structure* is stable.
    # A common way is to compute ARI between different runs on the *same* dataset.
    # Or, for bootstrap, compute ARI between original labels and labels of original points in resample.

    # Let's use a more direct stability measure: how often do two points cluster together?
    # This is part of consensus clustering.

# Consensus Clustering (simplified implementation)
print("Performing Consensus Clustering...")
co_occurrence_matrix = np.zeros((len(X_scaled), len(X_scaled)))

for i in range(N_CONSENSUS_RUNS):
    kmeans_run = KMeans(n_clusters=optimal_k, random_state=i, n_init=10)
    labels_run = kmeans_run.fit_predict(X_scaled)
    
    # Update co-occurrence matrix
    for j in range(len(X_scaled)):
        for l in range(j + 1, len(X_scaled)):
            if labels_run[j] == labels_run[l]:
                co_occurrence_matrix[j, l] += 1
                co_occurrence_matrix[l, j] += 1 # Symmetric

# Normalize co-occurrence matrix to get similarity matrix (0-1)
similarity_matrix = co_occurrence_matrix / N_CONSENSUS_RUNS
np.fill_diagonal(similarity_matrix, 1) # A point is always similar to itself

# Convert similarity to distance for hierarchical clustering
distance_matrix = 1 - similarity_matrix

# Perform hierarchical clustering on the distance matrix
linked = linkage(distance_matrix, method='average')

plt.figure(figsize=(15, 7))
dendrogram(linked,
           orientation='top',
           distance_sort='descending',
           show_leaf_counts=True)
plt.title('Consensus Clustering Dendrogram')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.show()
print("Consensus clustering dendrogram generated. Stable clusters will show clear, deep branches.")

# --- 8. Analyze Water Quality Profiles for Each Cluster ---
print("\n--- 8. Analyzing Water Quality Profiles for Each Cluster (using KMeans) ---")

# Select one clustering result for detailed profiling, e.g., KMeans
df_profile = df_clustered.copy()
df_profile['Cluster'] = cluster_labels['KMeans']

cluster_profiles = {}
for cluster_id in sorted(df_profile['Cluster'].unique()):
    if cluster_id == -1: # Skip DBSCAN noise points if present
        continue
    cluster_data = df_profile[df_profile['Cluster'] == cluster_id]
    
    # Calculate mean of original and engineered features
    profile = cluster_data.drop(columns=['Cluster', 'Potability'], errors='ignore').mean().to_dict()
    
    # Calculate potability rate
    if 'Potability' in cluster_data.columns:
        profile['Potability_Rate'] = cluster_data['Potability'].mean() * 100
    else:
        profile['Potability_Rate'] = np.nan
    
    cluster_profiles[f'Cluster_{cluster_id}'] = profile

# Convert to DataFrame for easier viewing
cluster_profiles_df = pd.DataFrame.from_dict(cluster_profiles, orient='index')
print("\nCluster Profiles (Mean Values):")
print(cluster_profiles_df.round(2))

# --- 9. Comprehensive Cluster Visualization Dashboard ---
print("\n--- 9. Comprehensive Cluster Visualization Dashboard ---")

# Prepare data for plotting
plot_df = df_clustered.copy()
plot_df['KMeans_Cluster'] = cluster_labels['KMeans']
plot_df['Potability_Label'] = plot_df['Potability'].map({0: 'Not Potable', 1: 'Potable'})

# Add DR components to plot_df
plot_df['PCA1'] = dr_results['PCA_2D'][:, 0]
plot_df['PCA2'] = dr_results['PCA_2D'][:, 1]
plot_df['tSNE1'] = dr_results['tSNE_2D'][:, 0]
plot_df['tSNE2'] = dr_results['tSNE_2D'][:, 1]
plot_df['UMAP1'] = dr_results['UMAP_2D'][:, 0]
plot_df['UMAP2'] = dr_results['UMAP_2D'][:, 1]

plot_df['PCA_3D_1'] = dr_results['PCA_3D'][:, 0]
plot_df['PCA_3D_2'] = dr_results['PCA_3D'][:, 1]
plot_df['PCA_3D_3'] = dr_results['PCA_3D'][:, 2]

plot_df['UMAP_3D_1'] = dr_results['UMAP_3D'][:, 0]
plot_df['UMAP_3D_2'] = dr_results['UMAP_3D'][:, 1]
plot_df['UMAP_3D_3'] = dr_results['UMAP_3D'][:, 2]

# 2D Plots (PCA, t-SNE, UMAP)
print("Generating 2D/3D interactive plots...")
fig_pca = px.scatter(plot_df, x='PCA1', y='PCA2', color='KMeans_Cluster',
                     title='PCA 2D Visualization of Clusters',
                     hover_data=X_imputed.columns.tolist() + ['Potability_Label'])
fig_pca.show()

fig_tsne = px.scatter(plot_df, x='tSNE1', y='tSNE2', color='KMeans_Cluster',
                      title='t-SNE 2D Visualization of Clusters',
                      hover_data=X_imputed.columns.tolist() + ['Potability_Label'])
fig_tsne.show()

fig_umap = px.scatter(plot_df, x='UMAP1', y='UMAP2', color='KMeans_Cluster',
                      title='UMAP 2D Visualization of Clusters',
                      hover_data=X_imputed.columns.tolist() + ['Potability_Label'])
fig_umap.show()

# 3D Plots (PCA, UMAP)
fig_pca_3d = px.scatter_3d(plot_df, x='PCA_3D_1', y='PCA_3D_2', z='PCA_3D_3', color='KMeans_Cluster',
                           title='PCA 3D Visualization of Clusters',
                           hover_data=X_imputed.columns.tolist() + ['Potability_Label'])
fig_pca_3d.show()

fig_umap_3d = px.scatter_3d(plot_df, x='UMAP_3D_1', y='UMAP_3D_2', z='UMAP_3D_3', color='KMeans_Cluster',
                            title='UMAP 3D Visualization of Clusters',
                            hover_data=X_imputed.columns.tolist() + ['Potability_Label'])
fig_umap_3d.show()

# Radar Charts for Cluster Profiles
print("Generating Radar Charts for Cluster Profiles...")
# Select key features for radar chart (original and engineered)
radar_features = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity',
                  'Organic_carbon', 'Trihalomethanes', 'Turbidity',
                  'Contamination_Risk_Score', 'WQI_Overall']

# Normalize cluster profiles for radar chart (0-1 scale across all clusters for each feature)
radar_df = cluster_profiles_df[radar_features].copy()
radar_df_scaled = pd.DataFrame(MinMaxScaler().fit_transform(radar_df.T).T,
                               columns=radar_df.columns, index=radar_df.index)

fig_radar = go.Figure()
for i, cluster_name in enumerate(radar_df_scaled.index):
    fig_radar.add_trace(go.Scatterpolar(
        r=radar_df_scaled.loc[cluster_name].values,
        theta=radar_df_scaled.columns,
        fill='toself',
        name=cluster_name
    ))

fig_radar.update_layout(
    polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0, 1]
        )),
    showlegend=True,
    title='Radar Chart of Cluster Profiles (Normalized Features)'
)
fig_radar.show()

# Potability Rate per Cluster
if y is not None:
    potability_rates = df_profile.groupby('KMeans_Cluster')['Potability'].mean() * 100
    fig_potability = px.bar(x=potability_rates.index, y=potability_rates.values,
                            labels={'x': 'Cluster', 'y': 'Potability Rate (%)'},
                            title='Potability Rate per Cluster')
    fig_potability.show()

# --- 10. External Validation and Domain Knowledge ---
print("\n--- 10. External Validation and Domain Knowledge ---")
print("Interpreting cluster profiles against water quality standards:")
print("For example, WHO guidelines for drinking water:")
print("- pH: 6.5-8.5")
print("- Chloramines: < 4 mg/L")
print("- Trihalomethanes: < 0.08 mg/L (80 µg/L)")
print("- Turbidity: < 5 NTU (ideally < 1 NTU)")
print("- Sulfate: < 250 mg/L")
print("- Hardness: No health guideline, but aesthetic issues above 200 mg/L (often expressed as CaCO3)")
print("- TDS (Solids): < 500 mg/L (aesthetic, taste)")
print("\nBy examining the 'Cluster Profiles (Mean Values)' table, one can compare the mean values of each parameter within a cluster to these standards.")
print("For instance, a cluster with high 'Contamination_Risk_Score', high 'Chloramines', 'Trihalomethanes', and low 'WQI_Overall' might represent water samples that are likely unsafe or require significant treatment.")
print("Conversely, a cluster with values within ideal ranges for most parameters and a high 'WQI_Overall' would represent high-quality water.")

# --- 11. Cluster-based Anomaly Detection ---
print("\n--- 11. Cluster-based Anomaly Detection ---")

# Method 1: Distance from Cluster Centroid
print("Anomaly Detection: Distance from Cluster Centroid")
kmeans_model = KMeans(n_clusters=optimal_k, random_state=RANDOM_STATE, n_init=10)
kmeans_model.fit(X_scaled)
cluster_labels_kmeans = kmeans_model.labels_
centroids = kmeans_model.cluster_centers_

# Calculate distance of each point to its assigned centroid
distances_to_centroid = []
for i, label in enumerate(cluster_labels_kmeans):
    if label != -1: # Exclude DBSCAN noise if using DBSCAN labels
        distances_to_centroid.append(cdist(X_scaled.iloc[[i]], centroids[[label]], 'euclidean')[0][0])
    else:
        distances_to_centroid.append(np.nan) # For noise points

df_anomalies = df_clustered.copy()
df_anomalies['KMeans_Cluster'] = cluster_labels_kmeans
df_anomalies['Distance_to_Centroid'] = distances_to_centroid

# Define anomaly threshold (e.g., 99th percentile of distances)
anomaly_threshold = df_anomalies['Distance_to_Centroid'].quantile(0.99)
df_anomalies['Is_Anomaly_Centroid'] = df_anomalies['Distance_to_Centroid'] > anomaly_threshold

print(f"Identified {df_anomalies['Is_Anomaly_Centroid'].sum()} anomalies based on distance to centroid (threshold: {anomaly_threshold:.2f}).")

# Method 2: Isolation Forest within each cluster
print("Anomaly Detection: Isolation Forest within each cluster")
df_anomalies['Is_Anomaly_IsolationForest'] = False

for cluster_id in sorted(df_anomalies['KMeans_Cluster'].unique()):
    if cluster_id == -1: # Skip noise points
        continue
    
    cluster_data_scaled = X_scaled[cluster_labels_kmeans == cluster_id]
    
    if len(cluster_data_scaled) > 1: # Need at least 2 samples for IsolationForest
        iso_forest = IsolationForest(random_state=RANDOM_STATE, contamination='auto') # 'auto' tries to estimate
        iso_forest.fit(cluster_data_scaled)
        
        # -1 for outliers, 1 for inliers
        outlier_predictions = iso_forest.predict(cluster_data_scaled)
        
        # Map back to original DataFrame
        original_indices = df_anomalies[df_anomalies['KMeans_Cluster'] == cluster_id].index
        df_anomalies.loc[original_indices[outlier_predictions == -1], 'Is_Anomaly_IsolationForest'] = True

print(f"Identified {df_anomalies['Is_Anomaly_IsolationForest'].sum()} anomalies based on Isolation Forest within clusters.")

# Display some anomalies
print("\nSample of identified anomalies (Distance to Centroid):")
print(df_anomalies[df_anomalies['Is_Anomaly_Centroid']].head())

print("\nSample of identified anomalies (Isolation Forest):")
print(df_anomalies[df_anomalies['Is_Anomaly_IsolationForest']].head())

print("\n--- Analysis Complete ---")
```