```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, OPTICS, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from scipy.stats import f_oneway, chi2_contingency, kruskal
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.metrics.pairwise import euclidean_distances
import umap.umap_ as umap # Requires `pip install umap-learn`
import tensorflow as tf # Requires `pip install tensorflow`
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from mlxtend.frequent_patterns import apriori, association_rules # Requires `pip install mlxtend`
from itertools import combinations
import warnings

# Suppress warnings for cleaner output, especially from sklearn and tensorflow
warnings.filterwarnings('ignore')

# --- Configuration ---
DATASET_PATH = 'diabetes.csv'
RANDOM_STATE = 42 # For reproducibility
N_COMPONENTS_AE = 2 # Dimension for autoencoder embeddings (for 2D visualization)
N_CLUSTERS_MAX = 10 # Max number of clusters to test for algorithms requiring 'k'

# --- 1. Load and Preprocess Dataset with Advanced Feature Engineering ---
print("--- 1. Loading and Preprocessing Data with Advanced Feature Engineering ---")

try:
    df = pd.read_csv(DATASET_PATH)
    print(f"Dataset loaded successfully from {DATASET_PATH}. Shape: {df.shape}")
except FileNotFoundError:
    print(f"Error: {DATASET_PATH} not found. Please ensure the file is in the current directory.")
    exit()

# Define original features and target
features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
target = 'Outcome'

# Handle missing values (0s in certain columns are often invalid/missing in this dataset)
# Replace 0s with NaN for imputation
cols_to_impute = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in cols_to_impute:
    df[col] = df[col].replace(0, np.nan)

# Impute missing values using the median strategy
imputer = SimpleImputer(strategy='median')
df[features] = imputer.fit_transform(df[features])

# Advanced Feature Engineering: Metabolic Syndrome Indicators
# Metabolic syndrome is a cluster of conditions. We'll create proxies based on available features.
# Thresholds are simplified for demonstration and may vary clinically.
# Glucose: Fasting blood sugar >= 100 mg/dL (prediabetes/diabetes)
# BloodPressure: Hypertension >= 80 mmHg (diastolic) or >= 130 mmHg (systolic) - using 80 for simplicity
# BMI: Overweight >= 25, Obese >= 30
df['High_Glucose'] = (df['Glucose'] >= 100).astype(int)
df['High_BP'] = (df['BloodPressure'] >= 80).astype(int)
df['High_BMI'] = (df['BMI'] >= 25).astype(int)

# Create a composite metabolic syndrome score (sum of binary indicators)
df['Metabolic_Syndrome_Score'] = df['High_Glucose'] + df['High_BP'] + df['High_BMI']

# Create a binary indicator for potential metabolic syndrome (e.g., score >= 2 components)
df['Potential_Metabolic_Syndrome'] = (df['Metabolic_Syndrome_Score'] >= 2).astype(int)

# Update the list of features to include the newly engineered ones
engineered_features = features + ['High_Glucose', 'High_BP', 'High_BMI', 'Metabolic_Syndrome_Score', 'Potential_Metabolic_Syndrome']

# Scale numerical features for clustering and dimensionality reduction
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[engineered_features])
X_scaled_df = pd.DataFrame(X_scaled, columns=engineered_features)

print(f"Data preprocessed and scaled. {len(engineered_features) - len(features)} new features added. Scaled data shape: {X_scaled_df.shape}")

# --- 2. Advanced Dimensionality Reduction Techniques for Visualization ---
print("\n--- 2. Advanced Dimensionality Reduction for Visualization ---")

# t-SNE for non-linear dimensionality reduction
print("Running t-SNE (this may take a moment)...")
tsne = TSNE(n_components=2, random_state=RANDOM_STATE, perplexity=30, n_iter=1000, learning_rate=200)
X_tsne = tsne.fit_transform(X_scaled)
print(f"t-SNE reduced data shape: {X_tsne.shape}")

# UMAP for non-linear dimensionality reduction (often faster than t-SNE)
print("Running UMAP...")
reducer = umap.UMAP(n_components=2, random_state=RANDOM_STATE)
X_umap = reducer.fit_transform(X_scaled)
print(f"UMAP reduced data shape: {X_umap.shape}")

# Autoencoder-based embeddings
print("Building and training Autoencoder for embeddings...")
input_dim = X_scaled.shape[1]

# Encoder architecture
input_layer = Input(shape=(input_dim,))
encoder = Dense(64, activation='relu')(input_layer)
encoder = Dense(32, activation='relu')(encoder)
encoder_output = Dense(N_COMPONENTS_AE, activation='linear')(encoder) # Linear activation for embeddings

# Decoder architecture
decoder = Dense(32, activation='relu')(encoder_output)
decoder = Dense(64, activation='relu')(decoder)
decoder_output = Dense(input_dim, activation='linear')(decoder) # Linear activation for reconstruction

# Autoencoder model
autoencoder = Model(inputs=input_layer, outputs=decoder_output)
# Encoder model to extract embeddings
encoder_model = Model(inputs=input_layer, outputs=encoder_output)

autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
# Train the autoencoder (verbose=0 to suppress output)
history = autoencoder.fit(X_scaled, X_scaled, epochs=50, batch_size=32, shuffle=True, verbose=0)
X_ae = encoder_model.predict(X_scaled)
print(f"Autoencoder embeddings shape: {X_ae.shape}")

# --- 3. Multiple Clustering Algorithms ---
print("\n--- 3. Implementing Multiple Clustering Algorithms ---")

clustering_results = {}
X_for_clustering = X_scaled # Use the scaled data for clustering

# K-Means Clustering
print("Running K-Means...")
# Determine optimal K using Elbow Method and Silhouette Score
inertia = []
silhouette_scores_kmeans = []
k_range = range(2, N_CLUSTERS_MAX + 1)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    kmeans.fit(X_for_clustering)
    inertia.append(kmeans.inertia_)
    if k > 1:
        silhouette_scores_kmeans.append(silhouette_score(X_for_clustering, kmeans.labels_))

# Plot Elbow Method
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(k_range, inertia, marker='o')
plt.title('Elbow Method for K-Means')
plt.xlabel('Number of clusters (K)')
plt.ylabel('Inertia')

# Plot Silhouette Scores
plt.subplot(1, 2, 2)
plt.plot(k_range[1:], silhouette_scores_kmeans, marker='o')
plt.title('Silhouette Scores for K-Means')
plt.xlabel('Number of clusters (K)')
plt.ylabel('Silhouette Score')
plt.tight_layout()
plt.show()

# Choose optimal K based on highest silhouette score (or visual inspection of elbow)
optimal_k_kmeans = k_range[1:][np.argmax(silhouette_scores_kmeans)] if silhouette_scores_kmeans else 3
print(f"Optimal K for K-Means (based on silhouette): {optimal_k_kmeans}")
kmeans = KMeans(n_clusters=optimal_k_kmeans, random_state=RANDOM_STATE, n_init=10)
clustering_results['KMeans'] = kmeans.fit_predict(X_for_clustering)

# Gaussian Mixture Models (GMM)
print("Running GMM...")
# Determine optimal K using BIC and AIC
bic_scores = []
aic_scores = []
for k in k_range:
    gmm = GaussianMixture(n_components=k, random_state=RANDOM_STATE)
    gmm.fit(X_for_clustering)
    bic_scores.append(gmm.bic(X_for_clustering))
    aic_scores.append(gmm.aic(X_for_clustering))

# Plot BIC/AIC
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(k_range, bic_scores, marker='o', label='BIC')
plt.title('BIC for GMM')
plt.xlabel('Number of components (K)')
plt.ylabel('BIC Score')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(k_range, aic_scores, marker='o', label='AIC')
plt.title('AIC for GMM')
plt.xlabel('Number of components (K)')
plt.ylabel('AIC Score')
plt.legend()
plt.tight_layout()
plt.show()

# Choose optimal K based on lowest BIC score
optimal_k_gmm = k_range[np.argmin(bic_scores)]
print(f"Optimal K for GMM (based on BIC): {optimal_k_gmm}")
gmm = GaussianMixture(n_components=optimal_k_gmm, random_state=RANDOM_STATE)
clustering_results['GMM'] = gmm.fit_predict(X_for_clustering)

# Hierarchical Clustering (Agglomerative)
print("Running Hierarchical Clustering...")
# Determine optimal K using Silhouette Score
silhouette_scores_hc = []
for k in k_range:
    hc = AgglomerativeClustering(n_clusters=k)
    labels = hc.fit_predict(X_for_clustering)
    if k > 1:
        silhouette_scores_hc.append(silhouette_score(X_for_clustering, labels))

plt.figure(figsize=(6, 5))
plt.plot(k_range[1:], silhouette_scores_hc, marker='o')
plt.title('Silhouette Scores for Hierarchical Clustering')
plt.xlabel('Number of clusters (K)')
plt.ylabel('Silhouette Score')
plt.tight_layout()
plt.show()

optimal_k_hc = k_range[1:][np.argmax(silhouette_scores_hc)] if silhouette_scores_hc else 3
print(f"Optimal K for Hierarchical Clustering (based on silhouette): {optimal_k_hc}")
hc = AgglomerativeClustering(n_clusters=optimal_k_hc)
clustering_results['Hierarchical'] = hc.fit_predict(X_for_clustering)

# DBSCAN
print("Running DBSCAN...")
# Estimate eps using k-distance graph (e.g., k = min_samples)
# A common heuristic for min_samples is 2 * number of features for high-dimensional data.
min_samples_dbscan = 2 * X_for_clustering.shape[1]
neighbors = NearestNeighbors(n_neighbors=min_samples_dbscan)
neighbors_fit = neighbors.fit(X_for_clustering)
distances, indices = neighbors_fit.kneighbors(X_for_clustering)
distances = np.sort(distances[:, -1], axis=0) # Sort distances to the k-th nearest neighbor

plt.figure(figsize=(8, 6))
plt.plot(distances)
plt.title('K-distance Graph for DBSCAN (to estimate eps)')
plt.xlabel('Points sorted by distance')
plt.ylabel(f'{min_samples_dbscan}-th Nearest Neighbor Distance')
plt.grid(True)
plt.show()

# Visually inspect the elbow point in the k-distance graph to choose eps.
# For demonstration, a placeholder value is used. This should be chosen from the plot.
eps_dbscan = 0.5 # Example: This value should be determined by inspecting the k-distance graph
print(f"Chosen eps for DBSCAN: {eps_dbscan}, min_samples: {min_samples_dbscan}")

dbscan = DBSCAN(eps=eps_dbscan, min_samples=min_samples_dbscan)
clustering_results['DBSCAN'] = dbscan.fit_predict(X_for_clustering)
print(f"DBSCAN found {len(np.unique(clustering_results['DBSCAN'])) - (1 if -1 in clustering_results['DBSCAN'] else 0)} clusters (excluding noise).")

# OPTICS
print("Running OPTICS...")
# OPTICS is similar to DBSCAN but doesn't require 'eps' explicitly.
# It builds a reachability plot from which clusters can be extracted.
optics = OPTICS(min_samples=min_samples_dbscan, xi=0.05, min_cluster_size=0.05) # xi and min_cluster_size for automatic cluster extraction
clustering_results['OPTICS'] = optics.fit_predict(X_for_clustering)
print(f"OPTICS found {len(np.unique(clustering_results['OPTICS'])) - (1 if -1 in clustering_results['OPTICS'] else 0)} clusters (excluding noise).")

# Spectral Clustering
print("Running Spectral Clustering...")
# Determine optimal K using Silhouette Score
silhouette_scores_sc = []
for k in k_range:
    try:
        # Spectral clustering can be computationally intensive. 'nearest_neighbors' affinity is common.
        sc = SpectralClustering(n_clusters=k, random_state=RANDOM_STATE, assign_labels='kmeans', affinity='nearest_neighbors')
        labels = sc.fit_predict(X_for_clustering)
        if k > 1 and len(np.unique(labels)) > 1: # Ensure more than one cluster for silhouette
            silhouette_scores_sc.append(silhouette_score(X_for_clustering, labels))
        else:
            silhouette_scores_sc.append(-1) # Indicate invalid score
    except Exception as e:
        print(f"Warning: Spectral Clustering failed for k={k} with error: {e}. Skipping.")
        silhouette_scores_sc.append(-1) # Indicate failure

plt.figure(figsize=(6, 5))
plt.plot(k_range[1:], silhouette_scores_sc, marker='o')
plt.title('Silhouette Scores for Spectral Clustering')
plt.xlabel('Number of clusters (K)')
plt.ylabel('Silhouette Score')
plt.tight_layout()
plt.show()

optimal_k_sc = k_range[1:][np.argmax(silhouette_scores_sc)] if silhouette_scores_sc and max(silhouette_scores_sc) > -1 else 3
print(f"Optimal K for Spectral Clustering (based on silhouette): {optimal_k_sc}")
sc = SpectralClustering(n_clusters=optimal_k_sc, random_state=RANDOM_STATE, assign_labels='kmeans', affinity='nearest_neighbors')
clustering_results['Spectral'] = sc.fit_predict(X_for_clustering)

# Store all cluster labels in the original DataFrame
for algo, labels in clustering_results.items():
    df[f'Cluster_{algo}'] = labels
    print(f"{algo} clustering completed. Unique clusters: {len(np.unique(labels)) - (1 if -1 in labels else 0)}")

# --- 4. Consensus Clustering ---
print("\n--- 4. Performing Consensus Clustering ---")

# Filter out algorithms that produced only noise or a single cluster for consensus
valid_algos_for_consensus = [algo for algo, labels in clustering_results.items() if len(np.unique(labels)) > 1 and -1 not in labels]

if not valid_algos_for_consensus:
    print("No valid clustering algorithms found for consensus clustering (all produced only noise or 1 cluster). Skipping consensus.")
    df['Cluster_Consensus'] = np.zeros(df.shape[0], dtype=int) # Assign all to a single default cluster
else:
    n_samples = X_scaled.shape[0]
    co_occurrence_matrix = np.zeros((n_samples, n_samples))

    # Build a co-occurrence matrix: count how many times each pair of samples are in the same cluster
    for algo in valid_algos_for_consensus:
        labels = clustering_results[algo]
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                if labels[i] == labels[j]:
                    co_occurrence_matrix[i, j] += 1
                    co_occurrence_matrix[j, i] += 1

    # Normalize the co-occurrence matrix to get a similarity matrix
    num_algos_used = len(valid_algos_for_consensus)
    if num_algos_used > 0:
        similarity_matrix = co_occurrence_matrix / num_algos_used
        np.fill_diagonal(similarity_matrix, 1) # A sample is always similar to itself

        # Convert similarity to distance for hierarchical clustering linkage
        distance_matrix = 1 - similarity_matrix
        # Ensure distance matrix is symmetric and non-negative
        distance_matrix[distance_matrix < 0] = 0
        distance_matrix = (distance_matrix + distance_matrix.T) / 2 # Ensure perfect symmetry

        # Check if the distance matrix is valid for linkage (e.g., not all zeros)
        if np.all(distance_matrix == 0):
            print("Warning: Distance matrix for consensus clustering is all zeros. Cannot perform linkage.")
            df['Cluster_Consensus'] = np.zeros(n_samples, dtype=int) # Assign all to one cluster
        else:
            try:
                # Use linkage on the condensed distance matrix (upper triangle)
                linked = linkage(distance_matrix[np.triu_indices(n_samples, k=1)], method='average')

                # Plot dendrogram (optional, for visualization)
                plt.figure(figsize=(15, 7))
                dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True)
                plt.title('Consensus Clustering Dendrogram')
                plt.xlabel('Sample Index')
                plt.ylabel('Distance')
                plt.show()

                # Determine optimal number of consensus clusters using silhouette score
                consensus_silhouette_scores = []
                for k in k_range:
                    if k > 1:
                        consensus_labels = fcluster(linked, k, criterion='maxclust')
                        if len(np.unique(consensus_labels)) > 1:
                            consensus_silhouette_scores.append(silhouette_score(X_for_clustering, consensus_labels))
                        else:
                            consensus_silhouette_scores.append(-1) # Invalid score if only one cluster
                    else:
                        consensus_silhouette_scores.append(-1) # Invalid for k=1

                if any(s > -1 for s in consensus_silhouette_scores):
                    optimal_k_consensus = k_range[np.argmax(consensus_silhouette_scores)]
                    print(f"Optimal K for Consensus Clustering (based on silhouette): {optimal_k_consensus}")
                    consensus_labels = fcluster(linked, optimal_k_consensus, criterion='maxclust')
                else:
                    optimal_k_consensus = 3 # Default if no good silhouette found
                    consensus_labels = fcluster(linked, optimal_k_consensus, criterion='maxclust')
                    print(f"Could not determine optimal K for Consensus, defaulting to {optimal_k_consensus} clusters.")

                df['Cluster_Consensus'] = consensus_labels
                print(f"Consensus clustering completed. Number of clusters: {len(np.unique(consensus_labels))}")
            except Exception as e:
                print(f"Error during consensus clustering linkage: {e}. Assigning default cluster.")
                df['Cluster_Consensus'] = np.zeros(n_samples, dtype=int) # Fallback
    else:
        print("Not enough valid algorithms to perform consensus clustering.")
        df['Cluster_Consensus'] = np.zeros(n_samples, dtype=int) # Fallback

# --- 5. Detailed Patient Phenotype Profiles for Each Cluster ---
print("\n--- 5. Creating Detailed Patient Phenotype Profiles ---")

# Use Consensus clusters for phenotyping if available, otherwise K-Means as a fallback
phenotype_cluster_col = 'Cluster_Consensus' if 'Cluster_Consensus' in df.columns else 'Cluster_KMeans'
if phenotype_cluster_col not in df.columns:
    print("No valid cluster column found for phenotyping. Skipping phenotyping.")
else:
    print(f"Phenotyping based on '{phenotype_cluster_col}'...")
    clusters = sorted(df[phenotype_cluster_col].unique())
    print(f"Identified {len(clusters)} clusters for phenotyping: {clusters}")

    # Prepare data for statistical testing (original features + engineered)
    phenotype_features_for_analysis = features + ['Metabolic_Syndrome_Score', 'Potential_Metabolic_Syndrome']

    # Descriptive statistics for each cluster
    cluster_profiles = df.groupby(phenotype_cluster_col)[phenotype_features_for_analysis].agg(['mean', 'std', 'count']).T
    print("\nCluster Phenotype Profiles (Mean, Std, Count):")
    print(cluster_profiles)

    # Statistical Significance Testing
    print("\nStatistical Significance Testing (ANOVA/Kruskal-Wallis for continuous, Chi-squared for categorical):")
    print("Comparing each feature across clusters.")

    # Continuous features (using Kruskal-Wallis for non-parametric comparison)
    for feature in phenotype_features_for_analysis:
        groups = [df[df[phenotype_cluster_col] == c][feature].dropna() for c in clusters]
        # Filter out groups with insufficient data for testing
        valid_groups = [g for g in groups if len(g) > 1]
        if len(valid_groups) > 1: # Need at least 2 valid groups for comparison
            stat, p_val = kruskal(*valid_groups)
            print(f"  Feature '{feature}': Kruskal-Wallis H={stat:.2f}, p={p_val:.3f} {'(Significant)' if p_val < 0.05 else ''}")
        else:
            print(f"  Feature '{feature}': Not enough valid groups or samples for statistical test.")

    # Categorical features (e.g., 'Outcome', 'Potential_Metabolic_Syndrome')
    categorical_features_for_test = [target, 'Potential_Metabolic_Syndrome']
    for cat_feature in categorical_features_for_test:
        if cat_feature in df.columns:
            contingency_table = pd.crosstab(df[phenotype_cluster_col], df[cat_feature])
            # Ensure contingency table is not empty and has enough dimensions for chi-squared
            if contingency_table.shape[0] > 1 and contingency_table.shape[1] > 1 and contingency_table.sum().sum() > 0:
                try:
                    chi2, p_val, _, _ = chi2_contingency(contingency_table)
                    print(f"  Feature '{cat_feature}': Chi-squared={chi2:.2f}, p={p_val:.3f} {'(Significant)' if p_val < 0.05 else ''}")
                except ValueError as e:
                    print(f"  Feature '{cat_feature}': Chi-squared test failed: {e}. (Likely due to zero counts in some cells or insufficient data)")
            else:
                print(f"  Feature '{cat_feature}': Not enough categories or clusters for Chi-squared test.")

    # Visualize cluster profiles (e.g., box plots for key features)
    print("\nVisualizing Cluster Profiles for key features:")
    key_features_for_viz = ['Glucose', 'BMI', 'Age', 'BloodPressure', 'Metabolic_Syndrome_Score', target]
    for feature in key_features_for_viz:
        if feature in df.columns:
            plt.figure(figsize=(8, 5))
            sns.boxplot(x=phenotype_cluster_col, y=feature, data=df)
            plt.title(f'{feature} Distribution Across Clusters')
            plt.xlabel('Cluster')
            plt.ylabel(feature)
            plt.show()

# --- 6. Anomaly Detection ---
print("\n--- 6. Anomaly Detection ---")

# Initialize a DataFrame to store anomaly scores/flags
anomaly_scores_df = pd.DataFrame(index=df.index)

# Isolation Forest
print("Running Isolation Forest...")
# contamination: The proportion of outliers in the dataset. 'auto' tries to estimate.
iso_forest = IsolationForest(random_state=RANDOM_STATE, contamination='auto')
anomaly_scores_df['IsolationForest_Score'] = iso_forest.fit_predict(X_scaled) # -1 for outliers, 1 for inliers
anomaly_scores_df['IsolationForest_Anomaly'] = (anomaly_scores_df['IsolationForest_Score'] == -1).astype(int)
print(f"Isolation Forest detected {anomaly_scores_df['IsolationForest_Anomaly'].sum()} anomalies.")

# One-Class SVM
print("Running One-Class SVM...")
# nu: An upper bound on the fraction of training errors and a lower bound of the fraction of support vectors.
# gamma: Kernel coefficient. 'auto' uses 1 / n_features.
oc_svm = OneClassSVM(gamma='auto', nu=0.01) # Assuming ~1% outliers
anomaly_scores_df['OneClassSVM_Score'] = oc_svm.fit_predict(X_scaled) # -1 for outliers, 1 for inliers
anomaly_scores_df['OneClassSVM_Anomaly'] = (anomaly_scores_df['OneClassSVM_Score'] == -1).astype(int)
print(f"One-Class SVM detected {anomaly_scores_df['OneClassSVM_Anomaly'].sum()} anomalies.")

# Local Outlier Factor (LOF)
print("Running Local Outlier Factor (LOF)...")
# n_neighbors: Number of neighbors to consider for the local density.
lof = LocalOutlierFactor(n_neighbors=20, contamination='auto')
# fit_predict returns -1 for outliers, 1 for inliers.
anomaly_scores_df['LOF_Score'] = lof.fit_predict(X_scaled)
anomaly_scores_df['LOF_Anomaly'] = (anomaly_scores_df['LOF_Score'] == -1).astype(int)
print(f"LOF detected {anomaly_scores_df['LOF_Anomaly'].sum()} anomalies.")

# Add anomaly flags to the main DataFrame
df = pd.concat([df, anomaly_scores_df], axis=1)

# Visualize anomalies on dimensionality reduced plots
print("\nVisualizing Anomalies on Dimensionality Reduced Plots:")
plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=df['IsolationForest_Anomaly'], palette='coolwarm', s=50, alpha=0.7)
plt.title('t-SNE with Isolation Forest Anomalies')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')

plt.subplot(1, 3, 2)
sns.scatterplot(x=X_umap[:, 0], y=X_umap[:, 1], hue=df['OneClassSVM_Anomaly'], palette='coolwarm', s=50, alpha=0.7)
plt.title('UMAP with One-Class SVM Anomalies')
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')

plt.subplot(1, 3, 3)
sns.scatterplot(x=X_ae[:, 0], y=X_ae[:, 1], hue=df['LOF_Anomaly'], palette='coolwarm', s=50, alpha=0.7)
plt.title('Autoencoder Embeddings with LOF Anomalies')
plt.xlabel('AE 1')
plt.ylabel('AE 2')

plt.tight_layout()
plt.show()

# --- 7. Time-Series Clustering (Conceptual) ---
print("\n--- 7. Time-Series Clustering (Conceptual) ---")
print("The 'diabetes.csv' dataset (Pima Indian Diabetes) is cross-sectional, meaning it contains a single snapshot of data per patient.")
print("Therefore, direct time-series clustering cannot be applied to this dataset as it lacks temporal patterns.")
print("However, if temporal patterns existed (e.g., patient health records collected over multiple visits),")
print("the approach would typically involve:")
print("  a. Data Preparation: Reshaping data to have a time dimension for each patient (e.g., a list of measurements over time).")
print("  b. Feature Engineering: Extracting time-series specific features (e.g., trends, seasonality, rate of change, volatility).")
print("  c. Distance Metrics: Using time-series specific distance metrics like Dynamic Time Warping (DTW) to compare sequences of varying lengths or phases.")
print("  d. Clustering Algorithms: Applying algorithms like K-Means with DTW as the distance metric, Hierarchical Clustering with DTW, or specialized time-series clustering algorithms (e.g., from the `tslearn` library).")
print("  e. Visualization: Plotting representative time series for each cluster to understand temporal phenotypes.")
print("This section serves as a conceptual outline to address the requirement for time-series clustering.")

# --- 8. Association Rule Mining ---
print("\n--- 8. Association Rule Mining ---")

# Create a copy of the DataFrame for association rule mining
df_arm = df.copy()

# Discretize continuous features into bins for association rule mining
# Use quantiles to create bins, which helps handle skewed distributions
arm_features_to_bin = features + ['Metabolic_Syndrome_Score', 'Age'] # Select relevant features for ARM

for col in arm_features_to_bin:
    if pd.api.types.is_numeric_dtype(df_arm[col]):
        try:
            # Use qcut for quantile-based discretization, handling potential duplicates
            df_arm[col + '_bin'] = pd.qcut(df_arm[col], q=4, labels=[f'{col}_Q1', f'{col}_Q2', f'{col}_Q3', f'{col}_Q4'], duplicates='drop')
        except Exception as e:
            # Fallback to cut if qcut fails (e.g., too many identical values leading to non-unique bin edges)
            df_arm[col + '_bin'] = pd.cut(df_arm[col], bins=4, labels=[f'{col}_B1', f'{col}_B2', f'{col}_B3', f'{col}_B4'], include_lowest=True)
            print(f"Warning: qcut failed for {col}, using cut instead. Error: {e}")

# Include original categorical features and engineered binary features as is
df_arm['Outcome_bin'] = df_arm[target].map({0: 'No_Diabetes', 1: 'Diabetes'})
df_arm['Potential_Metabolic_Syndrome_bin'] = df_arm['Potential_Metabolic_Syndrome'].map({0: 'No_Metabolic_Syndrome', 1: 'Metabolic_Syndrome'})

# Select columns for ARM (binned features + target/engineered binary)
arm_cols = [col for col in df_arm.columns if '_bin' in col]
df_arm_selected = df_arm[arm_cols]

# Convert to one-hot encoded format for Apriori algorithm
df_arm_ohe = pd.get_dummies(df_arm_selected.astype(str), prefix='', prefix_sep='')

# Apply Apriori algorithm to find frequent itemsets
min_support = 0.05 # Minimum support threshold (e.g., 5% of transactions)
frequent_itemsets = apriori(df_arm_ohe, min_support=min_support, use_colnames=True)
print(f"Found {len(frequent_itemsets)} frequent itemsets with min_support={min_support}.")

# Generate association rules from frequent itemsets
min_confidence = 0.7 # Minimum confidence threshold (e.g., 70% confidence)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
rules = rules.sort_values(by='confidence', ascending=False)

print(f"\nGenerated {len(rules)} association rules with min_confidence={min_confidence}.")
print("Top 10 Association Rules (sorted by confidence):")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10))

# Example: Filter rules leading to 'Diabetes' outcome
diabetes_rules = rules[rules['consequents'].apply(lambda x: 'Diabetes' in str(x))]
print(f"\nTop 5 rules leading to 'Diabetes' outcome:")
print(diabetes_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(5))

# --- 9. Patient Risk Stratification System based on Clustering Results ---
print("\n--- 9. Building Patient Risk Stratification System ---")

# Use the consensus clusters (or K-Means if consensus not available) as the basis for risk groups.
# The phenotyping step provides the characteristics of each cluster.
risk_strat_cluster_col = phenotype_cluster_col # Use the same cluster column as for phenotyping

if risk_strat_cluster_col not in df.columns:
    print("No valid cluster column found for risk stratification. Skipping risk stratification.")
else:
    print(f"Stratifying patients based on '{risk_strat_cluster_col}' clusters.")
    # Summarize key risk indicators for each cluster
    cluster_risk_summary = df.groupby(risk_strat_cluster_col).agg(
        Diabetes_Prevalence=('Outcome', 'mean'), # Mean of Outcome (0/1) gives prevalence
        Avg_Glucose=('Glucose', 'mean'),
        Avg_BMI=('BMI', 'mean'),
        Avg_BP=('BloodPressure', 'mean'),
        Avg_Metabolic_Score=('Metabolic_Syndrome_Score', 'mean'),
        Num_Patients=('Age', 'count') # Count of patients in each cluster
    ).sort_values(by='Diabetes_Prevalence', ascending=False) # Sort by diabetes prevalence

    print("\nCluster Risk Summary:")
    print(cluster_risk_summary)

    # Assign qualitative risk levels (e.g., Low, Medium, High) based on the summary.
    # This is a heuristic and would typically involve clinical expert input for precise definitions.
    sorted_clusters_by_risk = cluster_risk_summary.index.tolist()
    num_clusters = len(sorted_clusters_by_risk)

    risk_mapping = {}
    if num_clusters == 1:
        risk_mapping[sorted_clusters_by_risk[0]] = 'Low Risk'
    elif num_clusters == 2:
        risk_mapping[sorted_clusters_by_risk[0]] = 'High Risk'
        risk_mapping[sorted_clusters_by_risk[1]] = 'Low Risk'
    elif num_clusters >= 3:
        # Assign risk levels based on tertiles or simple division of sorted clusters
        for i, cluster_id in enumerate(sorted_clusters_by_risk):
            if i < num_clusters / 3:
                risk_mapping[cluster_id] = 'High Risk'
            elif i < 2 * num_clusters / 3:
                risk_mapping[cluster_id] = 'Medium Risk'
            else:
                risk_mapping[cluster_id] = 'Low Risk'

    df['Risk_Level'] = df[risk_strat_cluster_col].map(risk_mapping)

    print("\nPatient Risk Stratification (Example Mapping):")
    print(df[['Age', 'Glucose', 'BMI', 'Outcome', risk_strat_cluster_col, 'Risk_Level']].head())
    print("\nDistribution of Patients by Risk Level:")
    print(df['Risk_Level'].value_counts())

    # Clinical Interpretation Example:
    print("\n--- Clinical Interpretation of Risk Stratification ---")
    for cluster_id in sorted_clusters_by_risk:
        risk_level = risk_mapping.get(cluster_id, 'Unknown Risk')
        cluster_data = cluster_risk_summary.loc[cluster_id]
        print(f"\nCluster {cluster_id} (Assigned Risk Level: {risk_level}):")
        print(f"  - Diabetes Prevalence: {cluster_data['Diabetes_Prevalence']:.2%}")
        print(f"  - Average Glucose: {cluster_data['Avg_Glucose']:.1f} mg/dL")
        print(f"  - Average BMI: {cluster_data['Avg_BMI']:.1f} kg/m^2")
        print(f"  - Average Blood Pressure: {cluster_data['Avg_BP']:.1f} mmHg")
        print(f"  - Average Metabolic Syndrome Score: {cluster_data['Avg_Metabolic_Score']:.1f} (out of 3)")
        print(f"  - Number of Patients: {int(cluster_data['Num_Patients'])}")
        print(f"  Clinical Implication: Patients in this cluster exhibit characteristics consistent with {risk_level.lower()} for diabetes and related metabolic conditions. For example, 'High Risk' clusters typically show higher average glucose, BMI, and blood pressure, along with higher diabetes prevalence. Targeted interventions (e.g., intensive lifestyle modification, medication review) could be designed based on their specific phenotype and risk level.")

print("\n--- Advanced Clustering and Patient Segmentation Analysis Complete ---")
```