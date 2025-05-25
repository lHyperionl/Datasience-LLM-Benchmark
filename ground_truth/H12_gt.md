```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.manifold import TSNE
import umap.umap_ as umap
from scipy.stats import f_oneway, chi2_contingency
from sklearn.ensemble import IsolationForest
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# --- 1. Load and Preprocess the Dataset ---
print("--- 1. Loading and Preprocessing Data ---")
try:
    df = pd.read_csv('video_games_sales.csv')
except FileNotFoundError:
    print("Error: video_games_sales.csv not found. Please ensure the file is in the current directory.")
    exit()

# Initial data cleaning and handling missing values
# Drop rows where 'Name', 'Genre', 'Platform', 'Publisher' are missing as they are crucial identifiers/categories
df.dropna(subset=['Name', 'Genre', 'Platform', 'Publisher'], inplace=True)

# For 'Year_of_Release', impute with the mode or median, or drop if too many missing.
# Given it's a year, mode might be more appropriate for categorical-like distribution.
# Or, for simplicity, drop rows with missing year if not too many.
# Let's drop for now to keep the dataset clean for numerical operations.
df.dropna(subset=['Year_of_Release'], inplace=True)
df['Year_of_Release'] = df['Year_of_Release'].astype(int)

# Fill missing sales values with 0, assuming NaN means no sales in that region
sales_cols = ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales']
for col in sales_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

# Create new features

# 'Regional_Sales_Distribution'
# Handle cases where Global_Sales is 0 to avoid division by zero
df['NA_Sales_Prop'] = df.apply(lambda row: row['NA_Sales'] / row['Global_Sales'] if row['Global_Sales'] > 0 else 0, axis=1)
df['EU_Sales_Prop'] = df.apply(lambda row: row['EU_Sales'] / row['Global_Sales'] if row['Global_Sales'] > 0 else 0, axis=1)
df['JP_Sales_Prop'] = df.apply(lambda row: row['JP_Sales'] / row['Global_Sales'] if row['Global_Sales'] > 0 else 0, axis=1)
df['Other_Sales_Prop'] = df.apply(lambda row: row['Other_Sales'] / row['Global_Sales'] if row['Global_Sales'] > 0 else 0, axis=1)

# 'Market_Position' (rank within genre by Global_Sales)
df['Market_Position'] = df.groupby('Genre')['Global_Sales'].rank(method='dense', ascending=False)

# 'Publisher_Portfolio_Size'
publisher_portfolio = df.groupby('Publisher')['Name'].nunique().reset_index()
publisher_portfolio.rename(columns={'Name': 'Publisher_Portfolio_Size'}, inplace=True)
df = pd.merge(df, publisher_portfolio, on='Publisher', how='left')

# Select features for clustering
numerical_features = ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales',
                      'NA_Sales_Prop', 'EU_Sales_Prop', 'JP_Sales_Prop', 'Other_Sales_Prop',
                      'Market_Position', 'Publisher_Portfolio_Size', 'Year_of_Release']
categorical_features = ['Platform', 'Genre', 'Publisher'] # 'Name' is too granular for clustering directly

# Create a copy of the original dataframe for later profiling
df_original = df.copy()

# Preprocessing Pipeline
# Identify numerical and categorical columns for transformation
numerical_cols = [col for col in numerical_features if col in df.columns]
categorical_cols = [col for col in categorical_features if col in df.columns]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

# Fit and transform the data
X_processed = preprocessor.fit_transform(df)

# Get feature names after one-hot encoding
ohe_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols)
all_feature_names = numerical_cols + list(ohe_feature_names)

X_processed_df = pd.DataFrame(X_processed, columns=all_feature_names, index=df.index)

print(f"Processed data shape: {X_processed_df.shape}")

# --- 2. Implement Multiple Clustering Algorithms & 4. Optimal Cluster Determination ---

# Determine optimal number of clusters (k) for K-Means, GMM, Hierarchical
# Using Elbow Method and Silhouette Score

max_clusters = 10 # Define a reasonable maximum number of clusters to test
inertias = []
silhouette_scores_kmeans = []
silhouette_scores_gmm = []
silhouette_scores_agg = []
k_range = range(2, max_clusters + 1)

print("\n--- 4. Determining Optimal Number of Clusters ---")
print("Running Elbow Method and Silhouette Analysis...")

for k in k_range:
    # K-Means
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(X_processed)
    inertias.append(kmeans.inertia_)
    silhouette_scores_kmeans.append(silhouette_score(X_processed, kmeans_labels))

    # Gaussian Mixture Models
    gmm = GaussianMixture(n_components=k, random_state=42)
    gmm_labels = gmm.fit_predict(X_processed)
    silhouette_scores_gmm.append(silhouette_score(X_processed, gmm_labels))

    # Hierarchical Clustering (Agglomerative)
    agg = AgglomerativeClustering(n_clusters=k)
    agg_labels = agg.fit_predict(X_processed)
    silhouette_scores_agg.append(silhouette_score(X_processed, agg_labels))

# Plot Elbow Method
plt.figure(figsize=(10, 6))
plt.plot(k_range, inertias, marker='o')
plt.title('Elbow Method for K-Means')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia (WCSS)')
plt.grid(True)
plt.show()

# Use KneeLocator to find the elbow point
kl = KneeLocator(k_range, inertias, S=1.0, curve="convex", direction="decreasing")
optimal_k_elbow = kl.elbow
print(f"Optimal k (Elbow Method): {optimal_k_elbow}")

# Plot Silhouette Scores
plt.figure(figsize=(12, 7))
plt.plot(k_range, silhouette_scores_kmeans, marker='o', label='K-Means Silhouette')
plt.plot(k_range, silhouette_scores_gmm, marker='o', label='GMM Silhouette')
plt.plot(k_range, silhouette_scores_agg, marker='o', label='Hierarchical Silhouette')
plt.title('Silhouette Scores for Different Clustering Algorithms')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.legend()
plt.grid(True)
plt.show()

# Determine optimal k based on highest silhouette score
optimal_k_kmeans_sil = k_range[np.argmax(silhouette_scores_kmeans)]
optimal_k_gmm_sil = k_range[np.argmax(silhouette_scores_gmm)]
optimal_k_agg_sil = k_range[np.argmax(silhouette_scores_agg)]

print(f"Optimal k (K-Means Silhouette): {optimal_k_kmeans_sil}")
print(f"Optimal k (GMM Silhouette): {optimal_k_gmm_sil}")
print(f"Optimal k (Hierarchical Silhouette): {optimal_k_agg_sil}")

# For consistency in subsequent steps, let's choose a single optimal k.
# We'll use the optimal k from K-Means Silhouette as a general guideline,
# but acknowledge that different algorithms might prefer different k.
# For consensus clustering, we'll use the optimal k for each algorithm.
chosen_optimal_k = optimal_k_kmeans_sil # Or average/median of the optimal k's

print(f"\nChosen optimal k for general use: {chosen_optimal_k}")

# --- 2. Implement Multiple Clustering Algorithms (with chosen optimal k) ---
print("\n--- 2. Implementing Clustering Algorithms ---")

# K-Means
kmeans_model = KMeans(n_clusters=chosen_optimal_k, random_state=42, n_init=10)
df['KMeans_Cluster'] = kmeans_model.fit_predict(X_processed)
print(f"K-Means clustering completed with {chosen_optimal_k} clusters.")

# Gaussian Mixture Models
gmm_model = GaussianMixture(n_components=chosen_optimal_k, random_state=42)
df['GMM_Cluster'] = gmm_model.fit_predict(X_processed)
print(f"GMM clustering completed with {chosen_optimal_k} clusters.")

# Hierarchical Clustering
agg_model = AgglomerativeClustering(n_clusters=chosen_optimal_k)
df['Hierarchical_Cluster'] = agg_model.fit_predict(X_processed)
print(f"Hierarchical clustering completed with {chosen_optimal_k} clusters.")

# DBSCAN - Requires careful parameter tuning (eps, min_samples)
# A common approach for eps is to use the k-distance graph.
# For simplicity and to ensure it runs, we'll pick some reasonable parameters.
# min_samples is often 2 * dimensionality or log(n_samples).
# Let's estimate eps using NearestNeighbors
print("\nAttempting DBSCAN (parameter tuning is critical and often iterative)...")
try:
    neighbors = NearestNeighbors(n_neighbors=2 * X_processed.shape[1]) # k = 2 * num_features
    neighbors_fit = neighbors.fit(X_processed)
    distances, indices = neighbors_fit.kneighbors(X_processed)
    distances = np.sort(distances[:, -1], axis=0) # Sort distances to the k-th nearest neighbor
    
    # Plot k-distance graph to visually identify elbow
    plt.figure(figsize=(10, 6))
    plt.plot(distances)
    plt.title('K-distance Graph for DBSCAN')
    plt.xlabel('Points sorted by distance')
    plt.ylabel(f'{2 * X_processed.shape[1]}-th Nearest Neighbor Distance')
    plt.grid(True)
    plt.show()

    # A heuristic for eps: find the "elbow" in the k-distance graph.
    # This is often done visually. For automated, we can use KneeLocator again.
    # However, KneeLocator might not be robust for all k-distance graphs.
    # Let's pick a value based on visual inspection or a percentile.
    # For demonstration, let's pick a percentile of distances.
    dbscan_eps = np.percentile(distances, 1) # e.g., 1st percentile
    dbscan_min_samples = 2 * X_processed.shape[1] # A common heuristic

    print(f"DBSCAN parameters: eps={dbscan_eps:.4f}, min_samples={dbscan_min_samples}")

    dbscan_model = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
    df['DBSCAN_Cluster'] = dbscan_model.fit_predict(X_processed)
    print(f"DBSCAN clustering completed. Number of clusters found: {len(df['DBSCAN_Cluster'].unique()) - (1 if -1 in df['DBSCAN_Cluster'].unique() else 0)}")
    print(f"Number of noise points (-1 label): {np.sum(df['DBSCAN_Cluster'] == -1)}")

except Exception as e:
    print(f"DBSCAN failed or encountered an issue: {e}. Skipping DBSCAN for now.")
    df['DBSCAN_Cluster'] = -2 # Indicate it was skipped

# --- 3. Dimensionality Reduction and Visualization ---
print("\n--- 3. Dimensionality Reduction and Visualization ---")

# t-SNE
print("Applying t-SNE for dimensionality reduction...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000) # Adjust perplexity based on data size
X_tsne = tsne.fit_transform(X_processed)
df['TSNE_1'] = X_tsne[:, 0]
df['TSNE_2'] = X_tsne[:, 1]

# UMAP
print("Applying UMAP for dimensionality reduction...")
reducer = umap.UMAP(n_components=2, random_state=42)
X_umap = reducer.fit_transform(X_processed)
df['UMAP_1'] = X_umap[:, 0]
df['UMAP_2'] = X_umap[:, 1]

# Visualize clusters in reduced dimensions
cluster_cols = ['KMeans_Cluster', 'GMM_Cluster', 'Hierarchical_Cluster']
if 'DBSCAN_Cluster' in df.columns and -2 not in df['DBSCAN_Cluster'].unique():
    cluster_cols.append('DBSCAN_Cluster')

for cluster_type in cluster_cols:
    plt.figure(figsize=(14, 6))

    # t-SNE plot
    plt.subplot(1, 2, 1)
    sns.scatterplot(x='TSNE_1', y='TSNE_2', hue=cluster_type, data=df, palette='viridis', s=20, alpha=0.7)
    plt.title(f't-SNE Visualization of {cluster_type}')
    plt.xlabel('TSNE_1')
    plt.ylabel('TSNE_2')
    plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')

    # UMAP plot
    plt.subplot(1, 2, 2)
    sns.scatterplot(x='UMAP_1', y='UMAP_2', hue=cluster_type, data=df, palette='viridis', s=20, alpha=0.7)
    plt.title(f'UMAP Visualization of {cluster_type}')
    plt.xlabel('UMAP_1')
    plt.ylabel('UMAP_2')
    plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.show()

# --- 5. Perform Consensus Clustering ---
print("\n--- 5. Performing Consensus Clustering ---")

# Create a co-occurrence matrix based on multiple clustering results
# For each pair of data points, count how many times they are in the same cluster
n_samples = X_processed.shape[0]
co_occurrence_matrix = np.zeros((n_samples, n_samples))

# Use the cluster labels from the algorithms that successfully ran
active_cluster_labels = {}
if 'KMeans_Cluster' in df.columns: active_cluster_labels['KMeans'] = df['KMeans_Cluster'].values
if 'GMM_Cluster' in df.columns: active_cluster_labels['GMM'] = df['GMM_Cluster'].values
if 'Hierarchical_Cluster' in df.columns: active_cluster_labels['Hierarchical'] = df['Hierarchical_Cluster'].values
if 'DBSCAN_Cluster' in df.columns and -2 not in df['DBSCAN_Cluster'].unique():
    # DBSCAN can have -1 (noise). For consensus, we can treat noise as its own cluster or exclude.
    # Let's treat noise as a separate cluster for co-occurrence.
    active_cluster_labels['DBSCAN'] = df['DBSCAN_Cluster'].values

num_algorithms = len(active_cluster_labels)

if num_algorithms > 1:
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            match_count = 0
            for algo_name, labels in active_cluster_labels.items():
                # Only count if both points are not noise in DBSCAN
                if algo_name == 'DBSCAN' and (labels[i] == -1 or labels[j] == -1):
                    continue
                if labels[i] == labels[j]:
                    match_count += 1
            co_occurrence_matrix[i, j] = match_count
            co_occurrence_matrix[j, i] = match_count # Symmetric matrix

    # Normalize the co-occurrence matrix to get a similarity matrix (0 to 1)
    # Divide by the number of algorithms that contributed to the comparison
    # If DBSCAN noise points were excluded, this normalization needs to be careful.
    # A simpler approach is to just use the raw match count and cluster that.
    # Or, normalize by the total number of algorithms.
    co_occurrence_matrix /= num_algorithms
    np.fill_diagonal(co_occurrence_matrix, 1.0) # A point is always in the same cluster as itself

    # Apply Hierarchical Clustering on the similarity matrix
    # We need a distance matrix for AgglomerativeClustering, so convert similarity to distance
    distance_matrix = 1 - co_occurrence_matrix
    
    # Use precomputed distance matrix
    consensus_model = AgglomerativeClustering(n_clusters=chosen_optimal_k, linkage='average', affinity='precomputed')
    df['Consensus_Cluster'] = consensus_model.fit_predict(distance_matrix)
    print(f"Consensus clustering completed with {chosen_optimal_k} clusters.")
else:
    print("Not enough clustering algorithms ran successfully for consensus clustering.")
    df['Consensus_Cluster'] = df['KMeans_Cluster'] # Fallback to KMeans if only one ran

# --- 6. Create Detailed Cluster Profiles with Statistical Significance Testing ---
print("\n--- 6. Creating Detailed Cluster Profiles ---")

# Use the original dataframe with cluster labels for profiling
df_profile = df_original.copy()
df_profile['Consensus_Cluster'] = df['Consensus_Cluster'] # Add consensus cluster labels

# Define features for profiling (original, unscaled features)
profile_numerical_features = ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales',
                              'Year_of_Release', 'Market_Position', 'Publisher_Portfolio_Size']
profile_categorical_features = ['Platform', 'Genre', 'Publisher']

# Profile each cluster
cluster_groups = df_profile.groupby('Consensus_Cluster')

print("\n--- Cluster Profiles (Mean/Mode) ---")
for cluster_id in sorted(df_profile['Consensus_Cluster'].unique()):
    print(f"\n--- Cluster {cluster_id} ---")
    cluster_data = cluster_groups.get_group(cluster_id)

    print("Numerical Features (Mean):")
    for col in profile_numerical_features:
        if col in cluster_data.columns:
            print(f"  {col}: {cluster_data[col].mean():.2f}")

    print("\nCategorical Features (Top 3 Modes):")
    for col in profile_categorical_features:
        if col in cluster_data.columns:
            top_modes = cluster_data[col].value_counts(normalize=True).head(3)
            print(f"  {col}:")
            for val, prop in top_modes.items():
                print(f"    - {val} ({prop:.2%})")

# Statistical Significance Testing
print("\n--- Statistical Significance Testing (Consensus Clusters vs. Overall) ---")

# Numerical features (ANOVA)
print("\nANOVA for Numerical Features:")
for col in profile_numerical_features:
    if col in df_profile.columns:
        # Create a list of arrays, one for each cluster's data for the current column
        cluster_data_for_anova = [cluster_groups.get_group(c_id)[col].dropna() for c_id in sorted(df_profile['Consensus_Cluster'].unique())]
        if all(len(arr) > 1 for arr in cluster_data_for_anova): # Ensure at least 2 data points per group
            f_stat, p_val = f_oneway(*cluster_data_for_anova)
            print(f"  {col}: F-statistic={f_stat:.2f}, p-value={p_val:.3f} {'(Significant)' if p_val < 0.05 else '(Not Significant)'}")
        else:
            print(f"  {col}: Not enough data in all clusters for ANOVA.")

# Categorical features (Chi-squared)
print("\nChi-squared Test for Categorical Features:")
for col in profile_categorical_features:
    if col in df_profile.columns:
        # Create a contingency table
        contingency_table = pd.crosstab(df_profile['Consensus_Cluster'], df_profile[col])
        if contingency_table.shape[0] > 1 and contingency_table.shape[1] > 1: # Ensure at least 2 rows/cols
            chi2, p_val, _, _ = chi2_contingency(contingency_table)
            print(f"  {col}: Chi2-statistic={chi2:.2f}, p-value={p_val:.3f} {'(Significant)' if p_val < 0.05 else '(Not Significant)'}")
        else:
            print(f"  {col}: Not enough variation for Chi-squared test.")

# --- 7. Implement Anomaly Detection ---
print("\n--- 7. Implementing Anomaly Detection ---")

# Use Isolation Forest on the processed data
iso_forest = IsolationForest(random_state=42, contamination=0.01) # contamination is the expected proportion of outliers
df['Anomaly_Score'] = iso_forest.fit_predict(X_processed)

# Anomaly_Score: -1 for outliers, 1 for inliers
num_anomalies = (df['Anomaly_Score'] == -1).sum()
print(f"Number of detected anomalies: {num_anomalies}")

# Display some anomalous games
print("\nTop 5 Anomalous Games:")
anomalous_games = df[df['Anomaly_Score'] == -1].sort_values(by='Anomaly_Score', ascending=True).head(5)
print(df_original.loc[anomalous_games.index][['Name', 'Platform', 'Genre', 'Publisher', 'Global_Sales']])

# --- 8. Use Association Rule Mining ---
print("\n--- 8. Performing Association Rule Mining ---")

# Prepare data for association rule mining
# We'll use 'Genre', 'Platform', 'Publisher', and the 'Consensus_Cluster' as items
# Convert these columns into a transactional format (one-hot encoded for mlxtend)
# Limit to a subset of features to avoid too sparse a matrix
arm_features = ['Genre', 'Platform', 'Consensus_Cluster']

# Create a temporary dataframe for ARM
df_arm = df_original.copy()
df_arm['Consensus_Cluster'] = df['Consensus_Cluster'].astype(str) # Convert cluster to string for OHE

# One-hot encode selected features for ARM
arm_ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
arm_processed = arm_ohe.fit_transform(df_arm[arm_features])
arm_processed_df = pd.DataFrame(arm_processed, columns=arm_ohe.get_feature_names_out(arm_features))

# Apply Apriori algorithm
min_support = 0.01 # Adjust as needed
frequent_itemsets = apriori(arm_processed_df, min_support=min_support, use_colnames=True)
print(f"Found {len(frequent_itemsets)} frequent itemsets with min_support={min_support}")

# Generate association rules
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0) # min_threshold for lift
rules = rules.sort_values(by=['lift', 'confidence'], ascending=[False, False])

print("\nTop 10 Association Rules (by Lift):")
print(rules.head(10))

# --- 9. Build a Recommendation System based on Clustering Results ---
print("\n--- 9. Building a Recommendation System ---")

# Simple content-based recommendation: recommend games from the same cluster
def recommend_games_by_cluster(game_name, df_data, num_recommendations=5, cluster_col='Consensus_Cluster'):
    game_info = df_data[df_data['Name'] == game_name]
    if game_info.empty:
        print(f"Game '{game_name}' not found in the dataset.")
        return pd.DataFrame()

    # If multiple entries for the same game name, pick the first one
    game_cluster = game_info.iloc[0][cluster_col]
    
    # Find other games in the same cluster, excluding the input game itself
    recommendations = df_data[
        (df_data[cluster_col] == game_cluster) & 
        (df_data['Name'] != game_name)
    ]
    
    # Sort by Global_Sales to recommend popular games within the cluster
    recommendations = recommendations.sort_values(by='Global_Sales', ascending=False)
    
    return recommendations[['Name', 'Platform', 'Genre', 'Publisher', 'Global_Sales']].head(num_recommendations)

# Example usage of the recommendation system
example_game = "Super Mario Bros."
print(f"\nRecommendations for '{example_game}':")
recs = recommend_games_by_cluster(example_game, df_original.assign(Consensus_Cluster=df['Consensus_Cluster']))
if not recs.empty:
    print(recs)
else:
    print("No recommendations found.")

example_game_2 = "Grand Theft Auto V"
print(f"\nRecommendations for '{example_game_2}':")
recs_2 = recommend_games_by_cluster(example_game_2, df_original.assign(Consensus_Cluster=df['Consensus_Cluster']))
if not recs_2.empty:
    print(recs_2)
else:
    print("No recommendations found.")

# --- 10. Validate Clustering Results ---
print("\n--- 10. Validating Clustering Results (Internal Metrics) ---")

# Internal Validation Metrics
# Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Index
# These require at least 2 clusters and more than 1 sample per cluster.

validation_results = {}

cluster_algorithms_to_validate = {
    'KMeans': df['KMeans_Cluster'],
    'GMM': df['GMM_Cluster'],
    'Hierarchical': df['Hierarchical_Cluster'],
    'Consensus': df['Consensus_Cluster']
}
if 'DBSCAN_Cluster' in df.columns and -2 not in df['DBSCAN_Cluster'].unique():
    cluster_algorithms_to_validate['DBSCAN'] = df['DBSCAN_Cluster']

for algo_name, labels in cluster_algorithms_to_validate.items():
    # Filter out noise points for DBSCAN if present, as they are not part of a cluster
    if algo_name == 'DBSCAN':
        valid_indices = labels != -1
        X_valid = X_processed[valid_indices]
        labels_valid = labels[valid_indices]
    else:
        X_valid = X_processed
        labels_valid = labels

    n_clusters_found = len(np.unique(labels_valid))
    
    if n_clusters_found > 1 and len(labels_valid) > n_clusters_found:
        try:
            sil_score = silhouette_score(X_valid, labels_valid)
            db_index = davies_bouldin_score(X_valid, labels_valid)
            ch_index = calinski_harabasz_score(X_valid, labels_valid)
            validation_results[algo_name] = {
                'Silhouette Score': sil_score,
                'Davies-Bouldin Index': db_index,
                'Calinski-Harabasz Index': ch_index
            }
        except Exception as e:
            validation_results[algo_name] = f"Error calculating metrics: {e}"
    else:
        validation_results[algo_name] = "Not enough clusters or samples for validation metrics."

for algo, metrics in validation_results.items():
    print(f"\n--- {algo} Validation ---")
    if isinstance(metrics, dict):
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.4f}")
    else:
        print(f"  {metrics}")

print("\n--- Analysis Complete ---")
```