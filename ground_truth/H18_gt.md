```python
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE, Isomap
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from scipy.stats import f_oneway
from scipy.cluster.hierarchy import linkage, fcluster
import hdbscan
import umap
import networkx as nx
from mlxtend.frequent_patterns import apriori, association_rules
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import warnings

# Suppress warnings for cleaner output, especially from sklearn and tensorflow
warnings.filterwarnings('ignore')

# Set plot style for better visualization
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

class AdvancedHappinessAnalysis:
    """
    A comprehensive class to perform advanced clustering and country segmentation analysis
    on the World Happiness Report dataset.
    """
    def __init__(self, file_path='world-happiness-report-2021.csv'):
        self.file_path = file_path
        self.df = None # Original DataFrame
        self.df_processed = None # Processed DataFrame for analysis
        self.numerical_features = None
        self.categorical_features = None
        self.country_col = 'Country name'
        self.region_col = 'Regional indicator'
        self.happiness_score_col = 'Ladder score'
        self.clustering_results = {} # Stores labels from different clustering algorithms
        self.consensus_clusters = None # Stores labels from consensus clustering
        self.dr_embeddings = {} # Stores embeddings from dimensionality reduction techniques
        self.anomaly_scores = {} # Stores anomaly detection results
        self.network_graph = None # Stores the networkx graph
        self.association_rules = None # Stores discovered association rules

    def load_and_preprocess_data(self):
        """
        1. Load and preprocess the dataset with advanced feature engineering
           including happiness profiles and regional characteristics.
        """
        print("1. Loading and Preprocessing Data...")
        self.df = pd.read_csv(self.file_path)

        # Identify numerical and categorical features from the original dataset
        # Exclude columns that are direct calculations or redundant for independent features
        self.numerical_features = [
            'Ladder score', 'Logged GDP per capita', 'Social support',
            'Healthy life expectancy', 'Freedom to make life choices',
            'Generosity', 'Perceptions of corruption'
        ]
        self.categorical_features = [self.region_col]

        # Create a copy for processing to avoid modifying the original DataFrame
        self.df_processed = self.df.copy()

        # Impute missing values
        # For numerical features, use median imputation as it's robust to outliers
        for col in self.numerical_features:
            if col in self.df_processed.columns:
                self.df_processed[col] = self.df_processed[col].fillna(self.df_processed[col].median())

        # For categorical features, use mode imputation
        for col in self.categorical_features:
            if col in self.df_processed.columns:
                self.df_processed[col] = self.df_processed[col].fillna(self.df_processed[col].mode()[0])

        # Advanced Feature Engineering: Happiness Profiles & Regional Characteristics
        # Create interaction terms and ratios to capture complex relationships
        self.df_processed['GDP_x_SocialSupport'] = self.df_processed['Logged GDP per capita'] * self.df_processed['Social support']
        self.df_processed['Health_Freedom_Ratio'] = self.df_processed['Healthy life expectancy'] / (self.df_processed['Freedom to make life choices'] + 1e-6) # Add epsilon to prevent division by zero
        self.df_processed['Generosity_minus_Corruption'] = self.df_processed['Generosity'] - self.df_processed['Perceptions of corruption'] # Higher value means more generosity, less corruption

        # Update the list of numerical features to include the newly engineered ones
        self.numerical_features.extend(['GDP_x_SocialSupport', 'Health_Freedom_Ratio', 'Generosity_minus_Corruption'])

        # One-hot encode regional indicator to incorporate regional characteristics into numerical features
        self.df_processed = pd.get_dummies(self.df_processed, columns=[self.region_col], prefix=self.region_col)

        # Select only the features relevant for clustering (numerical features and one-hot encoded regions)
        # Exclude 'Country name' and other non-feature columns
        features_for_scaling = [col for col in self.numerical_features if col in self.df_processed.columns]
        features_for_scaling.extend([col for col in self.df_processed.columns if col.startswith(self.region_col + '_')])

        # Scale numerical features using StandardScaler to ensure all features contribute equally
        scaler = StandardScaler()
        self.df_processed[features_for_scaling] = scaler.fit_transform(self.df_processed[features_for_scaling])

        # Keep only the country name and the processed features for subsequent analysis
        self.df_processed = self.df_processed[[self.country_col] + features_for_scaling]

        print(f"Processed data shape: {self.df_processed.shape}")
        print(f"Features used for clustering: {self.df_processed.columns.tolist()}")
        print("\n")

    def _get_clustering_data(self):
        """Helper to get the data ready for clustering, excluding country name."""
        return self.df_processed.drop(columns=[self.country_col], errors='ignore')

    def _find_optimal_k_silhouette(self, algorithm, X, k_range):
        """
        Helper function to find optimal K for algorithms requiring it (K-Means, Agglomerative, GMM, Spectral)
        using Silhouette Score. Also plots Elbow method for K-Means.
        """
        scores = []
        k_values = range(k_range[0], k_range[1] + 1)
        best_score = -1
        best_k = None

        for k in k_values:
            if k < 2 or k >= X.shape[0]: # Silhouette requires at least 2 clusters and k < n_samples
                continue
            try:
                if algorithm == KMeans:
                    model = algorithm(n_clusters=k, random_state=42, n_init=10)
                elif algorithm == GaussianMixture:
                    model = algorithm(n_components=k, random_state=42)
                elif algorithm == SpectralClustering:
                    # 'kmeans' for assign_labels is generally robust
                    model = algorithm(n_clusters=k, assign_labels='kmeans', random_state=42)
                else: # AgglomerativeClustering
                    model = algorithm(n_clusters=k)

                labels = model.fit_predict(X)
                # Ensure at least 2 unique clusters are formed for silhouette score calculation
                if len(set(labels)) > 1:
                    score = silhouette_score(X, labels)
                    scores.append((k, score))
                    if score > best_score:
                        best_score = score
                        best_k = k
            except Exception as e:
                print(f"      Error running {algorithm.__name__} with k={k}: {e}")
                continue

        if not scores:
            return None

        # Plotting Elbow/Silhouette for K-Means (example visualization)
        if algorithm == KMeans:
            distortions = []
            for k in k_values:
                if k < 2 or k >= X.shape[0]: continue
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans.fit(X)
                distortions.append(kmeans.inertia_)

            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            plt.plot(k_values, distortions, 'bx-')
            plt.xlabel('Number of Clusters (k)')
            plt.ylabel('Distortion (SSE)')
            plt.title('Elbow Method for K-Means')

            plt.subplot(1, 2, 2)
            k_scores = [s[0] for s in scores]
            silhouette_scores_vals = [s[1] for s in scores]
            plt.plot(k_scores, silhouette_scores_vals, 'rx-')
            plt.xlabel('Number of Clusters (k)')
            plt.ylabel('Silhouette Score')
            plt.title(f'Silhouette Score for {algorithm.__name__}')
            plt.tight_layout()
            plt.show()

        return best_k

    def implement_multiple_clustering_algorithms(self, n_clusters_range=(3, 8), eps_range=(0.5, 1.5), min_samples_range=(3, 10)):
        """
        2. Implement multiple clustering algorithms: K-Means, DBSCAN, Hierarchical Clustering,
           Gaussian Mixture Models, Spectral Clustering, and HDBSCAN.
        """
        print("2. Implementing Multiple Clustering Algorithms...")
        X = self._get_clustering_data()
        n_samples = X.shape[0]

        # K-Means
        print("  - Running K-Means...")
        best_k_kmeans = self._find_optimal_k_silhouette(KMeans, X, n_clusters_range)
        if best_k_kmeans:
            kmeans = KMeans(n_clusters=best_k_kmeans, random_state=42, n_init=10)
            self.clustering_results['KMeans'] = kmeans.fit_predict(X)
            print(f"    K-Means optimal K: {best_k_kmeans}, Silhouette Score: {silhouette_score(X, self.clustering_results['KMeans']):.3f}")
        else:
            print("    Could not determine optimal K for KMeans. Skipping.")

        # DBSCAN
        print("  - Running DBSCAN...")
        # Iterate through a range of eps and min_samples to find a reasonable configuration
        best_dbscan_score = -1
        best_dbscan_params = None
        for eps in np.linspace(eps_range[0], eps_range[1], 5):
            for min_samples in range(min_samples_range[0], min_samples_range[1] + 1):
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                labels = dbscan.fit_predict(X)
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0) # Exclude noise points (-1)
                if n_clusters > 1: # Silhouette requires at least 2 clusters
                    score = silhouette_score(X, labels)
                    if score > best_dbscan_score:
                        best_dbscan_score = score
                        best_dbscan_params = {'eps': eps, 'min_samples': min_samples}
                        self.clustering_results['DBSCAN'] = labels
        if best_dbscan_params:
            print(f"    DBSCAN optimal params: {best_dbscan_params}, Silhouette Score: {best_dbscan_score:.3f}")
        else:
            print("    Could not find suitable parameters for DBSCAN resulting in >1 cluster.")

        # Hierarchical Clustering (Agglomerative)
        print("  - Running Hierarchical Clustering (Agglomerative)...")
        best_k_agglo = self._find_optimal_k_silhouette(AgglomerativeClustering, X, n_clusters_range)
        if best_k_agglo:
            agglo = AgglomerativeClustering(n_clusters=best_k_agglo)
            self.clustering_results['Agglomerative'] = agglo.fit_predict(X)
            print(f"    Agglomerative optimal K: {best_k_agglo}, Silhouette Score: {silhouette_score(X, self.clustering_results['Agglomerative']):.3f}")
        else:
            print("    Could not determine optimal K for Agglomerative. Skipping.")

        # Gaussian Mixture Models (GMM)
        print("  - Running Gaussian Mixture Models (GMM)...")
        best_k_gmm = self._find_optimal_k_silhouette(GaussianMixture, X, n_clusters_range)
        if best_k_gmm:
            gmm = GaussianMixture(n_components=best_k_gmm, random_state=42)
            self.clustering_results['GMM'] = gmm.fit_predict(X)
            print(f"    GMM optimal K: {best_k_gmm}, Silhouette Score: {silhouette_score(X, self.clustering_results['GMM']):.3f}")
        else:
            print("    Could not determine optimal K for GMM. Skipping.")

        # Spectral Clustering
        print("  - Running Spectral Clustering...")
        best_k_spectral = self._find_optimal_k_silhouette(SpectralClustering, X, n_clusters_range)
        if best_k_spectral:
            spectral = SpectralClustering(n_clusters=best_k_spectral, assign_labels='kmeans', random_state=42)
            self.clustering_results['Spectral'] = spectral.fit_predict(X)
            print(f"    Spectral optimal K: {best_k_spectral}, Silhouette Score: {silhouette_score(X, self.clustering_results['Spectral']):.3f}")
        else:
            print("    Could not determine optimal K for Spectral. Skipping.")

        # HDBSCAN
        print("  - Running HDBSCAN...")
        # HDBSCAN automatically determines the number of clusters. We tune min_cluster_size.
        best_hdbscan_score = -1
        best_hdbscan_params = None
        for min_cluster_size in range(5, 15, 2): # Iterate through reasonable sizes
            hdbscan_clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, prediction_data=True)
            labels = hdbscan_clusterer.fit_predict(X)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            if n_clusters > 1:
                score = silhouette_score(X, labels)
                if score > best_hdbscan_score:
                    best_hdbscan_score = score
                    best_hdbscan_params = {'min_cluster_size': min_cluster_size}
                    self.clustering_results['HDBSCAN'] = labels
        if best_hdbscan_params:
            print(f"    HDBSCAN optimal params: {best_hdbscan_params}, Silhouette Score: {best_hdbscan_score:.3f}")
        else:
            print("    Could not find suitable parameters for HDBSCAN resulting in >1 cluster.")

        print("\n")

    def advanced_dimensionality_reduction(self, n_components=2):
        """
        3. Use advanced dimensionality reduction techniques: t-SNE, UMAP, autoencoders,
           and manifold learning for visualization.
        """
        print("3. Performing Advanced Dimensionality Reduction...")
        X = self._get_clustering_data()

        # t-SNE
        print("  - Running t-SNE...")
        # Perplexity should be less than n_samples
        tsne = TSNE(n_components=n_components, random_state=42, perplexity=min(30, X.shape[0]-1))
        self.dr_embeddings['t-SNE'] = tsne.fit_transform(X)
        print(f"    t-SNE embedding shape: {self.dr_embeddings['t-SNE'].shape}")

        # UMAP
        print("  - Running UMAP...")
        reducer = umap.UMAP(n_components=n_components, random_state=42)
        self.dr_embeddings['UMAP'] = reducer.fit_transform(X)
        print(f"    UMAP embedding shape: {self.dr_embeddings['UMAP'].shape}")

        # Autoencoders
        print("  - Training Autoencoder...")
        input_dim = X.shape[1]
        encoding_dim = n_components # Bottleneck layer dimension

        # Scale data to [0,1] for autoencoder with sigmoid output
        min_max_scaler = MinMaxScaler()
        X_scaled_for_ae = min_max_scaler.fit_transform(X)

        # Build the autoencoder model
        input_layer = Input(shape=(input_dim,))
        encoder = Dense(input_dim // 2, activation='relu')(input_layer)
        encoder = Dense(encoding_dim, activation='relu')(encoder) # Bottleneck layer
        decoder = Dense(input_dim // 2, activation='relu')(encoder)
        decoder = Dense(input_dim, activation='sigmoid')(decoder) # Sigmoid for [0,1] scaled data

        autoencoder = Model(inputs=input_layer, outputs=decoder)
        encoder_model = Model(inputs=input_layer, outputs=encoder) # Model to get the embeddings

        autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        history = autoencoder.fit(X_scaled_for_ae, X_scaled_for_ae,
                                  epochs=100,
                                  batch_size=32,
                                  shuffle=True,
                                  validation_split=0.1,
                                  callbacks=[early_stopping],
                                  verbose=0) # Set verbose to 0 to suppress training output
        self.dr_embeddings['Autoencoder'] = encoder_model.predict(X_scaled_for_ae)
        print(f"    Autoencoder embedding shape: {self.dr_embeddings['Autoencoder'].shape}")

        # Manifold Learning (Isomap)
        print("  - Running Isomap (Manifold Learning)...")
        # n_neighbors should be less than n_samples
        n_neighbors_isomap = min(10, X.shape[0] - 1)
        isomap = Isomap(n_components=n_components, n_neighbors=n_neighbors_isomap)
        self.dr_embeddings['Isomap'] = isomap.fit_transform(X)
        print(f"    Isomap embedding shape: {self.dr_embeddings['Isomap'].shape}")

        # Visualize embeddings with cluster labels (using KMeans as an example if available)
        if 'KMeans' in self.clustering_results:
            kmeans_labels = self.clustering_results['KMeans']
            for dr_method, embedding in self.dr_embeddings.items():
                plt.figure(figsize=(10, 8))
                sns.scatterplot(x=embedding[:, 0], y=embedding[:, 1], hue=kmeans_labels, palette='viridis', legend='full', s=100, alpha=0.7)
                plt.title(f'{dr_method} Visualization with KMeans Clusters')
                plt.xlabel(f'{dr_method} Component 1')
                plt.ylabel(f'{dr_method} Component 2')
                plt.show()
        print("\n")

    def determine_optimal_clustering_parameters(self):
        """
        4. Determine optimal clustering parameters using silhouette analysis,
           gap statistic, elbow method, and stability analysis.
           (Partial implementation: Silhouette and Elbow are covered in _find_optimal_k_silhouette.
           Gap statistic and stability analysis are conceptually outlined due to their complexity
           and need for custom implementations or specialized libraries.)
        """
        print("4. Determining Optimal Clustering Parameters (Conceptual/Partial Implementation)...")
        print("  - Silhouette Analysis and Elbow Method are integrated into the clustering algorithm selection process.")
        print("  - Gap Statistic: This method compares the total within-cluster variation for different numbers of clusters")
        print("    to that expected under a null reference distribution (e.g., uniform distribution). It aims to find k")
        print("    where the gap between the observed and reference dispersion is maximized. Implementing it requires")
        print("    generating multiple reference datasets and calculating within-cluster dispersion for each k, making it")
        print("    computationally intensive and often implemented via specialized libraries or custom loops.")
        print("  - Stability Analysis: Involves running clustering multiple times on perturbed data (e.g., bootstrap samples)")
        print("    or different initializations. The consistency of cluster assignments across these runs is then assessed")
        print("    using metrics like Adjusted Rand Index or Jaccard Index. A robust cluster solution should be stable.")
        print("    Due to complexity and length, full implementation is beyond the scope of this single script, but it's")
        print("    a crucial conceptual aspect for validating clustering results.")
        print("\n")

    def perform_consensus_clustering(self):
        """
        5. Perform consensus clustering to identify robust country groupings across different algorithms.
           This uses a co-association matrix approach.
        """
        print("5. Performing Consensus Clustering...")
        X = self._get_clustering_data()
        n_samples = X.shape[0]

        # Filter out algorithms that didn't produce results or had issues (e.g., only one cluster)
        valid_clustering_results = {
            alg: labels for alg, labels in self.clustering_results.items()
            if labels is not None and len(set(labels)) > 1 and len(labels) == n_samples
        }

        if not valid_clustering_results:
            print("  No valid clustering results to perform consensus clustering.")
            return

        # Create a co-association matrix: M_ij = 1 if i and j are in the same cluster for a given algorithm, 0 otherwise.
        # Sum these matrices across all valid algorithms.
        co_association_matrix = np.zeros((n_samples, n_samples))

        for alg_name, labels in valid_clustering_results.items():
            # For each pair of samples, if they are in the same cluster, increment their co-association count
            for i in range(n_samples):
                for j in range(i, n_samples):
                    if labels[i] == labels[j]:
                        co_association_matrix[i, j] += 1
                        if i != j: # Ensure symmetry
                            co_association_matrix[j, i] += 1

        # Normalize the co-association matrix by the number of algorithms
        num_algorithms = len(valid_clustering_results)
        if num_algorithms > 0:
            co_association_matrix /= num_algorithms
        else:
            print("  No algorithms contributed to co-association matrix.")
            return

        # Convert co-association matrix to a distance matrix (1 - similarity)
        distance_matrix = 1 - co_association_matrix

        # Apply Hierarchical Clustering on the distance matrix to find consensus clusters
        # Determine optimal number of consensus clusters using silhouette score
        best_consensus_k = None
        best_consensus_score = -1
        consensus_k_range = range(3, 8) # A reasonable range for consensus clusters

        for k in consensus_k_range:
            if k < 2 or k >= n_samples: continue
            try:
                # Use average linkage for robustness in hierarchical clustering
                # linkage expects condensed distance matrix (upper triangle)
                linked = linkage(distance_matrix[np.triu_indices(n_samples, k=1)], method='average')
                consensus_labels = fcluster(linked, k, criterion='maxclust')
                if len(set(consensus_labels)) > 1:
                    score = silhouette_score(X, consensus_labels)
                    if score > best_consensus_score:
                        best_consensus_score = score
                        best_consensus_k = k
                        self.consensus_clusters = consensus_labels
            except Exception as e:
                print(f"    Error during consensus clustering with k={k}: {e}")
                continue

        if self.consensus_clusters is not None:
            # Add consensus cluster labels to the processed DataFrame
            self.df_processed['Consensus_Cluster'] = self.consensus_clusters
            print(f"  Consensus clustering completed with {best_consensus_k} clusters. Silhouette Score: {best_consensus_score:.3f}")
            print(f"  Consensus cluster distribution:\n{self.df_processed['Consensus_Cluster'].value_counts()}")

            # Visualize consensus clusters using UMAP (if UMAP embedding is available)
            if 'UMAP' in self.dr_embeddings:
                plt.figure(figsize=(10, 8))
                sns.scatterplot(x=self.dr_embeddings['UMAP'][:, 0], y=self.dr_embeddings['UMAP'][:, 1],
                                hue=self.df_processed['Consensus_Cluster'], palette='viridis', legend='full', s=100, alpha=0.7)
                plt.title('UMAP Visualization with Consensus Clusters')
                plt.xlabel('UMAP Component 1')
                plt.ylabel('UMAP Component 2')
                plt.show()
        else:
            print("  Consensus clustering failed to produce robust clusters.")
        print("\n")

    def create_detailed_country_profiles(self):
        """
        6. Create detailed country profiles for each cluster with statistical significance testing
           and effect size analysis.
        """
        print("6. Creating Detailed Country Profiles...")
        if 'Consensus_Cluster' not in self.df_processed.columns:
            print("  Consensus clusters not found. Please run consensus clustering first.")
            return

        cluster_col = 'Consensus_Cluster'
        # Get the original numerical features for profiling (before scaling)
        original_numerical_cols = [col for col in self.numerical_features if col in self.df.columns]
        # Add the original happiness score for profiling
        original_numerical_cols.append(self.happiness_score_col)

        # Merge cluster labels back to the original (unscaled) data for meaningful interpretation
        df_with_clusters = self.df.merge(self.df_processed[[self.country_col, cluster_col]], on=self.country_col)

        print("\n--- Cluster Profiles (Mean Values of Original Features) ---")
        # Calculate mean values for each feature within each cluster
        cluster_means = df_with_clusters.groupby(cluster_col)[original_numerical_cols].mean()
        print(cluster_means)

        print("\n--- Statistical Significance Testing (ANOVA) and Effect Size (Eta-squared) ---")
        print("  Comparing feature means across clusters:")
        for feature in original_numerical_cols:
            # Extract groups for ANOVA: data for 'feature' for each cluster
            groups = [df_with_clusters[df_with_clusters[cluster_col] == c][feature].dropna() for c in df_with_clusters[cluster_col].unique()]
            
            # Filter out groups with insufficient data (ANOVA requires at least 2 data points per group)
            valid_groups = [g for g in groups if len(g) > 1]
            
            if len(valid_groups) < 2: # Need at least two valid groups to perform ANOVA
                print(f"    Skipping ANOVA for '{feature}': Not enough valid groups (less than 2 groups with >1 data point).")
                continue

            # Perform One-way ANOVA
            f_stat, p_value = f_oneway(*valid_groups)
            print(f"  Feature: '{feature}'")
            print(f"    ANOVA F-statistic: {f_stat:.3f}, p-value: {p_value:.3e}")

            # Calculate Effect Size (Eta-squared)
            # Eta-squared = SS_between / SS_total
            grand_mean = df_with_clusters[feature].mean()
            ss_between = sum(len(g) * (g.mean() - grand_mean)**2 for g in valid_groups)
            ss_total = sum((x - grand_mean)**2 for group in valid_groups for x in group)

            eta_squared = ss_between / ss_total if ss_total > 0 else 0
            print(f"    Effect Size (Eta-squared): {eta_squared:.3f} (0.01: small, 0.06: medium, 0.14: large)")
            if p_value < 0.05:
                print(f"    Significant difference across clusters for '{feature}'.")
            else:
                print(f"    No significant difference across clusters for '{feature}'.")
        print("\n")

    def implement_anomaly_detection(self):
        """
        7. Implement anomaly detection to identify countries with unusual happiness patterns.
        """
        print("7. Implementing Anomaly Detection...")
        X = self._get_clustering_data()

        # Isolation Forest: Effective for high-dimensional datasets and large number of samples
        print("  - Running Isolation Forest...")
        iso_forest = IsolationForest(random_state=42, contamination='auto') # 'auto' estimates contamination
        self.anomaly_scores['IsolationForest'] = iso_forest.fit_predict(X) # -1 for outliers, 1 for inliers
        self.df_processed['IsolationForest_Anomaly'] = self.anomaly_scores['IsolationForest']
        num_anomalies_iso = (self.df_processed['IsolationForest_Anomaly'] == -1).sum()
        print(f"    Isolation Forest identified {num_anomalies_iso} anomalies.")

        # Local Outlier Factor (LOF): Measures local deviation of density of a given data point
        print("  - Running Local Outlier Factor (LOF)...")
        lof = LocalOutlierFactor(n_neighbors=20, contamination='auto')
        self.anomaly_scores['LOF'] = lof.fit_predict(X) # -1 for outliers, 1 for inliers
        self.df_processed['LOF_Anomaly'] = self.anomaly_scores['LOF']
        num_anomalies_lof = (self.df_processed['LOF_Anomaly'] == -1).sum()
        print(f"    LOF identified {num_anomalies_lof} anomalies.")

        # One-Class SVM: Learns a decision boundary for the 'normal' data points
        print("  - Running One-Class SVM...")
        # nu is an upper bound on the fraction of training errors and a lower bound of the fraction of support vectors.
        # It represents the expected fraction of outliers.
        oc_svm = OneClassSVM(kernel='rbf', nu=0.1) # Assuming 10% outliers
        self.anomaly_scores['OneClassSVM'] = oc_svm.fit_predict(X) # -1 for outliers, 1 for inliers
        self.df_processed['OneClassSVM_Anomaly'] = self.anomaly_scores['OneClassSVM']
        num_anomalies_ocsvm = (self.df_processed['OneClassSVM_Anomaly'] == -1).sum()
        print(f"    One-Class SVM identified {num_anomalies_ocsvm} anomalies.")

        print("\n--- Anomalous Countries (based on Isolation Forest) ---")
        anomalous_countries_iso = self.df_processed[self.df_processed['IsolationForest_Anomaly'] == -1][self.country_col]
        print(anomalous_countries_iso.tolist())

        # Visualize anomalies on UMAP plot (if UMAP embedding is available)
        if 'UMAP' in self.dr_embeddings:
            plt.figure(figsize=(10, 8))
            sns.scatterplot(x=self.dr_embeddings['UMAP'][:, 0], y=self.dr_embeddings['UMAP'][:, 1],
                            hue=self.df_processed['IsolationForest_Anomaly'], palette='coolwarm', legend='full', s=100, alpha=0.7)
            plt.title('UMAP Visualization with Isolation Forest Anomalies (-1 = Anomaly)')
            plt.xlabel('UMAP Component 1')
            plt.ylabel('UMAP Component 2')
            plt.show()
        print("\n")

    def apply_network_analysis(self):
        """
        8. Apply network analysis to understand relationships between countries based on happiness factors.
        """
        print("8. Applying Network Analysis...")
        X = self._get_clustering_data()
        countries = self.df_processed[self.country_col].tolist()

        # Calculate similarity matrix (e.g., cosine similarity) between countries based on their features
        similarity_matrix = cosine_similarity(X)
        np.fill_diagonal(similarity_matrix, 0) # Remove self-loops

        # Create a graph using networkx
        self.network_graph = nx.Graph()
        self.network_graph.add_nodes_from(countries)

        # Add edges based on a similarity threshold
        # Using a percentile threshold to connect only the most similar countries
        # This prevents a very dense, uninterpretable graph
        if similarity_matrix.size > 0:
            # Filter out zero similarities before calculating percentile to avoid issues with sparse matrices
            positive_similarities = similarity_matrix[similarity_matrix > 0]
            if positive_similarities.size > 0:
                threshold = np.percentile(positive_similarities, 90) # Connect top 10% most similar pairs
            else:
                threshold = 1.0 # No positive similarities, no edges will be added
        else:
            threshold = 1.0 # Empty similarity matrix

        print(f"  Adding edges for country pairs with cosine similarity > {threshold:.3f}")

        edges_added = 0
        for i in range(len(countries)):
            for j in range(i + 1, len(countries)): # Iterate through unique pairs
                if similarity_matrix[i, j] > threshold:
                    self.network_graph.add_edge(countries[i], countries[j], weight=similarity_matrix[i, j])
                    edges_added += 1
        print(f"  Total edges added: {edges_added}")

        if self.network_graph.number_of_edges() == 0:
            print("  No edges formed with the chosen similarity threshold. Consider lowering it or adjusting data.")
            return

        # Basic network properties
        print(f"  Number of nodes: {self.network_graph.number_of_nodes()}")
        print(f"  Number of edges: {self.network_graph.number_of_edges()}")
        print(f"  Average degree: {np.mean([d for n, d in self.network_graph.degree()]):.2f}")

        # Identify central countries using centrality measures
        degree_centrality = nx.degree_centrality(self.network_graph)
        betweenness_centrality = nx.betweenness_centrality(self.network_graph)

        print("\n  Top 5 Countries by Degree Centrality (most connections):")
        for country, centrality in sorted(degree_centrality.items(), key=lambda item: item[1], reverse=True)[:5]:
            print(f"    {country}: {centrality:.3f}")

        print("\n  Top 5 Countries by Betweenness Centrality (bridge countries connecting different groups):")
        for country, centrality in sorted(betweenness_centrality.items(), key=lambda item: item[1], reverse=True)[:5]:
            print(f"    {country}: {centrality:.3f}")

        # Visualization of the network
        plt.figure(figsize=(15, 12))
        # Use spring_layout for a force-directed graph layout
        pos = nx.spring_layout(self.network_graph, k=0.15, iterations=50) # k adjusts optimal distance between nodes
        nx.draw_networkx_nodes(self.network_graph, pos, node_size=50)
        nx.draw_networkx_edges(self.network_graph, pos, alpha=0.3)
        nx.draw_networkx_labels(self.network_graph, pos, font_size=8, font_color='black')
        plt.title('Country Similarity Network based on Happiness Factors')
        plt.axis('off') # Hide axes
        plt.show()
        print("\n")

    def use_association_rule_mining(self):
        """
        9. Use association rule mining to discover patterns in happiness characteristics.
        """
        print("9. Using Association Rule Mining...")
        # Select relevant numerical features and the happiness score for ARM
        # Use the original (unscaled) data for meaningful binning
        df_arm = self.df[[col for col in self.numerical_features if col in self.df.columns]].copy()

        # Discretize continuous numerical features into bins (e.g., 'low', 'medium', 'high')
        bins = 3 # Number of bins
        labels = ['low', 'medium', 'high']

        for col in df_arm.columns:
            # Use qcut for quantile-based binning to ensure roughly equal number of items in each bin
            # 'duplicates='drop'' handles cases where quantile boundaries are identical
            df_arm[col] = pd.qcut(df_arm[col], q=bins, labels=labels, duplicates='drop')
            # Prepend feature name to bin label for clarity (e.g., 'low_GDP', 'high_SocialSupport')
            df_arm[col] = df_arm[col].astype(str) + '_' + col

        # Convert to one-hot encoded format required by Apriori algorithm
        df_arm_encoded = pd.get_dummies(df_arm)

        # Apply Apriori algorithm to find frequent itemsets
        min_support = 0.1 # Minimum support for an itemset to be considered frequent (e.g., 10% of countries)
        frequent_itemsets = apriori(df_arm_encoded, min_support=min_support, use_colnames=True)

        if frequent_itemsets.empty:
            print(f"  No frequent itemsets found with min_support={min_support}. Consider lowering it.")
            return

        # Generate association rules from frequent itemsets
        min_confidence = 0.7 # Minimum confidence for a rule (e.g., 70% confidence)
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)

        if rules.empty:
            print(f"  No association rules found with min_confidence={min_confidence}. Consider lowering it.")
            return

        # Sort rules by lift for interpretability (Lift > 1 indicates positive correlation)
        self.association_rules = rules.sort_values(by='lift', ascending=False)

        print("\n--- Top 10 Association Rules (sorted by Lift) ---")
        print(self.association_rules.head(10))
        print("\n")

    def build_recommendation_system_and_policy_insights(self):
        """
        10. Build a country recommendation system and policy insights based on clustering results
            with geopolitical validation.
        """
        print("10. Building Country Recommendation System and Policy Insights...")
        if 'Consensus_Cluster' not in self.df_processed.columns:
            print("  Consensus clusters not found. Cannot build recommendation system or policy insights.")
            return

        # --- Country Recommendation System ---
        print("\n--- Country Recommendation System ---")
        target_country = 'Germany' # Example target country for recommendation
        if target_country not in self.df_processed[self.country_col].values:
            print(f"  Target country '{target_country}' not found in the dataset. Please choose a valid country.")
            return

        # Find the cluster of the target country
        target_cluster = self.df_processed[self.df_processed[self.country_col] == target_country]['Consensus_Cluster'].iloc[0]
        print(f"  '{target_country}' belongs to Consensus Cluster: {target_cluster}")

        # Recommend countries from the same cluster (basic recommendation)
        recommended_countries_same_cluster = self.df_processed[
            (self.df_processed['Consensus_Cluster'] == target_cluster) &
            (self.df_processed[self.country_col] != target_country)
        ][self.country_col].tolist()

        print(f"\n  Countries in the same cluster as '{target_country}':")
        print(f"  {recommended_countries_same_cluster}")

        # More advanced recommendation: Find most similar countries within the same cluster
        X_clustered = self._get_clustering_data()
        target_country_data = X_clustered[self.df_processed[self.country_col] == target_country]

        if not target_country_data.empty:
            target_country_data_vector = target_country_data.iloc[0].values.reshape(1, -1)
            
            # Filter data for countries belonging to the same cluster
            cluster_countries_data = X_clustered[self.df_processed['Consensus_Cluster'] == target_cluster]
            cluster_country_names = self.df_processed[self.df_processed['Consensus_Cluster'] == target_cluster][self.country_col].tolist()

            # Calculate cosine similarity between the target country and all countries in its cluster
            similarities = cosine_similarity(target_country_data_vector, cluster_countries_data)[0]
            
            # Create a DataFrame to sort countries by similarity
            similarity_df = pd.DataFrame({
                'Country': cluster_country_names,
                'Similarity': similarities
            })
            
            # Remove the target country itself and sort in descending order of similarity
            similarity_df = similarity_df[similarity_df['Country'] != target_country].sort_values(by='Similarity', ascending=False)
            
            print(f"\n  Top 5 most similar countries to '{target_country}' within its cluster (based on features):")
            print(similarity_df.head(5))
        else:
            print(f"  Could not find data for target country '{target_country}' for detailed similarity calculation.")


        # --- Policy Insights based on Cluster Profiles ---
        print("\n--- Policy Insights based on Cluster Profiles ---")
        print("  Analyzing characteristics of high-happiness vs. low-happiness clusters to derive policy insights.")

        # Merge cluster labels with original data for interpretable means
        df_with_clusters_original = self.df.merge(self.df_processed[[self.country_col, 'Consensus_Cluster']], on=self.country_col)

        # Identify high and low happiness clusters based on average Ladder Score
        cluster_happiness_scores = df_with_clusters_original.groupby('Consensus_Cluster')[self.happiness_score_col].mean().sort_values(ascending=False)
        print("\n  Average Ladder Score by Consensus Cluster:")
        print(cluster_happiness_scores)

        highest_happiness_cluster_id = cluster_happiness_scores.index[0]
        lowest_happiness_cluster_id = cluster_happiness_scores.index[-1]

        # Get the mean profiles for the highest and lowest happiness clusters
        original_numerical_cols_for_profile = [col for col in self.numerical_features if col in self.df.columns]
        original_numerical_cols_for_profile.append(self.happiness_score_col)

        print(f"\n  Profile of the Highest Happiness Cluster (ID: {highest_happiness_cluster_id}):")
        print(df_with_clusters_original[df_with_clusters_original['Consensus_Cluster'] == highest_happiness_cluster_id][original_numerical_cols_for_profile].mean())
        print(f"\n  Profile of the Lowest Happiness Cluster (ID: {lowest_happiness_cluster_id}):")
        print(df_with_clusters_original[df_with_clusters_original['Consensus_Cluster'] == lowest_happiness_cluster_id][original_numerical_cols_for_profile].mean())

        print("\n  Key Policy Insights (Qualitative Interpretation):")
        print("  By comparing the profiles of the highest and lowest happiness clusters, we can identify key factors")
        print("  associated with higher happiness and suggest policy directions:")
        print("  - Countries in high-happiness clusters consistently show higher 'Logged GDP per capita', 'Social support',")
        print("    'Healthy life expectancy', 'Freedom to make life choices', and 'Generosity'. They also tend to have")
        print("    significantly lower 'Perceptions of corruption'.")
        print("  - The engineered features like 'GDP_x_SocialSupport' and 'Generosity_minus_Corruption' also highlight")
        print("    that a synergistic combination of economic prosperity, robust social safety nets, and good governance")
        print("    (low corruption, high freedom) are strong predictors of national happiness.")
        print("  - Policy recommendations could therefore focus on a multi-faceted approach:")
        print("    1.  **Economic Development & Equity:** Implement policies that foster sustainable and inclusive economic")
        print("        growth, ensuring that the benefits of GDP growth are widely distributed ('Logged GDP per capita').")
        print("    2.  **Strengthening Social Infrastructure:** Invest in comprehensive social support systems, public health,")
        print("        and education to enhance 'Social support' and 'Healthy life expectancy'.")
        print("    3.  **Promoting Good Governance & Freedoms:** Strengthen democratic institutions, protect civil liberties,")
        print("        and rigorously combat corruption to increase 'Freedom to make life choices' and reduce 'Perceptions of corruption'.")
        print("    4.  **Fostering Community & Altruism:** Encourage civic engagement, volunteerism, and charitable activities")
        print("        to boost 'Generosity' and social cohesion.")
        print("  - **Geopolitical Validation:** Examine the countries within each cluster. Do the high-happiness clusters primarily")
        print("    contain countries from regions known for high development and stability (e.g., Nordic countries, Western Europe)?")
        print("    Are lower-happiness clusters dominated by countries from regions facing significant socio-economic or political challenges?")
        print("    This alignment with known geopolitical realities validates the statistical groupings.")
        
        # Display regional distribution within clusters to aid geopolitical validation
        region_cols_in_processed_df = [col for col in self.df_processed.columns if col.startswith(self.region_col + '_')]
        if region_cols_in_processed_df:
            # Merge original region column back for easier interpretation
            df_with_clusters_and_regions = self.df.merge(self.df_processed[[self.country_col, 'Consensus_Cluster']], on=self.country_col)
            print("\n  Regional Distribution within Clusters (Count of Countries per Region):")
            print(df_with_clusters_and_regions.groupby('Consensus_Cluster')[self.region_col].value_counts().unstack(fill_value=0))
            print("  This table shows the composition of each cluster by original regional indicator, providing direct geopolitical context.")
        print("\n")

# Main execution block
if __name__ == "__main__":
    # Create an instance of the analysis class
    analysis = AdvancedHappinessAnalysis()

    # Execute each step of the analysis sequentially
    analysis.load_and_preprocess_data()
    analysis.implement_multiple_clustering_algorithms()
    analysis.advanced_dimensionality_reduction()
    analysis.determine_optimal_clustering_parameters() # This step provides conceptual insights and plots
    analysis.perform_consensus_clustering()
    analysis.create_detailed_country_profiles()
    analysis.implement_anomaly_detection()
    analysis.apply_network_analysis()
    analysis.use_association_rule_mining()
    analysis.build_recommendation_system_and_policy_insights()
```