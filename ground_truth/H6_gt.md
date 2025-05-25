```python
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, adjusted_rand_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap.umap_ as umap # Correct import for umap-learn library
from scipy.cluster.hierarchy import dendrogram, linkage
from collections import defaultdict

# Suppress warnings for cleaner output, especially from sklearn and seaborn
import warnings
warnings.filterwarnings('ignore')

# Set plot style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 100

# --- 1. Data Loading ---
try:
    df = pd.read_csv('titanic.csv')
except FileNotFoundError:
    print("Error: titanic.csv not found. Please ensure the file is in the current directory.")
    exit()

# Make a copy to preserve original data for later analysis and merging
df_original = df.copy()

# --- 2. Advanced Feature Engineering ---

# 2.1. Social Network Analysis & Family Features
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

# Impute Fare before calculating FarePerPerson to avoid NaN propagation
df['Fare'].fillna(df['Fare'].median(), inplace=True)

# Ticket Group Size: Count how many people share the same ticket
df['TicketGroupSize'] = df.groupby('Ticket')['PassengerId'].transform('count')
df['FarePerPerson'] = df['Fare'] / df['TicketGroupSize']
# Handle potential division by zero if Fare was 0, resulting in NaN for FarePerPerson
df['FarePerPerson'].fillna(0, inplace=True) # If Fare was 0, FarePerPerson is 0.

# 2.2. Socioeconomic Indicators & Title Extraction
def get_title(name):
    """Extracts title from passenger's name."""
    title_search = re.search(' ([A-Za-z]+)\.', name)
    if title_search:
        return title_search.group(1)
    return ""

df['Title'] = df['Name'].apply(get_title)
# Group rare titles for better categorical encoding
df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
df['Title'] = df['Title'].replace(['Mlle', 'Ms'], 'Miss')
df['Title'] = df['Title'].replace('Mme', 'Mrs')

# 2.3. Text-based Features from Names and Tickets
df['NameLength'] = df['Name'].apply(len)

# Ticket Prefix
def get_ticket_prefix(ticket):
    """Extracts the prefix from a ticket number."""
    parts = ticket.split(' ')
    if len(parts) > 1 and not parts[0].isdigit():
        return parts[0].replace('.', '').replace('/', '').upper()
    return 'NUM' # Numeric ticket

df['TicketPrefix'] = df['Ticket'].apply(get_ticket_prefix)
# Group less common prefixes into 'OTHER'
common_prefixes = df['TicketPrefix'].value_counts()[df['TicketPrefix'].value_counts() > 10].index
df['TicketPrefix'] = df['TicketPrefix'].apply(lambda x: x if x in common_prefixes else 'OTHER')

# Cabin Deck and HasCabin
df['CabinDeck'] = df['Cabin'].fillna('U').apply(lambda x: x[0])
df['HasCabin'] = df['Cabin'].notna().astype(int)

# 2.4. Interaction Features
df['Age_Class'] = df['Age'] * df['Pclass']
df['Fare_Class'] = df['Fare'] * df['Pclass']
df['FarePerPerson_Class'] = df['FarePerPerson'] * df['Pclass']

# --- 3. Preprocessing and Scaling ---

# Impute missing values for Age and Embarked
# Age: Use median for numerical stability and robustness to outliers
df['Age'].fillna(df['Age'].median(), inplace=True)
# Embarked: Use mode for categorical feature
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Define categorical and numerical features for preprocessing
categorical_features = ['Pclass', 'Sex', 'Embarked', 'Title', 'CabinDeck', 'TicketPrefix', 'IsAlone', 'HasCabin']
numerical_features = ['Age', 'Fare', 'FamilySize', 'TicketGroupSize', 'FarePerPerson',
                      'NameLength', 'Age_Class', 'Fare_Class', 'FarePerPerson_Class', 'SibSp', 'Parch']

# Create preprocessing pipelines for numerical and categorical features
numerical_transformer = Pipeline(steps=[
    ('scaler', StandardScaler()) # Scale numerical features
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore')) # One-hot encode categorical features
])

# Create a preprocessor using ColumnTransformer to apply different transformations to different columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Apply preprocessing to the DataFrame
X_processed = preprocessor.fit_transform(df)

# Get feature names after one-hot encoding for better interpretability
ohe_feature_names = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_features)
all_feature_names = numerical_features + list(ohe_feature_names)

# Convert processed data back to a DataFrame
df_processed = pd.DataFrame(X_processed, columns=all_feature_names)

# Add PassengerId and Survived back to the processed DataFrame for analysis later
df_processed['PassengerId'] = df['PassengerId']
df_processed['Survived'] = df['Survived']

# --- 4. Dimensionality Reduction for Visualization and Clustering Enhancement ---

# Data for dimensionality reduction (excluding PassengerId and Survived)
X_dim_reduction = df_processed.drop(columns=['PassengerId', 'Survived'])

# PCA for general variance explanation and initial visualization
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_dim_reduction)
df_processed['PCA1'] = X_pca[:, 0]
df_processed['PCA2'] = X_pca[:, 1]
print(f"PCA explained variance ratio: {pca.explained_variance_ratio_.sum():.2f}")

# t-SNE for non-linear visualization (computationally intensive, use smaller perplexity if needed)
# Perplexity is a balance between local and global aspects of the data.
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000, learning_rate='auto')
X_tsne = tsne.fit_transform(X_dim_reduction)
df_processed['TSNE1'] = X_tsne[:, 0]
df_processed['TSNE2'] = X_tsne[:, 1]

# UMAP for faster and often better visualization (preserves global structure better than t-SNE)
reducer = umap.UMAP(n_components=2, random_state=42)
X_umap = reducer.fit_transform(X_dim_reduction)
df_processed['UMAP1'] = X_umap[:, 0]
df_processed['UMAP2'] = X_umap[:, 1]

# Data for clustering (without PassengerId, Survived, and visualization dimensions)
X_cluster = df_processed.drop(columns=['PassengerId', 'Survived', 'PCA1', 'PCA2', 'TSNE1', 'TSNE2', 'UMAP1', 'UMAP2'])

# --- 5. Determine Optimal Number of Clusters using Multiple Validation Metrics ---

# Range of clusters to test
k_range = range(2, 11) # Test from 2 to 10 clusters

# Store scores for different algorithms
kmeans_scores = {'silhouette': [], 'calinski_harabasz': [], 'davies_bouldin': [], 'inertia': []}
gmm_scores = {'silhouette': [], 'calinski_harabasz': [], 'davies_bouldin': []}
hierarchical_scores = {'silhouette': [], 'calinski_harabasz': [], 'davies_bouldin': []}

print("\n--- Determining Optimal Number of Clusters ---")

for k in k_range:
    # K-Means
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10) # n_init='auto' or 10 for robustness
    kmeans_labels = kmeans.fit_predict(X_cluster)
    kmeans_scores['silhouette'].append(silhouette_score(X_cluster, kmeans_labels))
    kmeans_scores['calinski_harabasz'].append(calinski_harabasz_score(X_cluster, kmeans_labels))
    kmeans_scores['davies_bouldin'].append(davies_bouldin_score(X_cluster, kmeans_labels))
    kmeans_scores['inertia'].append(kmeans.inertia_)

    # Gaussian Mixture Models
    gmm = GaussianMixture(n_components=k, random_state=42)
    gmm.fit(X_cluster)
    gmm_labels = gmm.predict(X_cluster)
    gmm_scores['silhouette'].append(silhouette_score(X_cluster, gmm_labels))
    gmm_scores['calinski_harabasz'].append(calinski_harabasz_score(X_cluster, gmm_labels))
    gmm_scores['davies_bouldin'].append(davies_bouldin_score(X_cluster, gmm_labels))

    # Hierarchical Clustering (Agglomerative)
    agg_clustering = AgglomerativeClustering(n_clusters=k)
    agg_labels = agg_clustering.fit_predict(X_cluster)
    hierarchical_scores['silhouette'].append(silhouette_score(X_cluster, agg_labels))
    hierarchical_scores['calinski_harabasz'].append(calinski_harabasz_score(X_cluster, agg_labels))
    hierarchical_scores['davies_bouldin'].append(davies_bouldin_score(X_cluster, agg_labels))

# Plotting validation metrics
plt.figure(figsize=(18, 12))

# K-Means Elbow Method (Inertia)
plt.subplot(2, 2, 1)
plt.plot(k_range, kmeans_scores['inertia'], marker='o')
plt.title('K-Means Elbow Method (Inertia)')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia (Sum of Squared Distances)')
plt.grid(True)

# Silhouette Score (Higher is better)
plt.subplot(2, 2, 2)
plt.plot(k_range, kmeans_scores['silhouette'], marker='o', label='K-Means')
plt.plot(k_range, gmm_scores['silhouette'], marker='o', label='GMM')
plt.plot(k_range, hierarchical_scores['silhouette'], marker='o', label='Hierarchical')
plt.title('Silhouette Score (Higher is Better)')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Score')
plt.legend()
plt.grid(True)

# Calinski-Harabasz Index (Higher is better)
plt.subplot(2, 2, 3)
plt.plot(k_range, kmeans_scores['calinski_harabasz'], marker='o', label='K-Means')
plt.plot(k_range, gmm_scores['calinski_harabasz'], marker='o', label='GMM')
plt.plot(k_range, hierarchical_scores['calinski_harabasz'], marker='o', label='Hierarchical')
plt.title('Calinski-Harabasz Index (Higher is Better)')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Score')
plt.legend()
plt.grid(True)

# Davies-Bouldin Index (Lower is better)
plt.subplot(2, 2, 4)
plt.plot(k_range, kmeans_scores['davies_bouldin'], marker='o', label='K-Means')
plt.plot(k_range, gmm_scores['davies_bouldin'], marker='o', label='GMM')
plt.plot(k_range, hierarchical_scores['davies_bouldin'], marker='o', label='Hierarchical')
plt.title('Davies-Bouldin Index (Lower is Better)')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Score')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Based on the plots, choose an optimal k. For demonstration, let's pick optimal_k = 4.
# In a real scenario, you would analyze the plots to find the "elbow" or peak scores.
optimal_k = 4
print(f"\nOptimal number of clusters chosen for demonstration: {optimal_k}")

# --- 6. Apply Multiple Clustering Algorithms with Optimal K ---

# K-Means
kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df_processed['KMeans_Cluster'] = kmeans_final.fit_predict(X_cluster)

# Hierarchical Clustering (Agglomerative)
agg_final = AgglomerativeClustering(n_clusters=optimal_k)
df_processed['Hierarchical_Cluster'] = agg_final.fit_predict(X_cluster)

# Gaussian Mixture Models
gmm_final = GaussianMixture(n_components=optimal_k, random_state=42)
gmm_final.fit(X_cluster)
df_processed['GMM_Cluster'] = gmm_final.predict(X_cluster)

# DBSCAN (requires careful parameter tuning, often not suitable for all datasets or requires specific density)
# For demonstration, reasonable default values are chosen.
# In practice, eps can be estimated using a k-distance graph (e.g., k-NN distance for k=min_samples-1).
dbscan = DBSCAN(eps=0.5, min_samples=5) # These parameters are highly dataset-dependent
df_processed['DBSCAN_Cluster'] = dbscan.fit_predict(X_cluster)
print(f"\nDBSCAN found {len(df_processed['DBSCAN_Cluster'].unique()) - (1 if -1 in df_processed['DBSCAN_Cluster'].unique() else 0)} clusters (excluding noise points labeled -1).")


# --- 7. Cluster Stability Analysis (Simplified using Adjusted Rand Index for K-Means) ---
# This checks how consistent cluster assignments are across multiple runs with different initializations.
print("\n--- Cluster Stability Analysis (K-Means) ---")
n_runs = 10 # Number of times to run K-Means
kmeans_labels_runs = []
for i in range(n_runs):
    kmeans_stable = KMeans(n_clusters=optimal_k, random_state=i, n_init=10)
    kmeans_labels_runs.append(kmeans_stable.fit_predict(X_cluster))

# Calculate Adjusted Rand Index (ARI) between each run and the first run
# ARI measures the similarity of two clusterings, ignoring permutations.
ari_scores = []
for i in range(1, n_runs):
    ari_scores.append(adjusted_rand_score(kmeans_labels_runs[0], kmeans_labels_runs[i]))

print(f"Adjusted Rand Index scores for {n_runs} K-Means runs (vs. first run):")
print(np.round(ari_scores, 3))
print(f"Mean ARI: {np.mean(ari_scores):.3f}")
print(f"Std ARI: {np.std(ari_scores):.3f}")
if np.mean(ari_scores) > 0.75: # A common threshold for good stability
    print("K-Means clustering appears relatively stable across different initializations.")
else:
    print("K-Means clustering shows some variability. Consider different initializations, algorithms, or feature sets.")

# --- 8. Analyze Cluster Characteristics and Survival Patterns ---

clustering_algorithms = ['KMeans', 'Hierarchical', 'GMM', 'DBSCAN']

for algo in clustering_algorithms:
    print(f"\n--- Analysis for {algo} Clustering ---")
    cluster_col = f'{algo}_Cluster'

    if cluster_col not in df_processed.columns:
        print(f"Skipping {algo} as its cluster column '{cluster_col}' is not found.")
        continue

    # Merge cluster labels back to the original dataframe for easier interpretation of raw features
    df_analysis = df_original.copy()
    df_analysis[cluster_col] = df_processed[cluster_col]

    # Handle DBSCAN noise points (-1 label) by excluding them from cluster analysis
    if algo == 'DBSCAN':
        df_analysis = df_analysis[df_analysis[cluster_col] != -1].copy()
        if df_analysis.empty:
            print(f"No valid clusters found for {algo} after removing noise points. Skipping detailed analysis.")
            continue
        print(f"Note: DBSCAN noise points (label -1) are excluded from this analysis.")

    # Calculate survival rates per cluster
    survival_rates = df_analysis.groupby(cluster_col)['Survived'].mean().reset_index()
    survival_rates['Survived_Percentage'] = survival_rates['Survived'] * 100
    print(f"\nSurvival Rates by {algo} Cluster:")
    print(survival_rates.round(2))

    # Calculate mean/mode of key original features for each cluster profile
    cluster_profiles = df_analysis.groupby(cluster_col).agg(
        Count=('PassengerId', 'size'),
        Age=('Age', 'median'), # Median for age due to potential skew
        Fare=('Fare', 'median'), # Median for fare due to skew
        Pclass=('Pclass', lambda x: x.mode()[0]), # Mode for Pclass
        Sex=('Sex', lambda x: x.mode()[0]), # Mode for Sex
        FamilySize=('FamilySize', 'median'),
        TicketGroupSize=('TicketGroupSize', 'median'),
        Survived_Rate=('Survived', 'mean')
    ).reset_index()
    print(f"\nCluster Profiles for {algo} (Median/Mode of Original Features):")
    print(cluster_profiles.round(2))

    # --- 9. Comprehensive Visualizations ---

    # 9.1. Cluster distributions in reduced dimensions
    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    sns.scatterplot(x='PCA1', y='PCA2', hue=cluster_col, data=df_processed, palette='viridis', legend='full', alpha=0.7)
    plt.title(f'{algo} Clusters on PCA Components')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')

    plt.subplot(1, 3, 2)
    sns.scatterplot(x='TSNE1', y='TSNE2', hue=cluster_col, data=df_processed, palette='viridis', legend='full', alpha=0.7)
    plt.title(f'{algo} Clusters on t-SNE Components')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')

    plt.subplot(1, 3, 3)
    sns.scatterplot(x='UMAP1', y='UMAP2', hue=cluster_col, data=df_processed, palette='viridis', legend='full', alpha=0.7)
    plt.title(f'{algo} Clusters on UMAP Components')
    plt.xlabel('UMAP Component 1')
    plt.ylabel('UMAP Component 2')

    plt.tight_layout()
    plt.show()

    # 9.2. Survival rates per cluster
    plt.figure(figsize=(8, 5))
    sns.barplot(x=cluster_col, y='Survived_Percentage', data=survival_rates, palette='viridis')
    plt.title(f'Survival Rate by {algo} Cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Survival Percentage')
    plt.ylim(0, 100)
    plt.show()

    # 9.3. Feature importance/distribution for each segment (using violin plots for key features)
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    sns.violinplot(x=cluster_col, y='Age', data=df_analysis, palette='viridis')
    plt.title(f'Age Distribution by {algo} Cluster')

    plt.subplot(2, 2, 2)
    sns.violinplot(x=cluster_col, y='Fare', data=df_analysis, palette='viridis')
    plt.title(f'Fare Distribution by {algo} Cluster')

    plt.subplot(2, 2, 3)
    sns.countplot(x=cluster_col, hue='Pclass', data=df_analysis, palette='viridis')
    plt.title(f'Pclass Distribution by {algo} Cluster')

    plt.subplot(2, 2, 4)
    sns.countplot(x=cluster_col, hue='Sex', data=df_analysis, palette='viridis')
    plt.title(f'Sex Distribution by {algo} Cluster')

    plt.tight_layout()
    plt.show()

    # 9.4. Heatmap of scaled feature means for each cluster (using processed data)
    # Re-calculate cluster profiles using the scaled data for heatmap
    X_cluster_with_labels = df_processed.drop(columns=['PassengerId', 'Survived', 'PCA1', 'PCA2', 'TSNE1', 'TSNE2', 'UMAP1', 'UMAP2']).copy()
    X_cluster_with_labels[cluster_col] = df_processed[cluster_col]

    if algo == 'DBSCAN':
        X_cluster_with_labels = X_cluster_with_labels[X_cluster_with_labels[cluster_col] != -1].copy()
        if X_cluster_with_labels.empty:
            continue

    cluster_feature_means = X_cluster_with_labels.groupby(cluster_col).mean()

    plt.figure(figsize=(15, 10))
    sns.heatmap(cluster_feature_means.T, cmap='viridis', annot=True, fmt=".2f", linewidths=.5, linecolor='black')
    plt.title(f'Scaled Feature Means by {algo} Cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Feature')
    plt.show()


# --- 10. Validate Clustering Results using External Criteria (Survival Outcomes) ---

print("\n--- External Validation: Survival Outcomes per Cluster ---")

for algo in clustering_algorithms:
    cluster_col = f'{algo}_Cluster'
    if cluster_col not in df_processed.columns:
        continue

    df_validation = df_processed.copy()
    # Handle DBSCAN noise points (-1)
    if algo == 'DBSCAN':
        df_validation = df_validation[df_validation[cluster_col] != -1].copy()
        if df_validation.empty:
            print(f"No clusters to validate for {algo} after removing noise points.")
            continue

    # Calculate the proportion of survived (1) and not survived (0) within each cluster
    survival_by_cluster = df_validation.groupby(cluster_col)['Survived'].value_counts(normalize=True).unstack().fillna(0)
    survival_by_cluster['Total_Passengers'] = df_validation.groupby(cluster_col).size()
    survival_by_cluster['Survival_Rate'] = survival_by_cluster[1] * 100 # Column 1 is 'Survived'
    survival_by_cluster['Non_Survival_Rate'] = survival_by_cluster[0] * 100 # Column 0 is 'Not Survived'

    print(f"\nSurvival Distribution by {algo} Cluster:")
    print(survival_by_cluster[['Total_Passengers', 'Survival_Rate', 'Non_Survival_Rate']].round(2))

    # Visualizing survival rates for comparison
    plt.figure(figsize=(10, 6))
    sns.barplot(x=survival_by_cluster.index, y='Survival_Rate', data=survival_by_cluster, palette='coolwarm')
    plt.title(f'Survival Rate by {algo} Cluster (External Validation)')
    plt.xlabel('Cluster')
    plt.ylabel('Survival Rate (%)')
    plt.ylim(0, 100)
    plt.show()

print("\nComprehensive passenger segmentation and clustering analysis complete.")
```