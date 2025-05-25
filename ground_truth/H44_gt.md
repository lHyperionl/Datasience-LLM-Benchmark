```python
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Set a style for the plots for better aesthetics
sns.set_style("whitegrid")

# --- 1. Load the dataset ---
try:
    df = pd.read_csv('penguins.csv')
except FileNotFoundError:
    print("Error: 'penguins.csv' not found. Please ensure the file is in the current directory.")
    exit()

# --- 2. Select the features ---
selected_features = [
    'culmen_length_mm',
    'culmen_depth_mm',
    'flipper_length_mm',
    'body_mass_g'
]

# Create a DataFrame with only the selected features and the 'species' column for later comparison
# This ensures that when we drop NAs, the 'species' column is also aligned.
df_selected = df[selected_features + ['species']].copy()

# --- 3. Drop any rows where any of these four selected features have missing values (NA) ---
# Store the original species labels before dropping NAs, ensuring they correspond to the filtered data
df_cleaned = df_selected.dropna(subset=selected_features)

# Separate features (X) and the original species labels (y_true)
X = df_cleaned[selected_features]
original_species_labels = df_cleaned['species']

# --- 4. Standardize these features using StandardScaler ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- 5. Apply K-Means clustering to the standardized features ---
# Use K=3 clusters, n_init='auto', and random_state=42
kmeans = KMeans(n_clusters=3, n_init='auto', random_state=42)
kmeans_labels = kmeans.fit_predict(X_scaled)

# --- 6. Apply Principal Component Analysis (PCA) to reduce the standardized features to 2 principal components ---
pca = PCA(n_components=2)
principal_components = pca.fit_transform(X_scaled)

# Create a DataFrame for the principal components for easier plotting
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])

# Add the K-Means cluster labels and original species labels to the PCA DataFrame
pca_df['KMeans_Cluster'] = kmeans_labels
pca_df['Original_Species'] = original_species_labels.reset_index(drop=True) # Reset index to align with pca_df

# --- 7. Create a scatter plot of the two principal components, coloring the points by their K-Means assigned cluster labels ---
plt.figure(figsize=(10, 7))
sns.scatterplot(
    x='PC1',
    y='PC2',
    hue='KMeans_Cluster',
    palette='viridis', # A good color palette for distinct clusters
    data=pca_df,
    s=100, # Size of the points
    alpha=0.8 # Transparency
)
plt.title('PCA of Penguin Features: K-Means Clusters (K=3)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='K-Means Cluster')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# --- 8. Create a second scatter plot of the two principal components, this time coloring the points by the original `species` labels ---
plt.figure(figsize=(10, 7))
sns.scatterplot(
    x='PC1',
    y='PC2',
    hue='Original_Species',
    palette='tab10', # A good categorical color palette
    data=pca_df,
    s=100,
    alpha=0.8
)
plt.title('PCA of Penguin Features: Original Species Labels (for Comparison)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Original Species')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
```