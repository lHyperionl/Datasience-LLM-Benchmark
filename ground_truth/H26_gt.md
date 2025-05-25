```python
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# --- 1. Load the dataset ---
try:
    df = pd.read_csv('iris.csv')
except FileNotFoundError:
    print("Error: 'iris.csv' not found. Please ensure the file is in the current directory.")
    exit()

# --- 2. Select the features ---
features = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
X = df[features]

# --- 3. Standardize these features using StandardScaler ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- 4. Apply K-Means clustering to the standardized features ---
# Assuming K=3 to match the number of species
kmeans = KMeans(n_clusters=3, n_init='auto', random_state=42)
kmeans_labels = kmeans.fit_predict(X_scaled)

# --- 5. Apply Principal Component Analysis (PCA) to reduce the standardized features to 2 principal components ---
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# --- 6. Create a scatter plot of the two principal components, coloring the points by their K-Means assigned cluster labels ---
plt.figure(figsize=(10, 7))
scatter1 = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_labels, cmap='viridis', alpha=0.8)
plt.title('PCA of Iris Dataset: Points Colored by K-Means Cluster Labels')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(scatter1, label='K-Means Cluster')
plt.grid(True)
plt.show()

# --- 7. Create a second scatter plot of the two principal components, this time coloring the points by the true Species labels ---
# Encode Species to numerical values for coloring
label_encoder = LabelEncoder()
true_species_encoded = label_encoder.fit_transform(df['Species'])

plt.figure(figsize=(10, 7))
scatter2 = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=true_species_encoded, cmap='viridis', alpha=0.8)
plt.title('PCA of Iris Dataset: Points Colored by True Species Labels')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
# Create a color bar with actual species names
cbar = plt.colorbar(scatter2, ticks=np.unique(true_species_encoded))
cbar.set_ticklabels(label_encoder.inverse_transform(np.unique(true_species_encoded)))
cbar.set_label('True Species')
plt.grid(True)
plt.show()
```