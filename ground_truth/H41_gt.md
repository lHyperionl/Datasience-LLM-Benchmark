```python
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# --- 1. Load the dataset ---
try:
    df = pd.read_csv('imdb_top_250.csv')
except FileNotFoundError:
    print("Error: 'imdb_top_250.csv' not found. Please ensure the file is in the current directory.")
    exit()

# --- Data Cleaning for 'Runtime' (if necessary) ---
# The 'Runtime' column might be in a string format like '142 min'.
# We need to convert it to a numeric type (e.g., integer).
if 'Runtime' in df.columns:
    # Convert to string first to handle potential mixed types, then remove ' min'
    df['Runtime'] = df['Runtime'].astype(str).str.replace(' min', '', regex=False)
    # Convert to numeric, coercing errors (e.g., non-numeric values) to NaN
    df['Runtime'] = pd.to_numeric(df['Runtime'], errors='coerce')
else:
    print("Error: 'Runtime' column not found in the dataset.")
    exit()

# Ensure 'Rating' is numeric and handle potential missing values in selected features
if 'Rating' in df.columns:
    df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce')
else:
    print("Error: 'Rating' column not found in the dataset.")
    exit()

# Drop rows where 'Rating' or 'Runtime' are NaN after conversion
df.dropna(subset=['Rating', 'Runtime'], inplace=True)

# --- 2. Select the features `Rating` and `Runtime` for clustering ---
features = df[['Rating', 'Runtime']]

# --- 3. Standardize these features using StandardScaler ---
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# --- 4. Apply K-Means clustering ---
# Using K=3 clusters, n_init='auto', and random_state=42 for reproducibility
kmeans = KMeans(n_clusters=3, n_init='auto', random_state=42)
df['cluster_label'] = kmeans.fit_predict(scaled_features)

# --- 5. Create a scatter plot ---
# Plotting original (non-standardized) Runtime vs. Rating, colored by cluster labels
plt.figure(figsize=(10, 7))
scatter = plt.scatter(df['Runtime'], df['Rating'], c=df['cluster_label'], cmap='viridis', s=50, alpha=0.8)

# Add title and axis labels
plt.title('K-Means Clustering of Movies by Runtime and Rating (K=3)')
plt.xlabel('Runtime (minutes)')
plt.ylabel('Rating')

# Add a color bar to indicate cluster labels
plt.colorbar(scatter, label='Cluster Label')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# --- 6. For each cluster, calculate and print the mean `Rating` and mean `Runtime` ---
print("\n--- Mean Rating and Runtime for each cluster ---")
cluster_means = df.groupby('cluster_label')[['Rating', 'Runtime']].mean()
print(cluster_means)
```