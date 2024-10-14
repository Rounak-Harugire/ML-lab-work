import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Generate 300 data points with 4 centers
X, y = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# Define the KMeans model
kmeans = KMeans(n_clusters=4)  # We know there are 4 clusters

# Fit the model to the data
kmeans.fit(X)

# Get the predicted cluster labels and centroids
y_kmeans = kmeans.predict(X)
centroids = kmeans.cluster_centers_

# Plot the data points and centroids
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis', label='Data Points')
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=200, alpha=0.75, marker='X', label='Centroids')
plt.title('K-Means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True)
plt.show()
