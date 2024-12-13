import helpers
import personality as prs
import matplotlib.pyplot as plt
import os
import numpy as np
from models import mean_squared_error_gd

data_path = os.path.join(os.getcwd(), "Data", "Lifespan")

# Step 1 : Load the data
print("Loading worm data...")
X_train, X_test, y_train, y_test = load_whole_data(data_path=data_path, test_size=0.2, random_state=42)
print("Data loaded successfully!")


# Step 2: Initialize parameters
initial_w = np.zeros(X_train.shape[1])  # Initialize weights as zeros
max_iters = 1000  # Number of iterations
gamma = 0.0001  # Learning rate

# Step 2.2 : Check the shapes before matrix multiplication
print(f"Shape of X_train: {X_train.shape}")
print(f"Shape of y_train: {y_train.shape}")
print(f"Shape of initial_w: {initial_w.shape}")

# Step 3 : Run the model
w, loss = mean_squared_error_gd(y_train, X_train, initial_w, max_iters, gamma)

# Step 4: Print the results

#----------------------------------------------------------------------------------------------
# Personality
# Load data Optogenetics

# Step 1: Prepare the feature matrix
feature_matrix = prs.create_feature_matrix(combined_worms, percentile=50).drop(columns=['worm_name'])

# Step 2: Perform hierarchical clustering and visualize the dendrogram
prs.perform_hierarchical_clustering(feature_matrix, method='ward', output_file='dendrogram.png')

# Step 3: From the dendrogram, determine the number of clusters (e.g., k = 4)
num_clusters = 4  # Example: You decide the number of clusters from the dendrogram

# Step 4: Perform K-Means clustering
cluster_labels = prs.perform_kmeans_clustering(feature_matrix, num_clusters=num_clusters)

# Step 5: Add cluster labels to the feature matrix
feature_matrix['Cluster'] = cluster_labels

# Step 6: Analyze or visualize clusters
print(feature_matrix.head())

# Example: Visualize clusters in 2D space using PCA
from sklearn.decomposition import PCA
import seaborn as sns

pca = PCA(n_components=2)
reduced_features = pca.fit_transform(feature_matrix.drop(columns=['Cluster']))
feature_matrix['PCA1'], feature_matrix['PCA2'] = reduced_features[:, 0], reduced_features[:, 1]

plt.figure(figsize=(10, 6))
sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=feature_matrix, palette='viridis', s=50)
plt.title("Cluster Visualization (PCA Reduced)")
plt.show()

