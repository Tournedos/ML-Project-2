import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.spatial import ConvexHull
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans



def plot_speed_vs_time(worms, worm_names=None, output_dir=None): # to find threshold of high speed (vs low speed)
    """
    Plot speed vs. time for individual worms and save the plots.

    Args:
        worms (list): List of DataFrames, each representing one worm.
        worm_names (list, optional): List of worm names for labeling the plots. Defaults to None.
        output_dir (str, optional): Directory to save the plots. If None, plots are not saved.

    Returns:
        None
    """
    if worm_names is None:
        worm_names = [f"Worm {index + 1}" for index in range(len(worms))]

    for worm, worm_name in zip(worms, worm_names):
        # Ensure "Frame" and "Speed" columns exist
        if 'Frame' not in worm.columns or 'Speed' not in worm.columns:
            print(f"Skipping {worm_name} as required columns are missing.")
            continue

        # Plot Speed vs. Frame
        plt.figure(figsize=(10, 6))
        plt.plot(worm['Frame'], worm['Speed'], label="Speed", color='green', alpha=0.7)
        plt.xlabel("Frame (Time)", fontsize=12)
        plt.ylabel("Speed (Pixels/Second)", fontsize=12)
        plt.title(f"Speed vs. Time for {worm_name}", fontsize=14)
        plt.legend()

        # Save plot if output_dir is specified
        if output_dir:
            save_path = os.path.join(output_dir, f"{worm_name}_speed_vs_time.png")
            plt.savefig(save_path)

        plt.close()



def calculate_distance_and_area(data): # total distance parcoured by 1 worm, and area covered
    """
    Calculate the total distance covered and area explored for a single worm.

    Args:
        data (pd.DataFrame): A DataFrame containing worm data with 'X' and 'Y' columns.

    Returns:
        dict: A dictionary with 'total_distance' and 'area_explored'.
    """
    if 'X' not in data.columns or 'Y' not in data.columns:
        raise ValueError("The provided data must contain 'X' and 'Y' columns.")

    # Calculate Total Distance Covered
    distances = np.sqrt(np.diff(data['X'])**2 + np.diff(data['Y'])**2)
    total_distance = np.sum(distances)

    # Calculate Area Explored (Bounding Box or Convex Hull)
    coords = data[['X', 'Y']].drop_duplicates().to_numpy()

    if len(coords) < 3:
        # Convex Hull requires at least 3 unique points
        area_explored = 0.0
    else:
        try:
            hull = ConvexHull(coords)
            area_explored = hull.volume  # Convex hull area
        except Exception as e:
            print(f"Error calculating convex hull: {e}")
            area_explored = 0.0

    return {
        'total_distance': total_distance,
        'area_explored': area_explored
    }



def calculate_worm_statistics(worms, percentile=50):
    """
    Calculate comprehensive statistics (speed, movement frequency, distance, area) for multiple worms.

    Args:
        worms (list): List of DataFrames, each representing one worm's data.
                      Each DataFrame must include 'Speed', 'X', and 'Y' columns.
        percentile (int): The percentile of speed to define the activity threshold (e.g., 50 for median).

    Returns:
        list: A list of dictionaries containing statistics for each worm.
    """
    stats = []

    for i, worm in enumerate(worms):
        if 'Speed' not in worm.columns or 'X' not in worm.columns or 'Y' not in worm.columns:
            print(f"Skipping Worm {i+1}: Missing required columns ('Speed', 'X', 'Y').")
            continue

        # Speed Statistics -> fast worm
        average_speed = worm['Speed'].mean()
        variance_speed = worm['Speed'].var()

        # Movement Frequency
        threshold = worm['Speed'].quantile(percentile / 100.0)
        active_frames = (worm['Speed'] > threshold).sum()
        total_frames = len(worm)
        movement_frequency = (active_frames / total_frames) * 100

        # Ratio of Active Time vs. Total Time -> active worm
        active_time_ratio = active_frames / total_frames

        # Total Distance and Area Explored -> explorator worm
        distance_and_area = calculate_distance_and_area(worm)
        total_distance = distance_and_area['total_distance']
        area_explored = distance_and_area['area_explored']

        # Compile all statistics
        stats.append({
            'worm_name': f"Worm {i+1}",
            'average_speed': average_speed,
            'variance_speed': variance_speed,
            'threshold': threshold,
            'movement_frequency': movement_frequency,
            'active_time_ratio': active_time_ratio,
            'total_distance': total_distance,
            'area_explored': area_explored
        })

    return stats



def create_feature_matrix(worms, percentile=50): # Creates a feature matrix of all stats given from calculate_worm_statistics()
    """
    Create a feature matrix for worms based on movement and exploration features.

    Args:
        worms (list): List of DataFrames, each representing one worm's data.
                      Each DataFrame must include 'Speed', 'X', and 'Y' columns.
        percentile (int): The percentile of speed to define the activity threshold (e.g., 50 for median).

    Returns:
        pd.DataFrame: A DataFrame where each row represents a worm, and columns are the extracted features.
    """
    worm_stats = calculate_worm_statistics(worms, percentile=percentile)
    feature_matrix = pd.DataFrame(worm_stats)

    return feature_matrix



def perform_hierarchical_clustering(feature_matrix, method='ward', output_file=None): # to find the number of clusters
    """
    Perform hierarchical clustering and plot a dendrogram.

    Args:
        feature_matrix (pd.DataFrame): Feature matrix for clustering (without labels).
        method (str): Linkage method for hierarchical clustering (default: 'ward').
        output_file (str, optional): Path to save the dendrogram plot. If None, the plot is not saved.

    Returns:
        None
    """
    # Perform hierarchical clustering
    linkage_matrix = linkage(feature_matrix, method=method)

    # Plot dendrogram
    plt.figure(figsize=(12, 8))
    dendrogram(linkage_matrix, leaf_rotation=90, leaf_font_size=10)
    plt.title("Hierarchical Clustering Dendrogram")
    plt.xlabel("Data Points (Worms)")
    plt.ylabel("Distance")

    # Save or show the plot
    if output_file:
        plt.savefig(output_file)
    else:
        plt.show()



def perform_kmeans_clustering(feature_matrix, num_clusters): # K-Means clustering
    """
    Perform K-Means clustering on the feature matrix.

    Args:
        feature_matrix (pd.DataFrame): Feature matrix for clustering (without labels).
        num_clusters (int): Number of clusters (k).

    Returns:
        pd.Series: Cluster labels for each data point in the feature matrix.
    """
    # Perform K-Means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(feature_matrix)

    # Return cluster labels
    return pd.Series(cluster_labels, name='Cluster')


