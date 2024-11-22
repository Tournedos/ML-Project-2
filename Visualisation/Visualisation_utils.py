import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Load the CSV file
def load_worm_data(file_path):
    """
    Load worm data from a CSV file.
    
    Args:
        file_path (str): Path to the CSV file.
    
    Returns:
        pd.DataFrame: DataFrame containing the worm data.
    """
    return pd.read_csv(file_path)

# Step 2: Plot the trajectory
def plot_trajectory(worm_data, worm_id="Worm 1"):
    """
    Plot the trajectory of the worm based on its X and Y positions.
    
    Args:
        worm_data (pd.DataFrame): DataFrame containing worm data with X and Y columns.
        worm_id (str): Identifier for the worm (for plot title).
    """
    plt.figure(figsize=(8, 6))
    plt.plot(worm_data['X'], worm_data['Y'], label="Trajectory")
    plt.scatter(worm_data['X'].iloc[0], worm_data['Y'].iloc[0], color='green', label="Start", zorder=5)
    plt.scatter(worm_data['X'].iloc[-1], worm_data['Y'].iloc[-1], color='red', label="End", zorder=5)
    plt.title(f"Worm Trajectory - {worm_id}")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.legend()
    plt.grid()
    plt.show()
