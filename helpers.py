import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def find_repo_root():
    percorso_corrente = os.path.dirname(os.path.abspath(__file__))
    while percorso_corrente != os.path.dirname(percorso_corrente):  # As long as we dont reach the right root
        if os.path.exists(os.path.join(percorso_corrente, '.git')):
            return percorso_corrente
        percorso_corrente = os.path.dirname(percorso_corrente)
    raise Exception("Radice della repository non trovata")


def plot_trajectory(worm_data, worm_id, output_file,show_plot=True,save_plot=False):
    """
    Plot the trajectory of the worm and save the plot to a file.

    Args:
        worm_data (pd.DataFrame): Worm data to be plotted.
        worm_id (str): Identifier for the worm.
        output_file (str): Path to save the plot image.

    Returns:
        None
    """
    print(f"{worm_id}: Plotting trajectory...")
    plt.figure()
    plt.plot(worm_data['X'], worm_data['Y'], label=f"{worm_id} Trajectory")
    plt.scatter(worm_data['X'].iloc[0], worm_data['Y'].iloc[0], color='green', label="Start", zorder=5)
    plt.scatter(worm_data['X'].iloc[-1], worm_data['Y'].iloc[-1], color='red', label="End", zorder=5)
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.title(f"Trajectory of {worm_id}")
    plt.legend()

    if save_plot and output_file: # Save the plot if save_plot=True
        plt.savefig(output_file)
        print(f"{worm_id}: Plot saved as {output_file}.")

    if show_plot: # Display the plot interactively if show_plot=True
        plt.show() 
    plt.close()



def split_data(worms, test_size=0.2, random_state=42):
    """
    Split worms into training and testing sets.

    Args:
        worms (list): List of DataFrames, each representing one worm.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Random seed for reproducibility.

    Returns:
        tuple: Two lists containing the training and testing worms.
    """
    train_worms, test_worms = train_test_split(worms, test_size=test_size, random_state=random_state)
    print(f"Data split complete: {len(train_worms)} training worms, {len(test_worms)} testing worms.")
    return train_worms, test_worms



def plot_changed_pixels_vs_time(worms, worm_names=None, output_dir=None, show_plot=True):
    """
    Plot changed pixels vs. time for individual worms and overwrite existing plots if needed.

    Args:
        worms (list): List of DataFrames, each representing one worm.
        worm_names (list, optional): List of worm names for labeling the plots. Defaults to None.
        output_dir (str, optional): Directory to save the plots. If None, plots are not saved.
        show_plot (bool): Whether to display the plots interactively. Defaults to True.

    Returns:
        None
    """
    if worm_names is None:
        worm_names = [f"Worm {i+1}" for i in range(len(worms))]

    for i, (worm, worm_name) in enumerate(zip(worms, worm_names)):
        # Ensure "Frame" and "Changed Pixels" columns exist
        if 'Frame' not in worm.columns or 'Changed Pixels' not in worm.columns:
            print(f"Skipping {worm_name} as required columns are missing.")
            continue

        # Plot Changed Pixels vs. Frame
        plt.figure(figsize=(10, 6))
        plt.plot(worm['Frame'], worm['Changed Pixels'], label="Changed Pixels", color='blue', alpha=0.7)
        plt.xlabel("Frame (Time)", fontsize=12)
        plt.ylabel("Changed Pixels", fontsize=12)
        plt.title(f"Changed Pixels vs. Time for {worm_name}", fontsize=14)
        plt.legend()

        # Save plot if output_dir is specified
        if output_dir:
            save_path = os.path.join(output_dir, f"{worm_name}_changed_pixels_vs_time.png")

            # Overwrite the file if it already exists
            # if os.path.exists(save_path):
            #     print(f"Overwriting existing plot: {save_path}")
            # else:
            #     print(f"Saving new plot: {save_path}")

            plt.savefig(save_path)

        # Show plot if required
        # if show_plot:
        #     plt.show()

        # Close the plot to avoid overlapping
        plt.close()

def print_fdict_summary(fdict):
    """
    Print a summary of the files loaded into fdict.

    Args:
        fdict (dict): Dictionary where keys are worm names and values are numpy arrays.
    """
    print("\nSummary of Loaded Worm Data:")
    print("=" * 40)
    for worm_name, data_array in fdict.items():
        print(f"Worm: {worm_name}")
        print(f"  Shape: {data_array.shape}")
        print("-" * 40)



def standardization(processed_worms, feature_columns):
    """
    Perform per-worm standardization.

    Args:
        processed_worms (dict): Dictionary of worm data (features, frames).
        feature_columns (list): Indices of the columns to standardize.

    Returns:
        dict: Dictionary of standardized worms.
    """
    standardized_worms = {}

    for worm_name, worm_data in processed_worms.items():
        standardized_data = worm_data.copy()
        features = worm_data[feature_columns, :]  # Extract selected features

        # Calculate mean and std for each feature of the worm
        worm_mean = features.mean(axis=1, keepdims=True)
        worm_std = features.std(axis=1, keepdims=True)

        # Standardize selected features
        standardized_data[feature_columns, :] = (features - worm_mean) / worm_std
        standardized_worms[worm_name] = standardized_data

    return standardized_worms