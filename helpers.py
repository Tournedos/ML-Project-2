import pandas as pd
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


def load_one_data(file_path, worm_id):
    """
    Load worm data from one CSV file and validate the file path.

    Args:
        file_path (str): Path to the worm data CSV file.
        worm_id (str): Identifier for the worm (e.g., "Control Worm 11").

    Returns:
        pd.DataFrame: Loaded worm data if the path is valid, None otherwise.
    """
    if not os.path.exists(file_path):
        print(f"{worm_id}: File not found. Check the path: {file_path}")
        return None

    print(f"{worm_id}: File path is valid. Loading data...")
    worm_data = pd.read_csv(file_path)
    print(f"{worm_id}: Data loaded successfully.")
    return worm_data


# def split_worms(data_path, test_size=0.2, random_state=42):
#     """
#     Splits worms into separate training and testing datasets for control and drugged groups,
#     while converting .xlsx files to .csv.

#     Args:
#         data_path (str): Path to the Lifespan folder containing control and companyDrug subfolders.
#         test_size (float): Proportion of worms to include in the test split.
#         random_state (int): Random seed for reproducibility.

#     Returns:
#         dict: A dictionary containing train and test splits for control and drugged groups.
#     """
#     worm_data = {'control': [], 'companyDrug': []}

#     # Subfolders for control and companyDrug
#     subfolders = ['control', 'companyDrug']

#     for subfolder in subfolders:
#         subfolder_path = os.path.join(data_path, subfolder)

#         # Check if the subfolder exists
#         if not os.path.exists(subfolder_path):
#             print(f"Warning: {subfolder_path} does not exist. Skipping...")
#             continue

#         # Process all .csv and .xlsx files in the subfolder
#         for filename in os.listdir(subfolder_path):
#             file_path = os.path.join(subfolder_path, filename)

#             if filename.endswith(".csv"):
#                 worm_data[subfolder].append(file_path)

#             elif filename.endswith(".xlsx"):
#                 # Convert .xlsx to .csv
#                 try:
#                     df = pd.read_excel(file_path)  # Read the .xlsx file
#                     csv_path = file_path.replace(".xlsx", ".csv")  # Change file extension to .csv
#                     df.to_csv(csv_path, index=False)  # Save as .csv
#                     print(f"Converted {file_path} to {csv_path}.")
#                     worm_data[subfolder].append(csv_path)  # Add the .csv path to the dataset
#                 except Exception as e:
#                     print(f"Failed to convert {file_path}: {e}")

#     # Separate splits for control and drugged groups
#     splits = {}
#     for group, files in worm_data.items():
#         train_files, test_files = train_test_split(files, test_size=test_size, random_state=random_state)
#         splits[group] = {
#             'train': train_files,
#             'test': test_files
#         }
#         print(f"{group}: {len(train_files)} training files, {len(test_files)} testing files.")

#     return splits


def load_file_data(category_path):
    """
    Load data for a specific category (control or companyDrug).

    Args:
        category_path (str): Path to the folder containing worm CSV files.

    Returns:
        list: A list of DataFrames, each corresponding to one worm.
    """
    # Initialize an empty list to hold worm DataFrames
    worms = []

    # Check if the path exists
    if not os.path.exists(category_path):
        raise FileNotFoundError(f"Path does not exist: {category_path}")

    # Loop through all CSV files in the directory
    for filename in os.listdir(category_path):
        if filename.endswith(".csv"):  # Only process CSV files
            file_path = os.path.join(category_path, filename)
            print(f"Loading file: {file_path}")
            df = pd.read_csv(file_path)
            worms.append(df)

    print(f"Loaded {len(worms)} worms from {category_path}")
    return worms


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


# def adjust_time(df):
#     """
#     Adjust the 'Frame' column to calculate the absolute time in seconds,
#     accounting for 6-hour sessions with 10800 frames per session.

#     Args:
#         df (pd.DataFrame): Worm data with 'Frame' column.

#     Returns:
#         pd.DataFrame: DataFrame with an added 'Absolute Time' column.
#     """
#     session_counter = 0
#     absolute_times = []

#     for i, frame in enumerate(df['Frame']):
#         # Detect frame reset
#         if i > 0 and frame < df['Frame'].iloc[i - 1]:
#             session_counter += 1  # Increment session counter

#         # Calculate absolute time
#         absolute_time = session_counter * 21600 + (frame - 1) * 2
#         absolute_times.append(absolute_time)

#     df['Absolute Time (s)'] = absolute_times
#     return df



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


def load_earlylifespan(worms, data_fraction=0.2):
    """
    Load only the early lifespan data for each worm.

    Args:
        worms (list): List of DataFrames, each representing one worm.
        data_fraction (float): Fraction of the lifespan to retain (e.g., 0.2 for the first 20%).

    Returns:
        list: A list of truncated DataFrames.
    """
    truncated_worms = []

    for worm in worms:
        # Calculate the number of rows to keep
        rows_to_keep = int(len(worm) * data_fraction)
        truncated_worm = worm.head(rows_to_keep)
        truncated_worms.append(truncated_worm)

    print(f"Truncated data to {data_fraction * 100}% of the lifespan for {len(truncated_worms)} worms.")
    return truncated_worms



def plot_changed_pixels(worms, worm_names=None, output_dir=None, show_plot=True):
    """
    Plot changed pixels vs. time for individual worms.

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
            plt.savefig(save_path)
            print(f"Plot saved for {worm_name} at {save_path}")

        # Show plot if required
        if show_plot:
            plt.show()

        # Close the plot to avoid overlapping
        plt.close()