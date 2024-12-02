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


def load_data(file_path, worm_id):
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


def split_worms(data_path, test_size=0.2, random_state=42):
    """
    Splits worms into separate training and testing datasets for control and drugged groups,
    while converting .xlsx files to .csv.

    Args:
        data_path (str): Path to the Lifespan folder containing control and companyDrug subfolders.
        test_size (float): Proportion of worms to include in the test split.
        random_state (int): Random seed for reproducibility.

    Returns:
        dict: A dictionary containing train and test splits for control and drugged groups.
    """
    worm_data = {'control': [], 'companyDrug': []}

    # Subfolders for control and companyDrug
    subfolders = ['control', 'companyDrug']

    for subfolder in subfolders:
        subfolder_path = os.path.join(data_path, subfolder)

        # Check if the subfolder exists
        if not os.path.exists(subfolder_path):
            print(f"Warning: {subfolder_path} does not exist. Skipping...")
            continue

        # Process all .csv and .xlsx files in the subfolder
        for filename in os.listdir(subfolder_path):
            file_path = os.path.join(subfolder_path, filename)

            if filename.endswith(".csv"):
                worm_data[subfolder].append(file_path)

            elif filename.endswith(".xlsx"):
                # Convert .xlsx to .csv
                try:
                    df = pd.read_excel(file_path)  # Read the .xlsx file
                    csv_path = file_path.replace(".xlsx", ".csv")  # Change file extension to .csv
                    df.to_csv(csv_path, index=False)  # Save as .csv
                    print(f"Converted {file_path} to {csv_path}.")
                    worm_data[subfolder].append(csv_path)  # Add the .csv path to the dataset
                except Exception as e:
                    print(f"Failed to convert {file_path}: {e}")

    # Separate splits for control and drugged groups
    splits = {}
    for group, files in worm_data.items():
        train_files, test_files = train_test_split(files, test_size=test_size, random_state=random_state)
        splits[group] = {
            'train': train_files,
            'test': test_files
        }
        print(f"{group}: {len(train_files)} training files, {len(test_files)} testing files.")

    return splits


def load_whole_data(data_path, test_size=0.2, random_state=42):
    """
    Loads worm data (X, Y positions and speed) from the Lifespan folder,
    combines it into a single dataset, and splits it into training and testing sets.

    Args:
        data_path (str): Path to the Lifespan folder containing control and companyDrug subfolders.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Random seed for reproducibility.

    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    all_data = []
    all_labels = []

    # Subfolders for control and companyDrug
    subfolders = {'control': 0, 'companyDrug': 1}

    for subfolder, label in subfolders.items():
        subfolder_path = os.path.join(data_path, subfolder)

        # Check if the subfolder exists
        if not os.path.exists(subfolder_path):
            print(f"Warning: {subfolder_path} does not exist. Skipping...")
            continue

        # Iterate through all CSV files in the subfolder
        for filename in os.listdir(subfolder_path):
            if filename.endswith(".csv"):
                file_path = os.path.join(subfolder_path, filename)
                
                #print(f"Loading data from: {file_path}")

                # Load the CSV file into a DataFrame
                df = pd.read_csv(file_path)

                # Validate required columns are present
                if not all(col in df.columns for col in ['X', 'Y', 'Speed']):
                    print(f"Skipping {filename} as it lacks required columns.")
                    continue

                # Add the label column
                df['Label'] = label

                # Append features (X, Y, Speed) and label
                all_data.append(df[['X', 'Y', 'Speed']])
                all_labels.append(df['Label'])

    # Combine all data and labels into single DataFrames
    X = pd.concat(all_data, ignore_index=True)
    y = pd.concat(all_labels, ignore_index=True)

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    print(f"Dataset split complete: {len(X_train)} training samples, {len(X_test)} testing samples.")
    return X_train, X_test, y_train, y_test


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


def adjust_time(df):
    """
    Adjust the 'Frame' column to calculate the absolute time in seconds,
    accounting for 6-hour sessions with 10800 frames per session.

    Args:
        df (pd.DataFrame): Worm data with 'Frame' column.

    Returns:
        pd.DataFrame: DataFrame with an added 'Absolute Time' column.
    """
    session_counter = 0
    absolute_times = []

    for i, frame in enumerate(df['Frame']):
        # Detect frame reset
        if i > 0 and frame < df['Frame'].iloc[i - 1]:
            session_counter += 1  # Increment session counter

        # Calculate absolute time
        absolute_time = session_counter * 21600 + (frame - 1) * 2
        absolute_times.append(absolute_time)

    df['Absolute Time (s)'] = absolute_times
    return df


