import os
import numpy as np
from helpers import *
from models import *   
from nan_imputation import impute_nan
import random

def load_lifespan(pathin):
    """
    Load all worms from multiple categories into a unified structure.
    Ensures data is in the format (features, frames).

    Args:
        pathin (str): Root path to the Lifespan directory.

    Returns:
        dict: A dictionary where keys are worm names (worm_1, worm_2, ...) and values 
              are NumPy arrays with the worm data and a `Category` column.
    """
    subfolders = {'control': 0,'companyDrug': 1,'controld2': 2}   #'Terbinafin': 3 Removed Terebafin for now
    worms = {}  # Unified dictionary for all worms
    worm_id = 1  # Worm numbering starts at 1

    for subf, category in subfolders.items():
        subfp = os.path.join(pathin, subf)
        if not os.path.exists(subfp):  # Skip if the subfolder doesn't exist
            print(f"Warning: Subfolder does not exist: {subfp}")
            continue
    
        filenms = [f for f in os.listdir(subfp) if f.endswith('.csv')]  # Load only CSV files

        for name in filenms:
            filepath = os.path.join(subfp, name)
            try:
                # Read the CSV
                data_raw = pd.read_csv(filepath, sep=',')
            except Exception as e:
                print(f"Failed to load {filepath}: {e}")
                continue

            # Add the numerical category as a column
            data_raw['Category'] = category

            # Convert to NumPy array and transpose to (features, frames)
            data_array = np.array(data_raw.apply(pd.to_numeric)).T
            worms[f'worm_{worm_id}_{subf}'] = data_array  # Add worm to the dictionary
            worm_id += 1  # Increment worm ID

    return worms


def load_optogenetics(pathin):
    """
    Load all worms from both categories (companyDrug and control) into a single structure.
    Ensures data is in the format (features, frames).

    Args:
        pathin (str): Root path to the Lifespan directory.

    Returns:
        dict: A dictionary where keys are worm names (worm_1, worm_2, ...) and values 
              are NumPy arrays with the worm data and a `Category` column.
    """
    subfolders = {'ATR-': 0, 'ATR+': 1}  # Folder names and their categories
    worms = {}  # Unified dictionary for all worms
    worm_id = 1  # Worm numbering starts at 1

    for subf, category in subfolders.items():
        subfp = os.path.join(pathin, subf)
        filenms = [f for f in os.listdir(subfp) if f.endswith('.csv')]  # Load only CSV files

        for name in filenms:
            filepath = os.path.join(subfp, name)
            #print(f"Loading file: {filepath}")
            try:
                # Read the CSV
                data_raw = pd.read_csv(filepath, sep=',')
                # print(f"File loaded: {filepath}")
                # print(data_raw.head())  # Debug raw data
            except Exception as e:
                print(f"Failed to load {filepath}: {e}")
                continue

            # Add the category column
            data_raw['Category'] = category

            # Convert to NumPy array and transpose to (features, frames)
            data_array = np.array(data_raw.apply(pd.to_numeric)).T
            worms[f'worm_{worm_id}'] = data_array  # Add worm to the dictionary
            worm_id += 1  # Increment worm ID

    #print(f"Loaded {len(worms)} worms with categories as a feature.")
    return worms


def load_earlylifespan(worms, data_fraction=0.2):
    """
    Load only the early lifespan data for each worm.

    Args:
        worms (dict): Dictionary where keys are worm names and values are NumPy arrays.
        data_fraction (float): Fraction of the lifespan to retain (e.g., 0.2 for the first 20%).

    Returns:
        dict: A dictionary of truncated worms with the same structure as the input.
    """
    truncated_worms = {}

    for worm_name, worm_data in worms.items():
        # Calculate the number of columns to keep (frames)
        cols_to_keep = int(worm_data.shape[1] * data_fraction)
        truncated_worm = worm_data[:, :cols_to_keep]  # Keep only the first `cols_to_keep` columns
        truncated_worms[worm_name] = truncated_worm

    print(f"Truncated data to {data_fraction * 100}% of the lifespan for {len(truncated_worms)} worms.")
    return truncated_worms


def load_one_data(file_path, worm_id):
    """
    Load worm data from ONE CSV file and validate the file path.

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


def split_worms(worms, test_size=0.2, random_seed=42):
    """
    Split worms into training and testing sets.

    Args:
        worms (dict): Dictionary of worms, where keys are worm names and values are NumPy arrays.
        test_size (float): Proportion of the worms to include in the test set (e.g., 0.2 for 20% test worms).
        random_seed (int): Seed for reproducibility of the random split.

    Returns:
        tuple: (train_worms, test_worms) where each is a dictionary of worms.
    """
    # Ensure reproducibility
    random.seed(random_seed)

    # Get a list of worm names
    worm_names = list(worms.keys())

    # Shuffle the worm names
    random.shuffle(worm_names)

    # Determine the split index
    split_index = int(len(worm_names) * (1 - test_size))

    # Split the worm names into training and testing sets
    train_names = worm_names[:split_index]
    test_names = worm_names[split_index:]

    # Create dictionaries for training and testing worms
    train_worms = {name: worms[name] for name in train_names}
    test_worms = {name: worms[name] for name in test_names}

    print(f"Split {len(worms)} worms into {len(train_worms)} training and {len(test_worms)} testing worms.")

    return train_worms, test_worms


def prepare_training_data(worms):
    """
    Prepares the features (X) and targets (y) for training or testing.

    Args:
        worms (dict): Dictionary where keys are worm names and values are NumPy arrays of worm data 
                      with the shape (features, frames).

    Returns:
        tuple: (X, y) where:
               - X is a dictionary of features (NumPy arrays with shape (features, frames)).
               - y is a dictionary of lifespans (total number of frames per worm).
    """
    X = {}
    y = {}

    for worm_name, worm_data in worms.items():
        # Features: Keep all rows except the first one (Frame)
        features = worm_data[1:, :]  # Remove the first row (Frame)
        X[worm_name] = features  # Shape remains (features, frames)


        # Target: Lifespan (total number of frames)
        lifespan = worm_data.shape[1]  # The number of columns represents the lifespan
        y[worm_name] = lifespan

    return X, y


def truncate_lifespan(worms, data_fraction=0.4):
    """
    Truncate each worm's data to a fraction of its lifespan.

    Args:
        worms (dict): Dictionary where keys are worm names and values are NumPy arrays of shape (features, frames).
        data_fraction (float): Fraction of the lifespan to retain (e.g., 0.4 for the first 40%).

    Returns:
        dict: A dictionary of worms truncated to the specified fraction of their lifespan.
    """
    truncated_worms = {}

    for worm_name, worm_data in worms.items():
        original_frames = worm_data.shape[1]  # Total number of frames
        cols_to_keep = int(original_frames * data_fraction)  # Calculate how many columns to keep
        truncated_worm = worm_data[:, :cols_to_keep]  # Retain only the first `cols_to_keep` columns
        truncated_worms[worm_name] = truncated_worm  # Store the truncated data

        print(f"{worm_name}: Original frames = {original_frames}, Truncated frames = {cols_to_keep}")

    return truncated_worms



