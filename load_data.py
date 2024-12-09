import os
import numpy as np
from helpers import *
from models import *   
from nan_imputation import impute_nan

def load_lifespan(pathin):
    subfolders = ['control','companyDrug']
    feat_dict = {}
    for subf in subfolders:
        subfp = os.path.join(pathin, subf)
        filenms = os.listdir(subfp)
        for cont, name in enumerate(filenms): # cont : counter for indeing worms
            filepath = os.path.join(subfp,name)
            print(filepath)
            try: # Try reading as a CSV
                data_raw = pd.read_csv(filepath, sep=',')
                print("File loaded as CSV.")
            except Exception as e:
                continue
                #print(f"Failed to load as CSV. Error: {e}")
                #try:
                #    # If CSV fails, try reading as Excel
                #    data_raw = pd.read_excel(filepath)
                #    print("File loaded as Excel.")
                #except Exception as e:
                #    print(f"Failed to load as Excel. Error: {e}")
                #    raise ValueError("File could not be loaded. Please check the format.")
            #print(np.array(data_raw.head()).T)
            sample_name = 'worm_' + str(cont) + '_' + subf # Unique name for each worm 
            data_n = data_raw.apply(pd.to_numeric) # Attempts to convert all columns in the dataframe to numeric values, ensuring consistency.
            feat_dict.update({sample_name: np.array(data_n).T}) # Converts the processed dataframe into a NumPy array and transposes it 
    return feat_dict

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