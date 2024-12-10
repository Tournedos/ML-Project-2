import os
import numpy as np
from helpers import *
from models import *   
from nan_imputation import impute_nan

def load_lifespan(pathin):
    subfolders = ['Lifespan/control','Lifespan/companyDrug', 'Optogenetics/ATR+','Optogenetics/ATR-']
    feat_dict = {} #save as dictionary because the arrays will have different lenghts
    for subf in subfolders:
        subfp = os.path.join(pathin, subf)
        filenms = os.listdir(subfp)
        for cont, name in enumerate(filenms): #keep the count to have unique names of worms
            filepath = os.path.join(subfp,name)
            print(filepath)
            try:
                # Try reading as a CSV. this because we noticed that .xlsx files are just copies of csv ones, so try .csv and if not succeed just continue
                data_raw = pd.read_csv(filepath, sep=',') #put right separator
                print("File loaded as CSV.")
            except Exception as e:
                continue #if file is not .csv ignore it
            sample_name = 'worm_' + str(cont) + '_' + subf # Unique name for each worm
            data_n = data_raw.apply(pd.to_numeric) # Attempts to convert all columns in the dataframe to numeric values to use them in calculations.
            feat_dict.update({sample_name: np.array(data_n).T}) # Converts the processed dataframe into a NumPy array and transposes it, then adds to the dictionary with its key
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