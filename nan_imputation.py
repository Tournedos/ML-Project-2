import numpy as np

def impute_nan(x):
    """
    Handles NaN values in worm data for X, Y, and Speed columns.
    - If X and Y are NaN, drop the row.
    - Impute any remaining NaN values with column means.

    Args:
        df (pd.DataFrame): Input DataFrame with 'X', 'Y', and 'Speed' columns.

    Returns:
        pd.DataFrame: DataFrame with NaN values handled.
    """
    # Step 1: Drop rows where both X and Y are NaN
    df = df.dropna(subset=['X', 'Y'])

    # Step 2: Impute any remaining NaN values with column means
    df.fillna(df.mean(), inplace=True)

    return x


def count_successive_missing(array):
    """
    Identifies sequences of consecutive NaN values in a 2D array.

    Args:
        array (np.ndarray): 2D array where rows represent features and columns represent time points.

    Returns:
        list of tuples: Each tuple contains (start_index, end_index, length) of consecutive NaN sequences.
    """
    missing_mask = np.isnan(array)  # Create a boolean mask where NaN is True
    successive_counts = np.diff(
        np.concatenate(([False], missing_mask.any(axis=0), [False])).astype(int)
    )
    start_indices = np.where(successive_counts == 1)[0]  # Start of NaN sequences
    end_indices = np.where(successive_counts == -1)[0]  # End of NaN sequences
    lengths = end_indices - start_indices  # Lengths of NaN sequences
    return list(zip(start_indices, end_indices, lengths))

def cut_array(array, rows_to_check):
    """
    Filters a NumPy array to retain only columns without NaNs in the specified rows.
    
    Args:
        array (np.ndarray): Input 2D NumPy array containing data for a single worm.
        rows_to_check (slice): Rows to check for NaN values (e.g., rows for X and Y coordinates).

    Returns:
        np.ndarray: A filtered 2D array with only valid columns (no NaNs in the specified rows).
    """
    # Check for NaNs in the specified rows
    # `~np.isnan(...)`: Creates a boolean mask where True means "not NaN"
    # `.any(axis=0)`: Checks if any NaN exists in each column of the specified rows
    # `~`: Negates the boolean mask to identify valid (non-NaN) columns
    missing_mask = ~np.isnan(array[rows_to_check, :]).any(axis=0)  
    
    # Use the mask to filter columns, retaining all rows but excluding invalid columns
    return array[:, missing_mask]