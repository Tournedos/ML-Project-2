import pandas as pd

def calculate_speed_stats(data):
    """
    Calculate the average and variance of speed for the given dataset.

    Args:
        data (pd.DataFrame): A DataFrame containing worm data. Must include a 'Speed' column.

    Returns:
        dict: A dictionary with keys 'average_speed' and 'variance_speed', containing the calculated values.
    """
    if 'Speed' not in data.columns:
        raise ValueError("The provided data does not contain a 'Speed' column.")

    # Calculate speed statistics
    average_speed = data['Speed'].mean()
    variance_speed = data['Speed'].var()

    return {
        'average_speed': average_speed,
        'variance_speed': variance_speed
    }
