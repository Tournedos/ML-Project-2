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