import numpy as np
from helpers_model import gradient_descent
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Performs linear regression using gradient descent (GD) to minimize the Mean Squared Error (MSE) loss function.

    This function iteratively updates the model parameters `w` by calculating the gradient of the MSE loss with respect to
    `w` and adjusting `w` in the opposite direction of the gradient, effectively minimizing the error over the dataset.

    Args:
        y (np.ndarray): Array of target values with shape (N,).
        tx (np.ndarray): Input data matrix with shape (N, D), where N is the number of samples and D is the number of features.
        initial_w (np.ndarray): Initial parameter vector with shape (D,), representing the starting values for the model parameters.
        max_iters (int): Number of iterations to run the gradient descent algorithm.
        gamma (float): Learning rate, controlling the size of each gradient update step.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: Final parameter vector `w` of shape (D,) after convergence.
            - float: Final Mean Squared Error (MSE) value at the last iteration.

    Notes:
        - The function calls `gradient_descent`, which performs the iterative updates to optimize the parameters.
        - This function is suited for datasets of manageable size, where batch processing is feasible.
    """
    # Perform gradient descent to optimize the weights
    w, loss = gradient_descent(y, tx, initial_w, max_iters, gamma)

    # Return the final weights and the final loss
    return w, loss

def aggregate_features(worms, window_size=1000):
    """
    Aggregate features by dividing frames into windows and computing statistics per window.

    Args:
        worms (dict): Dictionary where keys are worm names and values are (frames, features).
        window_size (int): Number of frames per window.

    Returns:
        tuple: X (aggregated features), y (lifespans).
    """
    X = []
    y = []
    
    for worm_name, worm_data in worms.items():
        features = worm_data[:, 1:]  # Exclude Frame number (column 0)
        n_frames = features.shape[0]
        
        # Skip worms with fewer frames than the window size
        if n_frames < window_size:
            print(f"Skipping {worm_name}: Not enough frames ({n_frames} frames, window size = {window_size}).")
            continue
        
        # Calculate the number of complete windows
        n_windows = n_frames // window_size
        trimmed_features = features[:n_windows * window_size]
        
        # Reshape into windows and compute stats
        reshaped = trimmed_features.reshape(n_windows, window_size, -1)  # (windows, frames_per_window, features)
        stats = np.hstack([reshaped.mean(axis=1), reshaped.std(axis=1), reshaped.min(axis=1), reshaped.max(axis=1)])
        
        # Flatten stats for all windows into one vector
        aggregated = stats.flatten()
        X.append(aggregated)
        y.append(worm_data[:, 0].max())  # Lifespan is the max Frame number
    
    return np.array(X, dtype=object), np.array(y)
