import numpy as np
from helpers_model import gradient_descent

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

