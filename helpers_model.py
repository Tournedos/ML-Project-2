import numpy as np

def compute_loss(y, tx, w):
    """Calculate the mean squared error (MSE) loss.

    Args:
        y (numpy.ndarray): Target values of shape (N,).
        tx (numpy.ndarray): Input data of shape (N, D).
        w (numpy.ndarray): Model parameters of shape (D,).

    Returns:
        float: The value of the MSE loss.
    """
    # Compute the error vector (difference between actual and predicted values)
    error = y - tx.dot(w)

    # Compute the mean squared error (MSE) loss
    mse = np.mean(error**2) / 2

    return mse

def compute_gradient(y, tx, w):
    """Compute the gradient of the loss.

    Args:
        y (numpy.ndarray): Target values of shape (N,).
        tx (numpy.ndarray): Input data of shape (N, D).
        w (numpy.ndarray): Model parameters of shape (D,).

    Returns:
        numpy.ndarray: Gradient of the loss with respect to w, of shape (D,).
    """
    # Compute the error vector (difference between actual and predicted values)
    error = y - tx.dot(w)

    # Compute the gradient of the loss function
    gradient = -tx.T.dot(error) / y.size

    return gradient


def gradient_descent(y, tx, initial_w, max_iters, gamma):
    """Perform gradient descent optimization.

    Args:
        y (numpy.ndarray): Target values of shape (N,).
        tx (numpy.ndarray): Input data of shape (N, D).
        initial_w (numpy.ndarray): Initial guess for the model parameters of shape (D,).
        max_iters (int): Total number of iterations for gradient descent.
        gamma (float): Step size (learning rate) for gradient updates.

    Returns:
        numpy.ndarray: Final weight vector of shape (D,).
        float: Final loss (MSE) value.
    """
    # Initialize lists to store weights and losses at each iteration
    ws = [initial_w]
    initial_loss = compute_loss(y, tx, initial_w)
    losses = [initial_loss]
    w = initial_w

    # Iterate over the number of iterations
    for n_iter in range(max_iters):
        # Compute the gradient of the loss function
        gradient = compute_gradient(y, tx, w)
        # Update the weights using the gradient and learning rate
        w -= gamma * gradient
        # Compute the loss with the updated weights
        loss = compute_loss(y, tx, w)
        # Store the updated weights and loss
        ws.append(w)
        losses.append(loss)

        # Print the current iteration, loss, and weights
        #print(f"GD iter. {n_iter}/{max_iters - 1}: loss={loss}, w0={w[0]}, w1={w[1]}")

    # Compute the final loss
    loss = compute_loss(y, tx, w)
    return w, loss

