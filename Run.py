from helpers import plot_trajectory, load_data, load_whole_data
import matplotlib.pyplot as plt
import os
import numpy as np
from models import mean_squared_error_gd

data_path = os.path.join(os.getcwd(), "Data", "Lifespan")

# Step 1 : Load the data
print("Loading worm data...")
X_train, X_test, y_train, y_test = load_whole_data(data_path=data_path, test_size=0.2, random_state=42)
print("Data loaded successfully!")


# Step 2: Initialize parameters
initial_w = np.zeros(X_train.shape[1])  # Initialize weights as zeros
max_iters = 1000  # Number of iterations
gamma = 0.0001  # Learning rate

# Step 2.2 : Check the shapes before matrix multiplication
print(f"Shape of X_train: {X_train.shape}")
print(f"Shape of y_train: {y_train.shape}")
print(f"Shape of initial_w: {initial_w.shape}")

# Step 3 : Run the model
w, loss = mean_squared_error_gd(y_train, X_train, initial_w, max_iters, gamma)

# Step 4: Print the results

# Personality
import pandas as pd
from personality import calculate_speed_stats

# Example: Load your data
repor_root = find_repo_root()
control_path = os.path.join(repor_root, 'Data/Lifespan/control')
control_worms = load_file_data(control_path)  # Assuming this loads a list of DataFrames

# Process each worm's data
for i, worm_data in enumerate(control_worms):
    speed_stats = calculate_speed_stats(worm_data)

