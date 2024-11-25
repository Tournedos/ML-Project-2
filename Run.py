from helpers import plot_trajectory, load_data, load_whole_data
import matplotlib.pyplot as plt
import os
import numpy as np
from models import mean_squared_error_gd

# >> 1st part to load csc files separately from 1 worm (useless but good to visualise it's position for ex
# # Control worm 11
# # check if path is valid 
# control_path_11 = "/Users/louistschanz/Documents/EPFL-Cours/MA1/ML/Project-2/ML-Project-2/Data/Lifespan/control/speeds_and_coordinates_20241016_11_updated.csv"
# if os.path.exists(control_path_11):
#     print("File path is valid.")
# else:
#     print("File not found. Check the path:", control_path_11)

# worm_data = load_worm_data(control_path_11)
# plt.figure()
# plot_trajectory(worm_data, worm_id="Control Worm 11")
# plt.close()

# # CompanyDrug Worm 1
# drug_path_1 = "/Users/louistschanz/Documents/EPFL-Cours/MA1/ML/Project-2/ML-Project-2/Data/Lifespan/companyDrug/speeds_and_coordinates_20241016_1_updated.csv"
# if os.path.exists(drug_path_1):
#     print("File path is valid.")
# else:
#     print("File not found. Check the path:", drug_path_1)

# worm_data = load_worm_data(drug_path_1)
# plt.figure()
# plot_trajectory(worm_data, worm_id="CompanyDrug Worm 1")
# plt.close()

# tasks = [
#         {
#             "file_path": "/Users/louistschanz/Documents/EPFL-Cours/MA1/ML/Project-2/ML-Project-2/Data/Lifespan/control/speeds_and_coordinates_20241016_11_updated.csv",
#             "worm_id": "Control Worm 11",
#             "output_file": "control_worm_11.png",
#         },
#         {
#             "file_path": "/Users/louistschanz/Documents/EPFL-Cours/MA1/ML/Project-2/ML-Project-2/Data/Lifespan/companyDrug/speeds_and_coordinates_20241016_1_updated.csv",
#             "worm_id": "CompanyDrug Worm 1",
#             "output_file": "company_drug_worm_1.png",
#         },
#     ]

# # Process each task
# for task in tasks:
#     worm_data = load_data(task["file_path"], task["worm_id"])
#     if worm_data is not None:
#         plot_trajectory(worm_data, task["worm_id"], task["output_file"], show_plot=True, save_plot=False)



# >> 2nd part to load all the data from the lifespan folder and try 1st model on it
# Define the path to the Lifespan folder
data_path = "/Users/louistschanz/Documents/EPFL-Cours/MA1/ML/Project-2/ML-Project-2/Data/Lifespan"

# Step 1 : Load the data
print("Loading worm data...")
X_train, X_test, y_train, y_test = load_whole_data(data_path=data_path, test_size=0.2, random_state=42)
print("Data loaded successfully!")

# # Display dataset information
# print("\nDataset Information:")
# print(f"Training set size: {len(X_train)} samples")
# print(f"Testing set size: {len(X_test)} samples")
# print("\nFirst few rows of training features:")
# print(X_train.head())
# print("\nFirst few labels of training data:")
# print(y_train.head())


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
print(f"Final Weights: {w}")
print(f"Final MSE Loss: {loss}")