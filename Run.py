from Visualisation.Visualisation_utils import plot_trajectory, load_worm_data
import os

# Path to the CSV file of the control worm 11
# check if path is valid 
file_path = "/Users/louistschanz/Documents/EPFL-Cours/MA1/ML/Project-2/ML-Project-2/Data/Lifespan/control/speeds_and_coordinates_20241016_11_updated.csv"
if os.path.exists(file_path):
    print("File path is valid.")
else:
    print("File not found. Check the path:", file_path)

worm_data = load_worm_data(file_path)
plot_trajectory(worm_data, worm_id="Control Worm 1")
