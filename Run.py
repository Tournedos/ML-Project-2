from Visualisation.Visualisation_utils import plot_trajectory, load_worm_data
import matplotlib.pyplot as plt
import os

# Control worm 11
# check if path is valid 
control_path_11 = "/Users/louistschanz/Documents/EPFL-Cours/MA1/ML/Project-2/ML-Project-2/Data/Lifespan/control/speeds_and_coordinates_20241016_11_updated.csv"
if os.path.exists(control_path_11):
    print("File path is valid.")
else:
    print("File not found. Check the path:", control_path_11)

worm_data = load_worm_data(control_path_11)
plt.figure()
plot_trajectory(worm_data, worm_id="Control Worm 11")
plt.close()


# CompanyDrug Worm 1
drug_path_1 = "/Users/louistschanz/Documents/EPFL-Cours/MA1/ML/Project-2/ML-Project-2/Data/Lifespan/companyDrug/speeds_and_coordinates_20241016_1_updated.csv"
if os.path.exists(drug_path_1):
    print("File path is valid.")
else:
    print("File not found. Check the path:", drug_path_1)

worm_data = load_worm_data(drug_path_1)
plt.figure()
plot_trajectory(worm_data, worm_id="CompanyDrug Worm 1")
plt.close()





