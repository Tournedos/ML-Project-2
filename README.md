#  README.md

# C-elegans behavioral analysis 
C-elegans are worms often studied in the lab. By behavioral analysis we mean analysis of their lifespan.

1. # Prerequisites

- Python 3.8+
- Required Libraries**:
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `scikit-survival`
  - `scikit-learn`

To install dependencies, run :
pip install numpy pandas matplotlib scikit-survival scikit-learn


2. # Running the Project
2.1	Clone the repository:
git clone https://github.com/Tournedos/ML-Project-2.git
2.2	Navigate to the project folder:
cd worm-lifespan-prediction
2.3 Open and run the notebook:
jupyter notebook run.ipynb


3. # Overview of Key Functions
•	run.ipynb
The main notebook that integrates all steps, from data loading and preprocessing to analysis and visualization.

•	helpers.py
Contains utility functions used throughout the project.

•	models.py
Includes machine learning models used for predictions.

•	nan_imputation.py
Provides functions for handling missing values in the data.

•	Preprocessing.py
Handles general preprocessing tasks.

•	preprocessing_features.py
Focuses on feature-specific preprocessing, such as scaling and extraction.

•	load_data.py
Includes functions like load_lifespan and load_earlylifespan for loading datasets.