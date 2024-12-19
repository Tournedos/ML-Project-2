
# C-elegans Lifespan Prediction - Final Notebook 

This notebook, run.ipynb, is the final implementation for predicting worm lifespan based on early behavioral data. It integrates data preprocessing, feature engineering, machine learning models, and evaluation techniques to produce accurate lifespan predictions.

# Table of Contents
1.	Project Overview
2.	Prerequies
3.  Overview of key functions
4.	Notebook Workflow
5.	Results and Outputs



1. Project Overview
The project predicts the lifespan of worms using behavioral time-course data from laboratory experiments. The behavioral features include center-of-mass coordinates, speed, and other derived metrics, allowing the training of machine learning models to make lifespan predictions.
### add part for Optogenetics (check what we have done so far)
This notebook is self-contained, calling modularized functions from external files for efficient computation and analysis.


2. #  Prerequies
- Python 3.8+
- Required Libraries**:
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `scikit-survival`
  - `scikit-learn`

# How to run
-   To install dependencies, run :
pip install numpy pandas matplotlib scikit-survival scikit-learn
-   Clone the repository:
git clone https://github.com/Tournedos/ML-Project-2.git
-   Navigate to the project folder:
cd worm-lifespan-prediction
-   Open and run the notebook:
jupyter notebook run.ipynb


3. # Overview of Key Functions
•	run.ipynb : The main notebook that integrates all steps, from data loading and preprocessing to analysis and visualization.
•	helpers.py : Contains utility functions used throughout the project.
•	models.py : Includes machine learning models used for predictions.
•	nan_imputation.py : Provides functions for handling missing values in the data.
•	Preprocessing.py : Handles general preprocessing tasks to clean the data.
•	preprocessing_features.py : Focuses on feature-specific preprocessing, such as scaling and extraction. Calculates new features based on the basic features that comes with the data
•	load_data.py : Includes functions like load_lifespan and load_earlylifespan for loading datasets.


4. # Notebook Workflow
4.1 Setup : 
-   import libraries and helper functions
-   load the worm lifespan data and optogenetics data
4.2 Data Preprocessing : 
-   Cleans data by imputing NaNs
-   Standardizes and prepares features for modeling
4.3 Feature Engineering:
-   Extracts early behavior metrics from the raw data.
-   Constructs datasets for regression and classification tasks.
4.4. Model Training and Evaluation:
-   Trains machine learning models to predict lifespan.
-   Evaluates model performance using metrics and visualizations.
4.5. Results Analysis:
-   Visualizes survival curves using Kaplan-Meier estimators.
-   Analyzes the relationship between features and lifespan predictions.


5. # Results and Outputs
•	Predictions:
	Provides lifespan predictions for worms based on their early behavior.
•	Visualizations:
	Includes Kaplan-Meier survival curves and other plots for understanding model performance.
•	Performance Metrics:
	Reports accuracy, RMSE, and survival analysis metrics.