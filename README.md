
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
The notebook run.ipynb is structured to guide you through the complete process of worm lifespan prediction.
 
Part 1 : 1. Lifespan prediction based on early behavior
1) Setup : 
-   Import libraries (numpy,pandas...) and custom modules (helpers.py, models.py...)
-   The root directory and data paths are set up for seamless data loading.
2) Data loading :
-    Load lifespan data, make sure of proper loading (only csc files)
3) Data Preprocessing : 
-   Cleans data by imputing NaNs
-   Remove frames where the worms are detected to be dead.
-   Standardizes to prepares features for modeling.
4) Feature Engineering:
-   Extracts early behavior metrics from the raw data.
-   Constructs datasets for regression and classification tasks.
5) Model Training and Evaluation:
-   Trains machine learning models to predict lifespan, using early behavioral features.
-   Evaluates model performance using metrics and visualizations.
6) Results Analysis:
-   Analyze predictions against ground truth using metrics like RMSE and accuracy.
-   Kaplan-Meier curves for survival analysis.
-   Error histograms for lifespan prediction models.



Part 2 : Assessment of personality of worms based on early behavior
1) Setup :
-   make any additionnal needed imports
-   load data, specifically Optogenetics file this time
2) Preprocessing optogenetics data :
-   NaN imputation
3) Feature Engeneering :
-   Derive personality metrics from early movement patterns such as consistency in movement and preferred activity levels
4) Clustering Analysis :
-   Perform clustering to group worms based on similar behavioral traits
-   Visualize clusters to identify distinct personality types
5) Behavioral Traits Evaluation :
-   Quantify differences between clusters using statistical methods
-   Highligh key behavioral features that differentiate groups
6) Visualization :
-   Generate plots to visualize personnality traits and cluster distributions
7) Insights and interpretation:
-   Draw connections between personnality traits and lifespan predictions from Part 1.
-   Provide actionable based on behavioral clustering
-   Cluster plots showing distinct worm personality types.
-   Behavioral feature distributions across clusters.


5. # Results and Outputs
•	Predictions:
	Provides lifespan predictions for worms based on their early behavior.
•	Visualizations:
	Includes Kaplan-Meier survival curves and other plots for understanding model performance.
•	Performance Metrics:
	Reports accuracy, RMSE, and survival analysis metrics.