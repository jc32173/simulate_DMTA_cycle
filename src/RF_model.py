#! /bin/usr/env python

# Random forest with nested CV

from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error

import numpy as np
import pandas as pd
import pickle as pk

# Number of MC CV cycles
mc_cv = 2

# Number of CV folds for hyperparameter tuning:
hyper_cv = 2

# Fraction of data to use as test set:
test_frac = 0.3

# Name of file to write results to:
results_filename = 'RF_results.txt'

# Name of file to write all predictions to:
predictions_filename = 'RF_predictions.csv'

# Variables to save model performance statistics:
r2_sum = 0
rmsd_sum = 0
bias_sum = 0
sdep_sum = 0

# Load dataset:



# List to save individual predictions from the models trained from
# each train/test split:
all_preds = np.empty((mc_cv, len(y)), dtype=float)
all_preds[:] = np.nan

# Initialise train test split:
train_test_split = ShuffleSplit(mc_cv, test_size=test_frac)

# Monte Carlo CV:
for n, [train_idx, test_idx] in enumerate(train_test_split.split(x)):

    # Separate data into training and test sets:
    # Have to use ".iloc" if x and y are pandas DataFrames and Series objects, 
    # if they are just numpy arrays remove ".iloc".
    x_train = x.iloc[train_idx]
    x_test = x.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    # Centre and scale all x features to have mean=0 and var=1:
    # (Not required for random forest, but important for some other ML methods)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # Initial parameters for RF hyperparameter tuning:
    
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 20, stop = 300, num = 10)]
    
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 100, num = 10)]
    max_depth.append(None)
    
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    
    # Method of selecting samples for training each tree
    bootstrap = [True]
    
    # Collect all hyperparameter values:
    init_param_grid = {'n_estimators': n_estimators,
                       'max_features': max_features,
                       'max_depth': max_depth,
                       'min_samples_split': min_samples_split,
                       'min_samples_leaf': min_samples_leaf,
                       'bootstrap': bootstrap}

    # Set up hyperparameter tuning using a random grid search over
    # different combinations of hyperparameters:
    rf = RandomizedSearchCV(estimator = RandomForestRegressor(),
                            param_distributions = init_param_grid,
                            n_iter = 10,
                            cv = hyper_cv,
                            refit = True,
                            verbose = 1,
                            n_jobs = 1)

    # Train RF model:
    rf.fit(x_train, y_train)

    # Use trained RF model to predict y data for the test set:
    y_pred = rf.predict(x_test)

    # Assess performace of model based on predictions:

    # Coefficient of determination
    r2 =
    # Root mean squared error
    rmsd =
    # Bias
    bias =
    # Standard deviation of the error of prediction
    sdep =

    # Save running sum of results:
    r2_sum += r2
    rmsd_sum += rmsd
    bias_sum += bias
    sdep_sum += sdep

    # Save individual predictions:
    all_preds[n,test_idx] = y_pred

# Average results over resamples:
r2_av = r2_sum/mc_cv
rmsd_av = rmsd_sum/mc_cv
bias_av = bias_sum/mc_cv
sdep_av = sdep_sum/mc_cv

# Write average results to a file:
results_file = open(results_filename, 'w')
results_file.write('r2: {:.3f}\n'.format(r2_av))
results_file.write('rmsd: {:.3f}\n'.format(rmsd_av))
results_file.write('bias: {:.3f}\n'.format(bias_av))
results_file.write('sdep: {:.3f}\n'.format(sdep_av))
results_file.close()

# Save all individual predictions to file:
predictions_file = open(predictions_filename, 'w')
# Write header:
predictions_file.write(','.join([str(i) for i in y.index]) + '\n')
# Write individual predictions from each MC CV cycle:
for n in range(mc_cv):
    predictions_file.write(','.join([str(p) if not np.isnan(p) else '' for p in all_preds[n]]) + '\n')
predictions_file.close()
