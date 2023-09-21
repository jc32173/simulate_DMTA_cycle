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
import sys
from datetime import datetime

def predict_uncertainty(rf, x, mem_save=True):
    """
    Predict uncertainty for RF regression model as the standard 
    deviation of the predictions from each individual decision tree
    in the forest.
    """
    if mem_save:
        # Potentially lower memory version:
        dt_pred_sum = np.zeros(len(x))
        dt_pred_sum2 = np.zeros(len(x))
        n = 0
        for dt in rf.best_estimator_.estimators_:
            dt_pred = dt.predict(x)
            dt_pred_sum += dt_pred
            dt_pred_sum2 += dt_pred**2
            n += 1
        return ((dt_pred_sum2/n) - ((dt_pred_sum/n)**2))**0.5
    else:
        # One line:
        dt_preds = np.array([dt.predict(x) for dt in rf.best_estimator_.estimators_])
        return np.std(dt_preds, axis=0)

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
bootstrap = [True, False]

# Collect all hyperparameter values:
init_param_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}
    
#def train_rf_nested_cv(x, y, 
def train_rf(x, y, 
             mc_cv=10,
             hyper_cv=5,
             test_frac=0.3,
             hp_grid=init_param_grid,
             n_iter=10,
             uncertainty=True, 
             results_dir='',
             #x_ext_test_file=None, 
             n_jobs=-1, 
             verbose=1):
    
    # Number of MC CV cycles
    #mc_cv = 2
    
    # Number of CV folds for hyperparameter tuning:
    #hyper_cv = 2
    
    # Fraction of data to use as test set:
    #test_frac = 0.3
   
    # Add final / to directory: 
    if (len(results_dir) > 0) and (results_dir[-1] != '/'):
        results_dir += '/'
    
    # Name of file containing dataset:
    #data_filename = results_dir + 'train.csv'
    
    # Name of file to write results to:
    results_filename = results_dir + 'RF_nestedCV_results.txt'
    
    # Name of file to write all predictions to:
    predictions_filename = results_dir + 'RF_nestedCV_predictions.csv'
    
    # Molecular descriptors (x) and binding data (y):
    #x, y = pk.load(open(data_filename, 'rb'))
    
    # Variables to save model performance statistics:
    r2_sum = 0
    rmsd_sum = 0
    bias_sum = 0
    sdep_sum = 0
    
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
    
        # Set up hyperparameter tuning using a random grid search over
        # different combinations of hyperparameters:
        rf = RandomizedSearchCV(estimator = RandomForestRegressor(random_state=3),
                                param_distributions = init_param_grid,
                                n_iter = n_iter,
                                cv = hyper_cv,
                                refit = True,
                                verbose = verbose,
                                n_jobs = n_jobs,
                                random_state=n*2)
    
        # Train RF model:
        rf.fit(x_train, y_train)
    
        # Use trained RF model to predict y data for the test set:
        y_pred = rf.predict(x_test)
    
        # Assess performace of model based on predictions:
    
        # Coefficient of determination
        r2 = r2_score(y_test, y_pred)
        # Root mean squared error
        rmsd = mean_squared_error(y_test, y_pred)**0.5
        # Bias
        bias = np.mean(y_pred - y_test)
        # Standard deviation of the error of prediction
        sdep = np.mean(((y_pred - y_test) - np.mean(y_pred - y_test))**2)**0.5
    
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
    
    ## Save all individual predictions to file:
    predictions_file = open(predictions_filename, 'w')
    # Write header:
    predictions_file.write(','.join([str(i) for i in y.index]) + '\n')
    # Write individual predictions from each MC CV cycle:
    for n in range(mc_cv):
        predictions_file.write(','.join([str(p) if not np.isnan(p) else '' for p in all_preds[n]]) + '\n')
    predictions_file.close()
   
    print('- Training final RF model on full dataset...')

    # Retrain on full dataset:
    rf = RandomizedSearchCV(estimator = RandomForestRegressor(random_state=13),
                            param_distributions = init_param_grid,
                            n_iter = n_iter,
                            cv = hyper_cv,
                            refit = True,
                            verbose = verbose,
                            n_jobs = n_jobs,
                            random_state=11)
    
    # Train RF model:
    rf.fit(x, y)
    
    pk.dump(rf, open(results_dir+'RF.pk', 'wb'))

    return rf


#    # Save new predictions:
#    df_all_desc = pd.read_csv(x_ext_test_file)
#    df_all_desc.set_index('ID', inplace=True)
#    df_all_desc.dropna(inplace=True)
#    
#    df_train_pred = pd.DataFrame(data=rf.predict(df_all_desc.loc[x.index]), 
#                                 columns=['train'], 
#                                 index=x.index)
#    
#    df_test_pred = pd.DataFrame(data=rf.predict(df_all_desc.loc[~df_all_desc.index.isin(x.index)]), 
#                                columns=['test'], 
#                                index=df_all_desc.index[~df_all_desc.index.isin(x.index)])
#    
#    if uncertainty:
#        df_train_pred.loc[x.index, 'train_uncert'] = \
#                predict_uncertainty(rf, df_all_desc.loc[x.index])
#        
#        df_test_pred.loc[df_all_desc.index[~df_all_desc.index.isin(x.index)], 'test_uncert'] = \
#                predict_uncertainty(rf, df_all_desc.loc[~df_all_desc.index.isin(x.index)])
#
#    df_preds = pd.concat([df_train_pred, df_test_pred])
#    
#    df_preds.to_csv(results_dir + 'all_preds.csv')


def make_predictions_all_batches(rf, x_files, train_idx=[], uncertainty=True, 
                                 index_col=0, results_dir=''):

    # Add final / to directory: 
    if (len(results_dir) > 0) and (results_dir[-1] != '/'):
        results_dir += '/'

    write_header = True
    for f in x_files:
        x = pd.read_csv(f, index_col=index_col)
        x.dropna(inplace=True)

        train_idx = set(x.index) & set(train_idx)
        test_idx = set(x.index) - set(train_idx)

        df_preds = pd.DataFrame(data=rf.predict(x), 
                                columns=['pred'], 
                                index=x.index)
        
        df_preds.loc[train_idx, 'train_pred'] = df_preds.loc[train_idx, 'pred']
        df_preds.loc[test_idx, 'test_pred'] = df_preds.loc[test_idx, 'pred']
        df_preds.drop(columns=['pred'], inplace=True)

        if uncertainty:
            df_preds['uncert'] = predict_uncertainty(rf, x)

            df_preds.loc[train_idx, 'train_uncert'] = df_preds.loc[train_idx, 'uncert']
            df_preds.loc[test_idx, 'test_uncert'] = df_preds.loc[test_idx, 'uncert']
            df_preds.drop(columns=['uncert'], inplace=True)

            # Reorder columns:
            df_preds = df_preds[['train_pred', 'train_uncert', 'test_pred', 'test_uncert']]

        df_preds.to_csv(results_dir + 'all_preds.csv.gz', mode='a', header=write_header)

        write_header = False


#def make_predictions(rf, x_file, uncertainty=True, index_col=0)
#
#    x = pd.read_csv(x_file, index_col=index_col)
#    x.dropna(inplace=True)
#
#    df_preds = pd.DataFrame(data=rf.predict(x), 
#                            columns=['pred'], 
#                            index=x.index)
#    
#    if uncertainty:
#        df_preds['uncert'] = predict_uncertainty(rf, x)
#
#    return df_preds


def make_predictions(rf, x_file, train_idx=[], uncertainty=True, 
                     index_col=0, results_dir='', n_cpus=1):

    x = pd.read_csv(x_file, index_col=index_col)
    x.dropna(inplace=True)

    train_idx = set(x.index) & set(train_idx)
    test_idx = set(x.index) - set(train_idx)

    df_preds = pd.DataFrame(data=rf.predict(x), 
                            columns=['pred'], 
                            index=x.index)
    
    df_preds.loc[train_idx, 'train_pred'] = df_preds.loc[train_idx, 'pred']
    df_preds.loc[test_idx, 'test_pred'] = df_preds.loc[test_idx, 'pred']
    df_preds.drop(columns=['pred'], inplace=True)

    if uncertainty:
        df_preds['uncert'] = predict_uncertainty(rf, x)

        df_preds.loc[train_idx, 'train_uncert'] = df_preds.loc[train_idx, 'uncert']
        df_preds.loc[test_idx, 'test_uncert'] = df_preds.loc[test_idx, 'uncert']
        df_preds.drop(columns=['uncert'], inplace=True)

        # Reorder columns:
        df_preds = df_preds[['train_pred', 'train_uncert', 'test_pred', 'test_uncert']]

    return df_preds
