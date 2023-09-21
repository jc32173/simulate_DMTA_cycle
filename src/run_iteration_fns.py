#import multiprocessing as mp
#print('Running with {} CPUs'.format(mp.cpu_count()))
import numpy as np
import pandas as pd
import pickle as pk
import os, sys
import time
import subprocess
from datetime import datetime

#np.random.seed(9)

# ASCI patterns for overwriting printed lines:
# https://itnext.io/overwrite-previously-printed-lines-4218a9563527
# LINE_UP = '\033[1A'
LINE_CLEAR = '\x1b[2K'
#LINE_CLEAR = '\n'


# Read csv file:
def get_exclusive_df_access(filename, max_wait_time=False, index_col=None):
    """
    Get exclusive access to csv file by renaming it with the current process ID
    before reading it.
    """
    curr_pid = os.getpid()
    tmp_filename = filename.split('.')[0] + '_pid' + str(curr_pid) + '.csv.gz'
    tot_wait_time = 0
    while not os.path.exists(filename):
        time.sleep(1)
        tot_wait_time += 1
        print('Waiting to open csv file: {} s'.format(tot_wait_time), end='r')
        if max_wait_time and tot_wait_time >= max_wait_time:
            sys.exit('Max wait time exceeded when accessing file: {}'.format(filename))
    os.rename(filename, tmp_filename)
    print(end=LINE_CLEAR)
    df = pd.read_csv(tmp_filename, index_col=index_col)
    return df


# Save docking file:
def write_df(df, filename):
    """
    Write csv file then rename to original filename.
    """
    curr_pid = os.getpid()
    tmp_filename = filename.split('.')[0] + '_pid' + str(curr_pid) + '.csv.gz'
    df.to_csv(tmp_filename)
    os.rename(tmp_filename, filename)


# Close docking file:
def close_df(filename): #, df=None
    """
    Close csv file without writing and then rename.
    """
    curr_pid = os.getpid()
    tmp_filename = filename.split('.')[0] + '_pid' + str(curr_pid) + '.csv.gz'
#     if 
#     df.to_csv(tmp_filename)
    os.rename(tmp_filename, filename)


## Function to bypass docking:
# def dock(molname, *args):
#     return [molname, np.random.random(), np.random.random()]


# Function to run docking:
def dock(docking_dir, smi, molname, n_cpus=1, read_if_exists=True, save_output=True):
    """
    Run docking from SMILES and return docking score.
    """
    docked = False
    docking_time = np.nan
    if not os.path.exists(docking_dir+molname+'/'+molname+'_all_gnina_data.csv'):
        curr_dir = os.getcwd()
        os.mkdir(docking_dir+molname+'/')
        os.chdir(docking_dir+molname+'/')
        # ./dock_gnina.sh smi molname
#         print(molname+'/'+molname+'_all_gnina_data.csv')
        start_time = datetime.now()
        if save_output:
            subprocess.run(['/users/xpb20111/Huw/programs/dock_gnina.sh', smi, molname, str(n_cpus)], 
                            stdout=open(docking_dir+molname+'/stdout_stderr.dat', 'w'), 
                            stderr=subprocess.STDOUT)
        else:
            subprocess.run(['/users/xpb20111/Huw/programs/dock_gnina.sh', smi, molname, str(n_cpus)])
            # To silence output from docking, use:
                           #stdout=subprocess.DEVNULL, 
                           #stderr=subprocess.DEVNULL)
        end_time = datetime.now()
        docking_time = (end_time - start_time).total_seconds()
        os.chdir(curr_dir)
        docked = True
    if docked or read_if_exists:
        try:
            best_docking_score = pd.read_csv(docking_dir+molname+'/'+molname+'_all_gnina_data.csv')\
                                   .nsmallest(1, 'affinity_(kcal/mol)')\
                                   [['molname', 'affinity_(kcal/mol)', 'time']]\
                                   .T.squeeze().to_list()
            best_docking_score[2] = docking_time
        except pd.errors.EmptyDataError:
            best_docking_score = [molname, False, np.nan]
#         subprocess.run(['tar', '-czvf', docking_dir+molname+'.tar.gz', docking_dir+molname, '--remove-files'])
    else:
        sys.exit('ERROR in docking function.')
    return best_docking_score


# Check remaining walltime for slurm job:
def check_remaining_walltime(max_seconds_for_iteration):
    """
    Stop script if remaining walltime to too low.
    """
    slurm_id = os.environ.get('SLURM_JOB_ID')
    if slurm_id is not None:
        seconds_remaining = subprocess.check_output(['/users/xpb20111/scripts/sys/slurm_seconds_remaining.sh', slurm_id])
        seconds_remaining = int(seconds_remaining.decode().strip())
        if seconds_remaining < max_seconds_for_iteration:
            sys.exit('Stopping script here to prevent leaving incomplete iteration (remaining walltime {} s)'.format(seconds_remaining))


# Convert molecule ID to molecule number:
def molid2molno(molid):
    """
    Convert molecule ID to molecule number, this is 
    specific for a particular dataset with hardcoded values.
    """
    if molid.startswith('metapara'):
        return int(molid.strip('metapara')) + 3588003
    elif molid.startswith('meta'):
        return int(molid.strip('meta'))
    elif molid.startswith('para'):
        return int(molid.strip('para')) + 2262784


# Get dataset file batch number from molecule ID:
def molid2batchno(molid, batch_len=100000):
    """
    Get dataset file batch number from molecule ID.
    """
    molno = molid2molno(molid)
    return molno//batch_len


# Make predictions for a set of descriptors:
def make_predictions(rf, x_file, train_idx=[], uncertainty=True,
                     index_col=0, results_dir='', n_cpus=1):
    """
    Read descriptors from a file and make predictions using RF model.
    """

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


# Predict uncertainty fro an RF model:
def predict_uncertainty(rf, x):
    """
    Predict uncertainty for RF regression model as the standard 
    deviation of the predictions from each individual decision tree
    in the forest.
    """
    dt_preds = np.array([dt.predict(x) for dt in rf.best_estimator_.estimators_])
    return np.std(dt_preds, axis=0)


# Get list of selected molecule IDs:
def get_selected_ids_betweens_its(all_preds_file_it0,
                                  all_preds_file_it1,
                                  chunksize=100000):
    """
    Get the list of molecule IDs which were selected in a given 
    iteration by reading the all_preds.csv.gz files in different 
    iterations.
    """

    df_it0_iter = pd.read_csv(all_preds_file_it0,
                              chunksize=chunksize,
                              header=0,
                              index_col=0)
    df_it1_iter =  pd.read_csv(all_preds_file_it1,
                               chunksize=chunksize,
                               header=0,
                               index_col=0)
    selected_ids = []

    while True:
        try:
            df_it0 = next(df_it0_iter)
            df_it1 = next(df_it1_iter)

            df = pd.merge(left=df_it0,
                          left_index=True,
                          right=df_it1,
                          right_index=True,
                          suffixes=['_it0', '_it1'])

            selected_ids += (df.loc[df['train_pred_it0'].isna() & \
                                    ~df['train_pred_it1'].isna()]\
                               .index.to_list())

        except StopIteration:
            break
    return selected_ids
