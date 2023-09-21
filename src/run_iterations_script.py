import multiprocessing as mp
import numpy as np
import pandas as pd
import pickle as pk
import os, sys
import time
import subprocess
from datetime import datetime
from run_iteration_fns import *
#from RF_model_iter import train_rf, make_predictions
#from selection_fns import get_top_selection

# Set initial values:

start_iter = 1 #sys.argv[1]
tot_iters = 1 #200
n_batches = 75
n_cpus = 5
max_seconds_for_iteration = 60*60*6

dataset_file = '/users/xpb20111/Huw/datasets/pymolgen_all_batch?.csv'
desc_file = '/users/xpb20111/Huw/datasets/pymolgen_all_batch?_desc.csv.gz'
docking_score_file = '/users/xpb20111/Huw/docking/pymolgen_all_docking_scores_batch?.csv.gz'

# Directories must end in "/":
docking_dir = '/users/xpb20111/Huw/docking/'
run_dir = '/users/xpb20111/Huw/results/test4/'
prev_it_dir = '/users/xpb20111/Huw/results/it0/'


# ============== #
# Run iterations #
# ============== #

#np.random.seed(9)

# Determine number of CPUs to use:
if n_cpus == -1:
    n_cpus = mp.cpu_count()
print('Running with {} CPUs'.format(n_cpus))

# Read any compounds without docking scores as these should be
# excluded from later selections:
excluded_cmpds_file = run_dir+'Compounds_without_docking_scores.csv'
if not os.path.exists(excluded_cmpds_file):
    with open(excluded_cmpds_file, 'w') as f:
        f.write('ID,Iteration\n')
excluded_cmpds = pd.read_csv(excluded_cmpds_file, index_col=0)

# Start iterations loop:
for iter_no in range(start_iter, start_iter+tot_iters):   
    it_start_time = datetime.now()
    print('\n= Iteration: {} ========================'.format(iter_no))
    
    it_dir = run_dir+'it'+str(iter_no)+'_running/'
    os.mkdir(it_dir)
    
    # ---------------------------- #
    # Select next set of compounds #
    # ---------------------------- #
    
    print('- Selecting next set of compounds...')
    
    # Select compounds:
    ######################################
    #### Change function on next line ####
    ######################################
    select_idx = get_top_selection(preds_file=prev_it_dir+'all_preds.csv.gz', 
                                   index_col=0, 
                                   sort_col='test_uncert', 
                                   n_cmpds=5)
    
    df_select = pd.DataFrame(data=[], columns=[], index=select_idx)
    df_select.index.rename('ID', inplace=True)
    df_select['batch_no'] = [molid2batchno(molid) for molid in df_select.index]
    
    # ------------------------------------------ #
    # Read selection and run docking if required #
    # ------------------------------------------ #
    
    start_t = datetime.now()
    print('- Running docking if required...')
    print('Docking results:')
    print('  {:>16s}\t{:>15s}\t{:>15s}'.format('ID', 'Docking score', '(Time taken)'))
    print('  {:>16s}\t{:>15s}\t{:>15s}'.format('----------------', '---------------', '--------------'))

    # Read selected compounds, grouped by batch:
    for batch_no, idxs_in_batch in df_select.reset_index()\
                                            .groupby('batch_no')['ID']\
                                            .apply(list)\
                                            .iteritems():

        # Read csv file containing docking scores to see if compounds have already been docked:
        docking_score_batch_file = docking_score_file.replace('?', str(batch_no))
        df_dock = get_exclusive_df_access(docking_score_batch_file, index_col=0)

        # Indices of compounds for docking:
        for_docking = df_dock.loc[idxs_in_batch]\
                             .loc[df_dock['Docking_score'].isna(), []]
        
        if len(for_docking) == 0:
            
            # Close dataframe file:
            close_df(docking_score_batch_file)
            
        else:

            # Get SMILES for compounds for docking:
            for_docking['SMILES'] = pd.read_csv(dataset_file.replace('?', str(batch_no)), 
                                                index_col=0).loc[for_docking.index, 'SMILES']
            
            # Mark compounds being docked as PD (pending) to stop any parallel runs also docking
            # these compounds at the same time:
            df_dock.loc[for_docking.index, 'Docking_score'] = 'PD'
            write_df(df_dock, docking_score_batch_file)
            
            # Run docking and record score in csv file:
            print('* Docking compounds: '+', '.join(for_docking.index.to_list()), end='\r')
            for mol_id, smi in for_docking['SMILES'].iteritems():
                docking_score = dock(docking_dir, smi, mol_id, n_cpus=n_cpus, read_if_exists=True)
                # Record docking scores in master dataset file:
                df_dock = get_exclusive_df_access(docking_score_batch_file, index_col=0)
                df_dock.loc[docking_score[0], ['Docking_score', 'Docking_time']] = docking_score[1:]
                write_df(df_dock, docking_score_batch_file)

        df_select.loc[idxs_in_batch, ['Docking_score', 'Docking_time']] = \
                df_dock.loc[idxs_in_batch, ['Docking_score', 'Docking_time']]

        print(end=LINE_CLEAR)
        for mol_id in idxs_in_batch:
            if df_dock.loc[mol_id, 'Docking_score'] != 'PD':
                print('  {:>16s}\t{:>15s}\t{:>15s}'.format(
                    str(mol_id), str(df_dock.loc[mol_id, 'Docking_score']),
                    '({:.0f} s)'.format(df_dock.loc[mol_id, 'Docking_time'])))

    # If some compounds are currently being docked by parallel runs, wait until these are finished 
    # and read the scores from the csv file:
    if 'PD' in df_select['Docking_score'].to_list():
        for batch_no, idxs_in_batch in df_select.loc[df_select['Docking_score'] == 'PD']\
                                                .reset_index()\
                                                .groupby('batch_no')['ID']\
                                                .apply(list)\
                                                .iteritems():
            wait_time = 0
            while 'PD' in df_select.loc[idxs_in_batch, 'Docking_score'].to_list():
                wait_time += 10
                print('Waiting for docking running in parallel process... {} s'.format(wait_time), end='\r')
                time.sleep(10)
                docking_score_batch_file = docking_score_file.replace('?', str(batch_no))
                df_dock = get_exclusive_df_access(docking_score_batch_file, index_col=0)
                df_select.loc[idxs_in_batch, ['Docking_score', 'Docking_time']] = \
                        df_dock.loc[idxs_in_batch, ['Docking_score', 'Docking_time']]
                close_df(docking_score_batch_file)
            
            print(end=LINE_CLEAR)
            for mol_id in idxs_in_batch:
                print('  {:>16s}\t{:>15s}\t{:>15s}'.format(
                    str(mol_id), str(df_dock.loc[mol_id, 'Docking_score']),
                    '({:.0f} s)'.format(df_dock.loc[mol_id, 'Docking_time'])))

    end_t = datetime.now()
    print('- (time taken: {:.1f} s)'.format((end_t - start_t).total_seconds()))
    
    # ------------- #
    # Retrain model #
    # ------------- #
    
    start_t = datetime.now()
    print('- Training a new model...')
    
    # Record any selected compounds without docking scores, 
    # e.g. due to having no valid conformations, to ensure
    # these are not picked in subsequent iterations and
    # remove from training data for this iteration:
    cmpd_errors = df_select.loc[df_select['Docking_score'] == 'False', []]
    cmpd_errors['Iteration'] = iter_no
    cmpd_errors['Iteration'].to_csv(excluded_cmpds_file, 
                                    mode='a', header=False)
    excluded_cmpds = pd.concat([excluded_cmpds, ])
    df_select = df_select.loc[df_select['Docking_score'] != 'False'] # None
   
    # Get previous training data:
    x_prev, y_prev = pk.load(open(prev_it_dir+'train.pk', 'rb'))
    
    # Get descriptors for new x data:
    x_new = []
    for batch_no, idxs_in_batch in df_select.reset_index()\
                                            .groupby('batch_no')['ID']\
                                            .apply(list)\
                                            .iteritems():
        desc_batch_file = desc_file.replace('?', str(batch_no))
        x_new.append(pd.read_csv(desc_batch_file, index_col=0).loc[idxs_in_batch])

    # Set order to match selection order:
    x_new = pd.concat(x_new).loc[df_select.index]
    x = pd.concat([x_prev, x_new])
    
    y = pd.concat([y_prev, df_select['Docking_score']]).astype(float)
    
    # Check overlap between indices:
    if not x.index.equals(y.index):
        sys.exit('ERROR: Indices of X and y data do not overlap.')
    
#     pd.concat([y, x])\
#       .to_csv('/users/xpb20111/Huw/results/it'+str(iter_no)+'_running/train.csv')

    # Save training data:
    pk.dump([x, y], 
            open(it_dir+'train.pk', 'wb'))
    
    # Train new model:
    ######################################
    #### Change function on next line ####
    ######################################
    rf = train_rf(x, y, 
                  mc_cv=50, 
                  hyper_cv=5, 
                  n_iter=250, 
                  results_dir=it_dir,
                  n_jobs=n_cpus
                 )

    end_t = datetime.now()
    print('- (time taken: {:.1f} s)'.format((end_t - start_t).total_seconds()))
    
    # -------------------- #
    # Make new predictions #
    # -------------------- #
    
    start_t = datetime.now()
    print('- Making predictions on full dataset...')

    # Without multiprocessing:
    #make_predictions(rf, 
    #                 [desc_file.replace('?', str(i)) for i in range(n_batches)],
    #                 results_dir=it_dir)

    # With multiprocessing:
    pool = mp.Pool(n_cpus)
    preds = [pool.apply_async(make_predictions, 
                              args=(rf, 
                                    desc_file.replace('?', str(batch_no)), 
                                    x.index)) 
             for batch_no in range(n_batches)]
    write_header = True
    for p in preds:
        df_preds = p.get()
        df_preds.to_csv(it_dir + 'all_preds.csv.gz', mode='a', header=write_header)
        write_header = False
    pool.close()

    end_t = datetime.now()
    print('- (time taken: {:.1f} s)'.format((end_t - start_t).total_seconds()))
    
    # Remove "running" from directory name to indicate completed iteration:
    os.rename(it_dir, run_dir+'it'+str(iter_no))
    prev_it_dir = run_dir+'it'+str(iter_no)+'/'
    
    it_end_time = datetime.now()
    it_time = (it_end_time - it_start_time).total_seconds()
    print('= (time taken for full iteration: {:.1f} s)'.format(it_time))
    
    # Stop script if remaining walltime is too low for another complete iteration:
    check_remaining_walltime(max_seconds_for_iteration)

