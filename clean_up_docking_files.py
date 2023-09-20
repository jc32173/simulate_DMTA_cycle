#! /usr/bin/env python

# Remove any 'PD' (pending) docking scores in docking score dataset files, in 
# case a run has stopped during docking

import pandas as pd
import numpy as np
import sys

n_batches = 75
docking_score_file = '/users/xpb20111/Huw/docking/pymolgen_all_docking_scores_batch?.csv.gz'

print('Pending docking scores:')
for batch_no in range(n_batches):

    docking_score_batch_file = docking_score_file.replace('?', str(batch_no))
    
    df = pd.read_csv(docking_score_batch_file, index_col=0)
    if 'PD' in df['Docking_score'].to_list():
        print('Batch no: {}'.format(batch_no))
        for molid in df.loc[df['Docking_score'] == 'PD'].index:
            print(molid)
        df.loc[df['Docking_score'] == 'PD', 'Docking_score'] = np.nan
        df.to_csv(docking_score_batch_file)
        #os.system(gzip)
