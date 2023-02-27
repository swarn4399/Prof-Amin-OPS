# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

###############################################################################

test_data_path = 'C:/Users/13528/Task 1/test_pred/pred_20180715.parquet'
track_id = 2988760
session_position = 16
session_length = 16
session_id = 0

###############################################################################

def prepare_new_data(test_data_path,track_id,session_position,session_length,session_id):
    test_data_1 = pd.read_parquet(test_data_path)
    track_ids = np.array(test_data_1['track_id_clean'].unique())
    i = np.where(track_ids == track_id)
    track_ids_new = np.delete(track_ids, i)

    df = pd.DataFrame(track_ids_new, columns =['track_id_clean'])
    df['session_position'] = session_position
    df['session_length'] = session_length
    df['session_id'] = session_id

    df.to_csv('C:/Users/13528/Task 1/test_fold/log_input_20230226_000000000000.csv', index = False)
    
###############################################################################

prepare_new_data(test_data_path,track_id,session_position,session_length,session_id)