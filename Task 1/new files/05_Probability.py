# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
import gc
import os
import glob
import hickle as hkl
from tqdm import tqdm
import joblib
from scipy.special import logit, expit
from itertools import islice

###############################################################################

test_data_path = 'C:/Users/13528/Task 1/test_pred/'

skip_data_path = 'C:/Users/13528/Task 1/Skip Data/'

test_files = glob.glob(test_data_path + 'pred_*.parquet')
#len(test_files)
test_files = list(np.sort(test_files))
#print(test_files)
test_data_1 = pd.read_parquet('C:/Users/13528/Task 1/test_pred/pred_20180715.parquet')

###############################################################################

with open('submission_0108_new_1.txt', 'w') as f:
    for item in tqdm(test_files):
        #data =0.35*hkl.load(data_path+item[15:23]+'_y_pred_mtsk.hkl') + 0.33*hkl.load(data_path+item[15:23]+'_y_pred_max.hkl') + 0.31*hkl.load(data_path+item[15:23]+'_y_pred.hkl')
       # data = 0.32*hkl.load(data_path+item[15:23]+'_y_pred_mtsk_larger_bn.hkl')+0.34*hkl.load(data_path+item[15:23]+'_y_pred_mtsk_larger_v2.hkl')+0.34*hkl.load(data_path+item[15:23]+'_y_pred_mtsk_larger.hkl')
     #   data = 0.25*hkl.load(data_path+item[15:23]+'_y_pred_mtsk_larger_bn.hkl')+0.25*hkl.load(data_path+item[15:23]+'_y_pred_mtsk_larger_v2.hkl')+0.25*hkl.load(data_path+item[15:23]+'_y_pred_mtsk_larger.hkl')+0.25*hkl.load(data_path+item[15:23]+'_y_pred_mtsk_larger_v3.hkl')
        #data = 0.2*hkl.load(data_path+item[15:23]+'_y_pred_mtsk_larger_bn.hkl')+0.2*hkl.load(data_path+item[15:23]+'_y_pred_mtsk_larger_v2.hkl')+0.2*hkl.load(data_path+item[15:23]+'_y_pred_mtsk_larger.hkl')+0.2*hkl.load(data_path+item[15:23]+'_y_pred_mtsk_larger_v3.hkl')+0.2*hkl.load(data_path+item[15:23]+'_y_pred_mtsk_larger_v4.hkl')
       # data = expit(0.2*logit(hkl.load(data_path+item[15:23]+'_y_pred_mtsk_larger_bn.hkl'))+0.2*logit(hkl.load(data_path+item[15:23]+'_y_pred_mtsk_larger_v2.hkl'))+0.2*logit(hkl.load(data_path+item[15:23]+'_y_pred_mtsk_larger.hkl'))+0.2*logit(hkl.load(data_path+item[15:23]+'_y_pred_mtsk_larger_v3.hkl'))+0.2*logit(hkl.load(data_path+item[15:23]+'_y_pred_mtsk_larger_v4.hkl')))
        #data = expit(0.166*logit(hkl.load(data_path+item[15:23]+'_y_pred_mtsk_larger_bn.hkl'))+0.166*logit(hkl.load(data_path+item[15:23]+'_y_pred_mtsk_larger_v2.hkl'))+0.167*logit(hkl.load(data_path+item[15:23]+'_y_pred_mtsk_larger.hkl'))+0.167*logit(hkl.load(data_path+item[15:23]+'_y_pred_mtsk_larger_v3.hkl'))+0.167*logit(hkl.load(data_path+item[15:23]+'_y_pred_mtsk_larger_v4.hkl'))+0.167*logit(hkl.load(data_path+item[15:23]+'_y_pred_mtsk_larger_v6.hkl')))
#         data = hkl.load(data_path+item[15:23]+'_y_pred_mtsk_larger_v6.hkl')
#         data = hkl.load(data_path+item[37:45]+'_y_pred_mtsk_larger_v6.hkl')
        data = joblib.load(skip_data_path+item[37:45]+'_y_pred_mtsk_larger_v6.pkl')

#         pred = data*0
#         pred[data>0.5] = 1        
#         pred = pred.astype(int)        
        preddf = pd.read_parquet(item)
        preddf = preddf[['session_id','session_length']]
        preddf = preddf.drop_duplicates(subset = 'session_id')
        session_length = np.array(preddf['session_length'])
        if data.shape[0] != len(session_length):
            print('error!')
        for i in range(len(session_length)):
            tmp = data[i,:]
            tmp_len = int(np.ceil(session_length[i]/2))
            tmp = tmp[range(tmp_len)]
            f.write("%s\n" % ','.join(map(str, tmp)))
            
###############################################################################

with open("C:/Users/13528/Task 1/submission_0108_new_1.txt") as myfile:
    head = list(islice(myfile,444491))
# print(head)
for i in range(len(head)):
    head[i] = head[i].removesuffix('\n')
    
# print(head)
count = 0
# len(head)
for i in range(len(head)):
    for j in range(len(head[i].split(','))):
        count = count+1
#print(count)

arr2 =[]
for i in range(len(head)):
    arr = head[i].split(',')
    for j in range(len(arr)):
        arr2.append(float(arr[j]))
#print(len(arr2))

###############################################################################

new_data = test_data_1

new_data['skip_probability'] = arr2
