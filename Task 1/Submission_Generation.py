# -*- coding: utf-8 -*-
"""

@author: Swarnabha
"""

import numpy as np
import pandas as pd
import gc
import os
import glob
import hickle as hkl
from tqdm import tqdm
from scipy.special import logit, expit

test_data_path = 'test_pred/'

data_path = 'Skip_Data/'

test_files = glob.glob(test_data_path + 'pred_*.parquet')
#print(len(test_files))
test_files = list(np.sort(test_files))

#print(test_files)

with open('submission_0108.txt', 'w') as f:
    for item in tqdm(test_files):
        #data =0.35*hkl.load(data_path+item[15:23]+'_y_pred_mtsk.hkl') + 0.33*hkl.load(data_path+item[15:23]+'_y_pred_max.hkl') + 0.31*hkl.load(data_path+item[15:23]+'_y_pred.hkl')
       # data = 0.32*hkl.load(data_path+item[15:23]+'_y_pred_mtsk_larger_bn.hkl')+0.34*hkl.load(data_path+item[15:23]+'_y_pred_mtsk_larger_v2.hkl')+0.34*hkl.load(data_path+item[15:23]+'_y_pred_mtsk_larger.hkl')
     #   data = 0.25*hkl.load(data_path+item[15:23]+'_y_pred_mtsk_larger_bn.hkl')+0.25*hkl.load(data_path+item[15:23]+'_y_pred_mtsk_larger_v2.hkl')+0.25*hkl.load(data_path+item[15:23]+'_y_pred_mtsk_larger.hkl')+0.25*hkl.load(data_path+item[15:23]+'_y_pred_mtsk_larger_v3.hkl')
        #data = 0.2*hkl.load(data_path+item[15:23]+'_y_pred_mtsk_larger_bn.hkl')+0.2*hkl.load(data_path+item[15:23]+'_y_pred_mtsk_larger_v2.hkl')+0.2*hkl.load(data_path+item[15:23]+'_y_pred_mtsk_larger.hkl')+0.2*hkl.load(data_path+item[15:23]+'_y_pred_mtsk_larger_v3.hkl')+0.2*hkl.load(data_path+item[15:23]+'_y_pred_mtsk_larger_v4.hkl')
       # data = expit(0.2*logit(hkl.load(data_path+item[15:23]+'_y_pred_mtsk_larger_bn.hkl'))+0.2*logit(hkl.load(data_path+item[15:23]+'_y_pred_mtsk_larger_v2.hkl'))+0.2*logit(hkl.load(data_path+item[15:23]+'_y_pred_mtsk_larger.hkl'))+0.2*logit(hkl.load(data_path+item[15:23]+'_y_pred_mtsk_larger_v3.hkl'))+0.2*logit(hkl.load(data_path+item[15:23]+'_y_pred_mtsk_larger_v4.hkl')))
        #data = expit(0.166*logit(hkl.load(data_path+item[15:23]+'_y_pred_mtsk_larger_bn.hkl'))+0.166*logit(hkl.load(data_path+item[15:23]+'_y_pred_mtsk_larger_v2.hkl'))+0.167*logit(hkl.load(data_path+item[15:23]+'_y_pred_mtsk_larger.hkl'))+0.167*logit(hkl.load(data_path+item[15:23]+'_y_pred_mtsk_larger_v3.hkl'))+0.167*logit(hkl.load(data_path+item[15:23]+'_y_pred_mtsk_larger_v4.hkl'))+0.167*logit(hkl.load(data_path+item[15:23]+'_y_pred_mtsk_larger_v6.hkl')))
        data = hkl.load(data_path+item[15:23]+'_y_pred_mtsk_larger_v6.hkl')

        pred = data*0
        pred[data>0.5] = 1        
        pred = pred.astype(int)        
        preddf = pd.read_parquet(item)
        preddf = preddf[['session_id','session_length']]
        preddf = preddf.drop_duplicates(subset = 'session_id')
        session_length = np.array(preddf['session_length'])
        if data.shape[0] != len(session_length):
            print('error!')
        for i in range(len(session_length)):
            tmp = pred[i,:]
            tmp_len = int(np.ceil(session_length[i]/2))
            tmp = tmp[range(tmp_len)]
            f.write("%s\n" % ''.join(map(str, tmp)))
