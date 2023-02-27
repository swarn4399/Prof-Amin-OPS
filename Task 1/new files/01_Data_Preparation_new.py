# -*- coding: utf-8 -*-

import numpy as np
#import pylab as Plot
import pandas as pd
import gc
import os
import joblib #direct import working on my machine
from sklearn.preprocessing import LabelEncoder
#from sklearn.model_selection import train_test_split
import hickle as hkl
import glob
from tqdm import tqdm
###############################################################################
skip_data_path = 'C:/Users/13528/Task 1/Skip Data/'
train_data_path = 'C:/Users/13528/Task 1/train_fold/'
test_data_path = 'C:/Users/13528/Task 1/test_fold/log_input_*_000000000000.csv'
test_prehist_data_path = 'C:/Users/13528/Task 1/test_fold/log_prehistory'
###############################################################################
#file1 = 'tf_000000000000.csv'
#file2 = 'tf_000000000001.csv'
def process_song_features(skip_data_path):
    file1 = 'tf_000000000000.csv'
    file2 = 'tf_000000000001.csv'
    song_fea_0 = pd.read_csv(skip_data_path+file1)
    song_fea_1 = pd.read_csv(skip_data_path+file2)    
    song_fea_0 = pd.concat((song_fea_0,song_fea_1))    
    le = LabelEncoder()
    song_fea_0['track_id'] = le.fit_transform(song_fea_0['track_id'])    
    joblib.dump(le, 'le_track_id.pkl')
    song_fea_0['mode'] = le.fit_transform(song_fea_0['mode'])
    song_fea_0.to_parquet('spotify_song_fea.parquet')
    song_fea_1 = []
    gc.collect()
###############################################################################   
def process_train_data(train_data_path):
    le_track_id = joblib.load('le_track_id.pkl')
    #le = LabelEncoder()
    dirs = os.listdir(train_data_path)
    count = 0
    for json_file in dirs:    
        print(0, count/len(dirs))    
        train_data = pd.read_csv(train_data_path+json_file)    
        le_session = LabelEncoder()
        train_data['session_id'] = le_session.fit_transform(train_data['session_id'])
        train_data['track_id_clean'] = le_track_id.transform(train_data['track_id_clean'])
        train_data.to_parquet(skip_data_path+json_file+'.parquet')     
        count = count + 1
###############################################################################
def process_test_data(test_data_path,test_prehist_data_path):
    test_files = glob.glob(test_data_path)
    le_track_id = joblib.load('le_track_id.pkl')
    for i in tqdm(range(len(test_files))):
        le = LabelEncoder()        
        file = pd.read_csv(test_files[i])
        prefix = test_prehist_data_path
        pre_file = pd.read_csv(prefix + test_files[i][41:]) #change as needed for new path        
        file['session_id'] = le.fit_transform(file['session_id'])        
        pre_file['session_id'] = le.transform(pre_file['session_id'])        
        pre_file['track_id_clean'] = le_track_id.transform(pre_file['track_id_clean'])
        file['track_id_clean'] = le_track_id.transform(file['track_id_clean'])
        file.to_parquet('C:/Users/13528/Task 1/test_pred/'+ 'pred_'+test_files[i][42:50]+'.parquet')        
        pre_file.to_parquet('C:/Users/13528/Task 1/test_pred/'+ 'prehist_'+test_files[i][42:50]+'.parquet')
###############################################################################
def prepare_glove_data(skip_data_path):
    data_path = skip_data_path
    train_files = glob.glob(data_path + '*.csv.parquet')
    #le_track_id = joblib.load('le_track_id.pkl')
    #le = LabelEncoder()
    all_data = []
    count = 0
    for file in train_files:    
        print(0, count/len(train_files))        
        train_data = pd.read_parquet(file)        
        cols = ['session_id', 'session_position', 'session_length', 'track_id_clean']        
        train_data = train_data[cols]        
        train_data['track_id_clean'] = train_data['track_id_clean'] + 1        
        raw_data = np.array(train_data.values)*1
        raw_data = raw_data.astype(np.int)            
        n_session = np.max(train_data['session_id'])+1        
        gc.collect()       
        data = np.zeros((n_session*30))        
        data[raw_data[:,0].astype(np.int)*30+raw_data[:,1].astype(np.int)-1] = raw_data[:,3]
        if count == 0:
            all_data = data
        else:
            all_data = np.concatenate((all_data,data))         
        count = count + 1
    np.savetxt('glove_data.txt',all_data,newline=' ', delimiter = ' ', fmt='%i')
###############################################################################
def process_glove_data():
    def get_coefs(word,*arr): 
        return (word), np.asarray(arr, dtype='float32')
    embeddings_index = dict(get_coefs(*o.strip().split()) for o in open('vectors_150.txt'))
    song_embedding_matrix = np.zeros((3706389,150))
    #keys = embeddings_index.keys()
    for i in range(0,song_embedding_matrix.shape[0]):
            tmp = embeddings_index.get(str(i))
            if tmp is not None:
                song_embedding_matrix[i,:] = tmp
    hkl.dump(song_embedding_matrix, 'song_embedding_matrix_150.hkl', mode='w')
    joblib.dump(song_embedding_matrix, 'song_embedding_matrix_150.pkl')
###############################################################################
#process_song_features(skip_data_path, file1, file2)
#process_train_data(train_data_path)
#process_test_data(test_data_path,test_prehist_data_path)
#prepare_glove_data(skip_data_path)
#process_glove_data()
###############################################################################
def data_preparation(skip_data_path,train_data_path, test_data_path,test_prehist_data_path):
    process_song_features(skip_data_path)
    process_train_data(train_data_path)
    process_test_data(test_data_path,test_prehist_data_path)
    prepare_glove_data(skip_data_path)
    process_glove_data()
###############################################################################
data_preparation(skip_data_path,train_data_path, test_data_path,test_prehist_data_path)