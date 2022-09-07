# -*- coding: utf-8 -*-

#conda install -c conda-forge hickle
#conda install -c conda-forge pyarrow

################################
#import all the libraries needed
################################

import numpy as np
import pylab as Plot
import pandas as pd
import gc
import os
#from sklearn.externals import joblib
import joblib #direct import working on my machine
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import hickle as hkl
import glob
from tqdm import tqdm

################################
# Preparation of the song metadata file, the track_id is encoded as numerical values to save space.
################################

data_path = 'C:/Users/13528/Task 1/Skip Data/'

song_fea_0 = pd.read_csv(data_path+'tf_000000000000.csv')
song_fea_1 = pd.read_csv(data_path+'tf_000000000001.csv')

song_fea_0 = pd.concat((song_fea_0,song_fea_1))

le = LabelEncoder()

song_fea_0['track_id'] = le.fit_transform(song_fea_0['track_id'])

joblib.dump(le, 'le_track_id.pkl')

song_fea_0['mode'] = le.fit_transform(song_fea_0['mode'])

song_fea_0.to_parquet('spotify_song_fea.parquet')

song_fea_1 = []
gc.collect()

################################
# Conversion of the training session csv files to parquet files for smaller size and fast loading, the track_ids and session ids are encoded as numerical values to save space. The generated files are saved in the 'Skip_Data/' folder.
# If done correcly, the name of the generated parquet files should be something like "log_3_20180918_000000000000.csv.parquet"
################################

le_track_id = joblib.load('le_track_id.pkl')

dirs = os.listdir( 'C:/Users/13528/Task 1/train_fold/' )
count = 0
for json_file in dirs:    
    print(0, count/len(dirs))
    
    train_data = pd.read_csv('C:/Users/13528/Task 1/train_fold/'+json_file)
    
    le_session = LabelEncoder()

    train_data['session_id'] = le_session.fit_transform(train_data['session_id'])

    train_data['track_id_clean'] = le_track_id.transform(train_data['track_id_clean'])

    train_data.to_parquet(data_path+json_file+'.parquet')
     
    count = count + 1
    
################################    
# Conversion of the training session csv files to parquet files for smaller size and fast loading, the track_ids and session ids are encoded as numerical values to save space. The generated files are saved in the 'Skip_Data/' folder.
# There are two kinds of test files, namely files with names that start with "log_input" or "log_prehistory". If this part of code is executed correctly, the names of the generated parquet files should be correspondingly be something like "pred_20180718.parquet" and "prehist_20180810.parquet" for 
################################    
    
test_files = glob.glob('C:/Users/13528/Task 1/test_fold/log_input_*_000000000000.csv')

for i in tqdm(range(len(test_files))):
    le = LabelEncoder()
    
    file = pd.read_csv(test_files[i])
    
    prefix = 'C:/Users/13528/Task 1/test_fold/log_prehistory'
    
    pre_file = pd.read_csv(prefix + test_files[i][41:]) #adjust
    
    file['session_id'] = le.fit_transform(file['session_id'])
    
    pre_file['session_id'] = le.transform(pre_file['session_id'])
    
    pre_file['track_id_clean'] = le_track_id.transform(pre_file['track_id_clean'])
    file['track_id_clean'] = le_track_id.transform(file['track_id_clean'])
    
    file.to_parquet('C:/Users/13528/Task 1/test_pred/'+ 'pred_'+test_files[i][42:50]+'.parquet')
    
    pre_file.to_parquet('C:/Users/13528/Task 1/test_pred/'+ 'prehist_'+test_files[i][42:50]+'.parquet')
 
    
################################
# Preparation of the file for Glove training.
################################ 
    
train_files = glob.glob(data_path + '*.csv.parquet')

all_data = []

count = 0
for file in train_files:    
     print(0, count/len(train_files))
     
     train_data = pd.read_parquet(file)
     
     cols = ['session_id', 'session_position', 'session_length', 'track_id_clean']
     
     train_data = train_data[cols]
     
     train_data['track_id_clean'] = train_data['track_id_clean'] + 1
     
     raw_data = np.array(train_data.values)*1

     #raw_data = raw_data.astype(np.int)    
     raw_data = raw_data.astype(int)
     
     n_session = np.max(train_data['session_id'])+1
     
     gc.collect()
     
     data = np.zeros((n_session*30))
     
     #data[raw_data[:,0].astype(np.int)*30+raw_data[:,1].astype(np.int)-1] = raw_data[:,3]
     data[raw_data[:,0].astype(int)*30+raw_data[:,1].astype(int)-1] = raw_data[:,3]

     if count == 0:
         all_data = data
     else:
         all_data = np.concatenate((all_data,data))
      
     count = count + 1

np.savetxt('glove_data.txt',all_data,newline=' ', delimiter = ' ', fmt='%i')

################################
# After learning the Glove embedding, let the embedding txt file be named as 'vectors_150.txt', this part of code convert the txt file to numpy format.
################################

## Glove embedding not done properly!

#from gensim.scripts.glove2word2vec import glove2word2vec
#glove_input_file = 'C:/Users/13528/Task 1/glove_data.txt'
#word2vec_output_file = 'C:/Users/13528/Task 1/word2vec.txt'
#glove2word2vec(glove_input_file, word2vec_output_file)


def get_coefs(word,*arr): return (word), np.asarray(arr, dtype='float32')
#embeddings_index = dict(get_coefs(*o.strip().split()) for o in open('vectors_150.txt'))
#embeddings_index = dict(get_coefs(*o.strip().split()) for o in open('glove_data.txt'))
embeddings_index = dict(get_coefs(*o.strip().split()) for o in open('word2vec.txt'))


#song_embedding_matrix = np.zeros((3706389,150))
song_embedding_matrix = np.zeros((1,58950989)) #hardcoded

keys = embeddings_index.keys()

for i in range(0,song_embedding_matrix.shape[0]):
        tmp = embeddings_index.get(str(i))
        if tmp is not None:
            song_embedding_matrix[i,:] = tmp
            
hkl.dump(song_embedding_matrix, 'song_embedding_matrix_150.hkl', mode='w', compression='gzip')