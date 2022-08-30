# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 19:50:31 2022

@author: Swarnabha
"""

import numpy as np
import pandas as pd
import gc
import os
import hickle as hkl
from numpy import random
from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import LabelEncoder
#from sklearn.externals import joblib
import joblib

from keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD
from keras import initializers, regularizers, constraints
from keras.layers import Bidirectional, GlobalMaxPooling1D, Dense, Embedding
from keras.layers import concatenate, Input, LSTM, GRU, merge, Lambda, Dot, Add, Multiply, \
wrappers, Dropout, GlobalAveragePooling1D, Bidirectional, BatchNormalization, Activation
import tensorflow as tf
from keras.models import Sequential
from keras.layers.core import Reshape
import keras.backend as K
#from keras.engine.topology import Layer
from tensorflow.keras.layers import Layer
from keras.layers import LSTM, Dense
from tqdm import tqdm
import glob
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback, ReduceLROnPlateau


import keras.backend as KTF
#import tensorflow as tf
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
KTF.set_session(sess)

test_data_path = 'test_pred/'

###############################################################################

batch_size = 1000

data_path = 'Skip_Data/'

le_track_id = joblib.load('le_track_id.pkl')

spotify_song_fea = pd.read_parquet('spotify_song_fea.parquet')

le = LabelEncoder()
##
spotify_song_fea['mode'] = le.fit_transform(spotify_song_fea['mode'])

track_id = np.array(spotify_song_fea['track_id'])

cols_to_be_normalized = ['duration', 'release_year', 'us_popularity_estimate',
       'acousticness', 'beat_strength', 'bounciness', 'danceability',
       'dyn_range_mean', 'energy', 'flatness', 'instrumentalness', 'key',
       'liveness', 'loudness', 'mechanism', 'mode', 'organism', 'speechiness',
       'tempo', 'time_signature', 'valence', 'acoustic_vector_0',
       'acoustic_vector_1', 'acoustic_vector_2', 'acoustic_vector_3',
       'acoustic_vector_4', 'acoustic_vector_5', 'acoustic_vector_6',
       'acoustic_vector_7']

scaler = MinMaxScaler()

spotify_song_array = scaler.fit_transform(spotify_song_fea[cols_to_be_normalized])

order = np.argsort(track_id)
spotify_song_array = spotify_song_array[order,:]

###############################################################################


song_zero_embedding = -2*np.ones((1,spotify_song_array.shape[1]))
spotify_song_array = np.concatenate((song_zero_embedding,spotify_song_array),axis=0)

#print(spotify_song_array.shape)
#assert spotify_song_array.shape==(3706389, 179)

spotify_song_array_glove = hkl.load('song_embedding_matrix_150.hkl')

gc.collect()

spotify_song_array = np.concatenate((spotify_song_array,spotify_song_array_glove),axis=1)


le_context_type = joblib.load(data_path+'le_context_type.pkl')
le_reason_start = joblib.load(data_path+'le_reason_start.pkl')
le_reason_end = joblib.load(data_path+'le_reason_end.pkl')

n_context_type = len(le_context_type.classes_) + 1
n_reason_start = len(le_reason_start.classes_) + 1
n_reason_end = len(le_reason_end.classes_) + 1


test_files = glob.glob(test_data_path + 'pred_*.parquet')
#print(len(test_files))
test_files = list(np.sort(test_files))

test_files = list(np.sort(test_files))

#print(test_files)

###############################################################################

def TestPredProcessSessionCate(pred_df):
    session_cols = ['session_id', 'session_position', 'session_length', 'track_id_clean']

    pred_df['track_id_clean'] = pred_df['track_id_clean'] + 1
    
    output_raw_data = np.array(pred_df[session_cols].values) * 1
    output_raw_data = output_raw_data.astype(np.float)

    output_raw_data_first_p = np.floor(output_raw_data[:, 2] / 2).astype(np.int)
    output_raw_data_first_p = -output_raw_data_first_p + output_raw_data[:, 1]

    n_session = int(np.max(output_raw_data[:,0])) + 1
    gc.collect()
    output_data = -2 * np.ones((n_session * 10, 2))
    
    output_raw_data[:,2] = output_raw_data[:,1]/output_raw_data[:,2]

    output_data[output_raw_data[:, 0].astype(np.int) * 10 + output_raw_data_first_p.astype(np.int) - 1,
    :] = output_raw_data[:, 2:4]
    output_data = np.reshape(output_data, (n_session, 10, output_data.shape[1]))

    output_data_id = output_data[:,:,1]
    
    output_data = output_data[:,:,0]
    
    output_data_id[output_data_id<0] = 0    
    output_data_id = output_data_id.astype(int)

    return output_data, output_data_id


def TestPreHistSessionCate(pre_hist_df):
    pre_hist_df['context_type'] = le_context_type.transform(pre_hist_df['context_type'])
    pre_hist_df['hist_user_behavior_reason_start'] = le_reason_start.transform(pre_hist_df['hist_user_behavior_reason_start'])
    pre_hist_df['hist_user_behavior_reason_end'] = le_reason_end.transform(pre_hist_df['hist_user_behavior_reason_end'])
    session_cols = ['session_id', 'session_position', 'session_length', 'track_id_clean',
                    'skip_1', 'skip_2', 'skip_3', 'not_skipped', 'context_switch',
                    'no_pause_before_play', 'short_pause_before_play',
                    'long_pause_before_play', 'hist_user_behavior_n_seekfwd',
                    'hist_user_behavior_n_seekback', 'hist_user_behavior_is_shuffle',
                    'hour_of_day', 'premium', 'context_type',
                    'hist_user_behavior_reason_start', 'hist_user_behavior_reason_end']

    pre_hist_df['track_id_clean'] = pre_hist_df['track_id_clean'] + 1
    pre_hist_df['hour_of_day'] = pre_hist_df['hour_of_day']/24

    input_raw_data = np.array(pre_hist_df[session_cols].values) * 1
    input_raw_data = input_raw_data.astype(np.float)

    input_raw_data_final_p = np.floor(input_raw_data[:, 2]/2).astype(np.int)
    input_raw_data_final_p = 10 - input_raw_data_final_p + input_raw_data[:, 1]

    n_session = int(np.max(input_raw_data[:,0])) + 1
    gc.collect()
    input_data = -2 * np.ones((n_session * 10, 18))

    input_data[:, 17] = n_reason_end - 1    
    input_data[:, 16] = n_reason_start - 1    
    input_data[:, 15] = n_context_type - 1
    
    input_raw_data[:,2] = input_raw_data[:,1]/input_raw_data[:,2]

    input_data[input_raw_data[:, 0].astype(np.int) * 10 + input_raw_data_final_p.astype(np.int) - 1,
    :] = input_raw_data[:, 2:20]
    input_data = np.reshape(input_data, (n_session, 10, input_data.shape[1]))

    input_data_context_id = input_data[:,:,15]
    input_data_start_id = input_data[:,:,16]
    input_data_end_id = input_data[:,:,17]

    input_data = input_data[:,:,range(15)]

    input_data_id = input_data[:,:,1]
    input_data = np.delete(input_data, 1, 2)
    
    input_data_context_id[input_data_context_id<0] = 0
    input_data_start_id[input_data_start_id<0] = 0
    input_data_end_id[input_data_end_id<0] = 0
    input_data_id[input_data_id<0] = 0
    
    input_data_context_id = input_data_context_id.astype(int)
    input_data_start_id = input_data_start_id.astype(int)
    input_data_end_id = input_data_end_id.astype(int)
    input_data_id = input_data_id.astype(int)
    

    return input_data, input_data_context_id, input_data_start_id, input_data_end_id,input_data_id


###############################################################################

#For v5

elements = [-2,-1,0,1,2]
probabilities = [0.1,0.25,0.3,0.25,0.1]
#np.random.choice(elements, 10, p=probabilities)

def ProcessSessionCate(df, if_gen = 0, batch_size = batch_size, sample = 0):
    df['date'] = pd.to_datetime(df['date'] )
        
    df['date_gap'] = df['date'] - pd.Timestamp(2018, 7, 13)
    df['date_gap'] = df['date_gap'].dt.days
    
    
    df['context_type'] = le_context_type.transform(df['context_type'])
    df['hist_user_behavior_reason_start'] = le_reason_start.transform(df['hist_user_behavior_reason_start'])
    df['hist_user_behavior_reason_end'] = le_reason_end.transform(df['hist_user_behavior_reason_end'])
    session_cols = ['session_id', 'session_position', 'session_length', 'track_id_clean',
                    'skip_1', 'skip_2', 'skip_3', 'not_skipped', 'context_switch',
                    'no_pause_before_play', 'short_pause_before_play',
                    'long_pause_before_play', 'hist_user_behavior_n_seekfwd',
                    'hist_user_behavior_n_seekback', 'hist_user_behavior_is_shuffle',
                    'hour_of_day', 'premium', 'context_type',
                    'hist_user_behavior_reason_start', 'hist_user_behavior_reason_end','date_gap']
    
    session_id = np.unique(df['session_id'])    
    session_position_offset = np.random.choice(elements, len(session_id), p=probabilities)
    df_offset = pd.DataFrame()
    df_offset['session_id'] = session_id
    if sample == 1:
        df_offset['session_position_offset'] = session_position_offset
    else:
        df_offset['session_position_offset'] = 0
        
    n0 = df.shape[0]
        
    df = pd.merge(df, df_offset, on = 'session_id', how = 'left')
    df['session_position'] = df['session_position'] + df['session_position_offset']
    df = df[df['session_position']<=df['session_length']].reset_index(drop=True)
    df = df[df['session_position']>=1].reset_index(drop=True)
    
    n1 = df.shape[0]
    
    #print('shrinkage:',n0, n1)

    df['track_id_clean'] = df['track_id_clean'] + 1
    df['hour_of_day'] = df['hour_of_day']/24

    raw_data = np.array(df[session_cols].values) * 1
    raw_data = raw_data.astype(np.float)

    input_raw_data = raw_data[raw_data[:,1] * 2 <= raw_data[:, 2], :]
    output_raw_data = raw_data[raw_data[:,1] * 2 > raw_data[:, 2], :]

    input_raw_data_final_p = np.floor(input_raw_data[:, 2] / 2).astype(np.int)
    input_raw_data_final_p = 10 - input_raw_data_final_p + input_raw_data[:, 1]

    output_raw_data_first_p = np.floor(output_raw_data[:, 2] / 2).astype(np.int)
    output_raw_data_first_p = -output_raw_data_first_p + output_raw_data[:, 1]

    n_session = int(np.max(raw_data[:,0])) + 1
    gc.collect()
    input_data = -2 * np.ones((n_session * 10, 18))
    output_data = -2 * np.ones((n_session * 10, 19))

    input_data[:, 17] = n_reason_end - 1
    output_data[:, 17] = n_reason_end - 1
    
    input_data[:, 16] = n_reason_start - 1
    output_data[:, 16] = n_reason_start - 1
    
    input_data[:, 15] = n_context_type - 1
    output_data[:, 15] = n_context_type - 1
    
    '''    
    output_premium = input_data[:,9,14]    
    output_premium = np.reshape(output_premium,(len(output_premium),1,1))
    output_premium = np.tile(output_premium,(1,10,1))
    '''
    
    input_raw_data[:,2] = input_raw_data[:,1]/input_raw_data[:,2]
    output_raw_data[:,2] = output_raw_data[:,1]/output_raw_data[:,2]


    input_data[input_raw_data[:, 0].astype(np.int) * 10 + input_raw_data_final_p.astype(np.int) - 1,
    :] = input_raw_data[:, 2:20]
    output_data[output_raw_data[:, 0].astype(np.int) * 10 + output_raw_data_first_p.astype(np.int) - 1,
    :] = output_raw_data[:, 2:21]
    input_data = np.reshape(input_data, (n_session, 10, input_data.shape[1]))
    output_data = np.reshape(output_data, (n_session, 10, output_data.shape[1]))
    output_data_target = output_data[:, :, 2:13]
    output_data_target[output_data_target>1] = 1
    output_data_target[output_data_target < 0] = 0
    
    output_date_id = output_data[:, :, 18]
    
    #output_data_target[:,:,3] = 1 - output_data_target[:,:,1]   
    
    output_data_id = output_data[:,:,1]

    order = np.arange(n_session)
    random.shuffle(order)
    
    if if_gen == 1:
        order = order[0:int(batch_size*np.floor(n_session/batch_size))]        

    input_data = input_data[order,:,:]  
 
    input_data_context_id = input_data[:,:,15]
    input_data_start_id = input_data[:,:,16]
    input_data_end_id = input_data[:,:,17]

    input_data = input_data[:,:,range(15)]

    input_data_id = input_data[:,:,1]
    input_data = np.delete(input_data, 1, 2)
    output_data_id = output_data_id[order,:]
    output_data_target = output_data_target[order,:]
    
    output_data = output_data[:,:,0]
    
    input_data_context_id[input_data_context_id<0] = 0
    input_data_start_id[input_data_start_id<0] = 0
    input_data_end_id[input_data_end_id<0] = 0
    input_data_id[input_data_id<0] = 0
    output_data_id[output_data_id<0] = 0
    
    input_data_context_id = input_data_context_id.astype(int)
    input_data_start_id = input_data_start_id.astype(int)
    input_data_end_id = input_data_end_id.astype(int)
    input_data_id = input_data_id.astype(int)
    output_data_id = output_data_id.astype(int)

    return input_data, output_data, input_data_context_id, input_data_start_id, input_data_end_id,\
           input_data_id, output_data_id, output_data_target, output_date_id
           
###############################################################################


# Specify the train and validation files.
#train_files = glob.glob(data_path + '*.csv.parquet')
#len(train_files)
valid_files = ['Skip_Data/log_0_20180715_000000000000.csv.parquet']
#train_files = list(set(train_files)-set(valid_files))
#print(len(train_files))

###############################################################################

# Constructing the validation set.

count = 0
for file in valid_files:
    tmp_data = pd.read_parquet(file)
    input_data_i, output_data_i, input_data_context_id_i, input_data_start_id_i, input_data_end_id_i, \
    input_data_id_i, output_data_id_i, output_data_target_i = ProcessSessionCate(tmp_data)

    if count == 0:
        valid_input = input_data_i
        valid_output = output_data_i
        valid_input_id = input_data_id_i
        valid_input_data_context_id = input_data_context_id_i
        valid_input_data_start_id = input_data_start_id_i
        valid_input_data_end_id = input_data_end_id_i
        valid_output_id = output_data_id_i
        valid_target = output_data_target_i
    else:
        valid_input = np.concatenate((valid_input,input_data_i),axis=0)
        valid_output = np.concatenate((valid_output,output_data_i),axis=0)
        valid_input_data_context_id = np.concatenate((valid_input_data_context_id,input_data_context_id_i),axis=0)
        valid_input_data_start_id = np.concatenate((valid_input_data_start_id,input_data_start_id_i),axis=0)
        valid_input_data_end_id = np.concatenate((valid_input_data_end_id,input_data_end_id_i),axis=0)
        valid_input_id = np.concatenate((valid_input_id,input_data_id_i),axis=0)
        valid_output_id = np.concatenate((valid_output_id,output_data_id_i),axis=0)
        valid_target = np.concatenate((valid_target,output_data_target_i),axis=0)

    count  = count + 1
    
input_fea_dim = valid_input.shape[2]

X_valid = {
   'context': valid_input_data_context_id,
    'start': valid_input_data_start_id,
    'end': valid_input_data_end_id,
    'input_fea': valid_input,
     'output_fea': valid_output,
    'input_id': valid_input_id,
    'output_id': valid_output_id,
}

Y_valid = valid_target

#hkl.dump(Y_valid, data_path+'Y_valid.hkl', mode='w', compression='gzip')

#hkl.dump(valid_output, data_path+'valid_output.hkl', mode='w', compression='gzip')
###############################################################################

def tile_tile(X):
    X = K.tile(X, [1,10,1]) 
    return X


def exp_smooth5(X):   
    weight = np.power(0.5,9 - np.arange(10))
    weight = weight/np.sum(weight)
    weight = np.reshape(weight, (1,10,1))
    att_output_weight = K.variable(weight)
    print(att_output_weight.shape)
    att_output_result = X*att_output_weight    
    print(X.shape)
    att_output_result = K.mean(att_output_result, axis=1)
    
    return att_output_result

def exp_smooth8(X):   
    weight = np.power(0.8,9 - np.arange(10))
    weight = weight/np.sum(weight)
    weight = np.reshape(weight, (1,10,1))
    att_output_weight = K.variable(weight)
    print(att_output_weight.shape)
    att_output_result = X*att_output_weight    
    print(X.shape)
    att_output_result = K.mean(att_output_result, axis=1)
    
    return att_output_result


class mstk_ndcg_callback(Callback):
    def __init__(self):
        return

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):      

        y_pred = self.model.predict(X_valid,batch_size=800)
        
        hkl.dump(y_pred, data_path+'y_pred.hkl', mode='w', compression='gzip')
        
        y_pred = y_pred[:,:,1]
        
        Y_valid_i = Y_valid[:,:,1]
        
        Y_pred = 0*y_pred
        Y_pred[y_pred>=0.5] = 1
        
        Y_pred = (Y_pred==Y_valid_i)
        
        Y_pred = Y_pred[valid_output>=0]
        
        #print(Y_pred.shape)

        print(np.mean(Y_pred))
        
        #hkl.dump(song_embedding_matrix, data_path+'song_embedding_matrix_80.hkl', mode='w', compression='gzip')

        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return

#K.clear_session()

#spotify_song_array = np.ones((100,2))
#input_fea_dim = 5
#n_context_type = 4
#n_reason_start = 3
#n_reason_end = 3

def skip_model_5_mtsk_att(cell_size = 350):

    input_data_context_id = Input(shape=[10], name="context")
    input_data_start_id = Input(shape=[10], name="start")
    input_data_end_id = Input(shape=[10], name="end")
    input_data = Input(shape=[10,14], name="input_fea")
    output_data_i = Input(shape=[10], name="output_fea")
    input_id = Input(shape=[10], name="input_id")
    output_id = Input(shape=[10], name="output_id")

    song_emb_layer = Embedding(
        input_dim=spotify_song_array.shape[0],
        output_dim=spotify_song_array.shape[1],
        weights=[spotify_song_array],
        trainable=False
    )

    context_emb_layer = Embedding(n_context_type,25)
    reason_start_emb_layer = Embedding(n_reason_start,25)
    reason_end_emb_layer = Embedding(n_reason_end,25)

    emb_input_id = song_emb_layer(input_id)
    emb_output_id = song_emb_layer(output_id)
    emb_input_data_context_id = context_emb_layer(input_data_context_id)
    emb_input_data_start_id = reason_start_emb_layer(input_data_start_id)
    emb_input_data_end_id = reason_end_emb_layer(input_data_end_id)

    input_data_a = concatenate([input_data,emb_input_id,emb_input_data_context_id,
                              emb_input_data_start_id,emb_input_data_end_id])

    # rnn layers
#    encoder_outputs, state_h
    
    encoder_outputs, rnn_layer = GRU(cell_size, return_sequences=True, return_state=True) (input_data_a)
    
    encoder_outputs_1, rnn_layer_1 = GRU(cell_size, return_sequences=True, return_state=True) (encoder_outputs)
    
    #rnn_layer_5 = Lambda(exp_smooth5)(encoder_outputs)    
    #rnn_layer_8 = Lambda(exp_smooth8)(encoder_outputs)    
#    print(encoder_outputs.shape)
#    print(rnn_layer_5.shape)
#    print(rnn_layer.shape)

    
    rnn_layer = concatenate([rnn_layer, rnn_layer_1])
    
    output_data = Reshape([10,1])(output_data_i)
    
    output_data = concatenate([output_data, emb_output_id])

    output_data = Dense(2*cell_size, activation='relu')(output_data)

    rnn_layer_multi = Multiply()([output_data, rnn_layer])
    
    rnn_layer_reshape = Reshape([1,2*cell_size])(rnn_layer)
    
    rnn_layer_reshape = Lambda(tile_tile)(rnn_layer_reshape)

    output_result = concatenate([rnn_layer_reshape, rnn_layer_multi, output_data])
    
    output_rnn_layer = Bidirectional(GRU(cell_size, return_sequences=True))(output_result)
    
    output_rnn_layer_2 = Bidirectional(GRU(cell_size, return_sequences=True))(output_rnn_layer)
    
    output_rnn_layer_3 = Bidirectional(GRU(200, return_sequences=True))(output_rnn_layer_2)
    
    output_rnn_layer_4 = Bidirectional(GRU(200, return_sequences=True))(output_rnn_layer_3)
    
    output_result = concatenate([output_result, output_rnn_layer, output_rnn_layer_2, output_rnn_layer_3, output_rnn_layer_4])

    output_result = Dense(1000, activation='relu')(output_result)
    output_result_1 = Dense(784, activation='relu')(output_result)
    #output_result_2 = Dense(784, activation='elu')(output_result_1)
    
    #output_result = concatenate([output_result, output_result_1, output_result_2])
    
    output_result = Dropout(0.2)(output_result_1)

    output_result = Dense(11, activation='sigmoid')(output_result)
    
    #output_result = Reshape([10])(output_result)


    model = Model(inputs=[input_data_context_id, input_data_start_id,
                          input_data_end_id, input_data, output_data_i, input_id, output_id], outputs=output_result)

    sgd = Adam(lr=0.0008)
    model.compile(loss='binary_crossentropy',
                  optimizer=sgd,
                  metrics=['binary_crossentropy', 'binary_accuracy'])
    model.summary()
    return model

model = skip_model_5_mtsk_att()

model.load_weights('Data/skip_net_glove_max_mtsk_more_layer_best_v6.hdf5')

###############################################################################

for item in tqdm(test_files):
    preddf = pd.read_parquet(item)
    
    prehistdf = pd.read_parquet(test_data_path + 'prehist_' + item[15:])   
    
    cols = ['session_id', 'session_position', 'session_length', 'track_id_clean']
    
    preddf = preddf[cols]    
    
    input_data, input_data_context_id, input_data_start_id, input_data_end_id,input_data_id = TestPreHistSessionCate(prehistdf)
    
    output_data, output_data_id = TestPredProcessSessionCate(preddf)
    
    X_test = {
       'context': input_data_context_id,
        'start': input_data_start_id,
        'end': input_data_end_id,
        'input_fea': input_data,
         'output_fea': output_data,
        'input_id': input_data_id,
        'output_id': output_data_id,
        }
    
    y_pred = model.predict(X_test,batch_size=1500,verbose=1)
    y_pred = y_pred[:,:,1]
    
    print(np.mean(y_pred))
        
    hkl.dump(y_pred, data_path+item[15:23]+'_y_pred_mtsk_larger_v6.hkl', mode='w', compression='gzip')
    
