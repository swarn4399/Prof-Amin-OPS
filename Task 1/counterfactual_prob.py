# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
test_data_1 = pd.read_parquet('C:/Users/13528/Task 1/test_pred/pred_20180715.parquet')

print(len(test_data_1["track_id_clean"]))
print(len(test_data_1["track_id_clean"].unique()))

indices = [ind for ind, ele in enumerate(test_data_1["track_id_clean"]) if ele == 671463]

from itertools import islice

with open("C:/Users/13528/Task 1/submission_0108_new.txt") as myfile:
    head = list(islice(myfile,444491))
# print(head)
for i in range(len(head)):
    head[i] = head[i].removesuffix('\n')
# print(head)
count = 0
# len(head)
for i in range(len(head)):
    for j in range(len(head[i])):
        count = count+1
print(count)

count = 0
arr = [len(head[0])-1]
for i in range(1,len(head)):
    arr.append(len(head[i])+arr[i-1])
# print(arr)  

arr1 = []
for i in range(len(head)):
    arr1.append(int(head[i])%10)
# print(arr1)

print(len(arr))
print(len(arr1))

session_last_tracks = test_data_1.iloc[arr]

session_last_tracks['skip'] = arr1

new_data = test_data_1

arr2 =[]
for i in range(len(head)):
    for j in range(len(head[i])):
        arr2.append(head[i][j])
print(len(arr2))

new_data['skip'] = arr2

new_data_1 = new_data.drop('session_id', axis=1)
new_data_1['skip'] = new_data_1['skip'].astype(int)

all_data = new_data_1

session_last_tracks = session_last_tracks.drop('session_id',axis= 1)

skipped_idx = [ind for ind, ele in enumerate(session_last_tracks['skip']) if ele == 0]

skipped_tracks = session_last_tracks.iloc[skipped_idx]

track_ids = np.array(all_data['track_id_clean'].unique())

###############################################
data_671463 = all_data.iloc[indices]

X=data_671463.iloc[:,[1,2]].values
Y=data_671463.iloc[:,-1].values

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size =0.1,random_state = 0)

# from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
# rfcmodel = RandomForestClassifier(max_depth=2, random_state=0)
dtmodel = DecisionTreeClassifier(random_state=0)
# rfcmodel.fit(X_train, Y_train)
dtmodel.fit(X_train, Y_train)

# predictions = rfcmodel.predict(X_test)
predictions = dtmodel.predict(X_test)

# rfcmodel.predict_proba(X_test)
dtmodel.predict_proba(X_test)

X_test_1 = [[12,16]]
print(dtmodel.predict_proba(X_test_1))





