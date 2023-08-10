#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  4 22:13:35 2022

@author: fa19
"""

import numpy as np

dataset = 'scan_age'

dataset2 = 'full'

scan_age_full = np.load('data/' + dataset + '/full.npy', allow_pickle = True)

scan_age_train = np.load('data/' + dataset + '/train.npy', allow_pickle = True)

scan_age_val = np.load('data/' + dataset + '/validation.npy', allow_pickle = True)
scan_age_test = np.load('data/' + dataset + '/test.npy', allow_pickle = True)



full_dataset = np.load('data/' + 'full' + '/full.npy', allow_pickle = True)



scan_age_nt = np.vstack([scan_age_val, scan_age_test])


scan_age_nt = full_dataset[[i for i in range(len(full_dataset)) if full_dataset[i,0] in scan_age_nt[:,0]]]
scan_age_train = full_dataset[[i for i in range(len(full_dataset)) if full_dataset[i,0] in scan_age_train[:,0]]]
scan_age_val = full_dataset[[i for i in range(len(full_dataset)) if full_dataset[i,0] in scan_age_val[:,0]]]
scan_age_test = full_dataset[[i for i in range(len(full_dataset)) if full_dataset[i,0] in scan_age_test[:,0]]]


# scan_age_nt = full_dataset[[i for i in range(len(full_dataset)) if full_dataset[i,0] in scan_age_nt[:,0]]]


remainder =  full_dataset[[i for i in range(len(full_dataset)) if full_dataset[i,0] not in scan_age_full[:,0]]]



test_dataset_arr = np.load('data/' + 'scan_age' + '/test.npy', allow_pickle = True)

everything_arr  = np.load('data/' +'full' + '/full.npy', allow_pickle = True)


test_names = [i.split('_')[0] for i in test_dataset_arr[:,0]]
all_shared = []
all_names = [i.split('_')[0] for i in everything_arr[:,0]]
for q in range(len(all_names)):
    n = all_names[q]
    
    shared = [i for i in range(len(everything_arr[:,0])) if n in everything_arr[i,0]]
    if len(shared) == 2:
        # print(everything_arr[shared], shared)
        if everything_arr[shared[0]][1] > everything_arr[shared[1]][1]:
            complete = shared[::-1]
        else:
            complete = shared
        complete.append(n)
        all_shared.append(complete)
        
        

all_shared_names =np.array( all_shared)[:,2]


remainder_train = remainder[[i for i in range(len(remainder)) if remainder[i,0].split('_')[0] not in all_shared_names]]

remainder_nt = remainder[[i for i in range(len(remainder)) if remainder[i,0].split('_')[0] in all_shared_names]]



new_train = np.vstack([ scan_age_train , remainder_train])

new_val  = np.vstack([scan_age_val, remainder_nt[:23]])

new_test  = np.vstack([scan_age_test, remainder_nt[23:]])


# np.save('data/ba/train.npy', new_train)
# np.save('data/ba/validation.npy', new_val)
# np.save('data/ba/test.npy', new_test)




