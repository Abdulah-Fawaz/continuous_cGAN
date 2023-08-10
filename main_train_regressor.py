#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 13:54:15 2022

@author: fa19
"""



import numpy as np

import torch
import torch.nn as nn


from data_utils.MyDataLoader import My_dHCP_Data_Graph, My_dHCP_Data

from torch_geometric.data import DataLoader

import copy
import nibabel as nb

from my_utils import save_as_metric
dataset = 'scan_age'

train_dataset_arr = np.load('data/' + str(dataset) + '/train.npy', allow_pickle = True)
val_dataset_arr = np.load('data/' + str(dataset) + '/validation.npy', allow_pickle = True)
# test_dataset_arr = np.load('data/' + str(dataset) + '/test.npy', allow_pickle = True)
# full_dataset_arr = np.load('data/' + 'full' + '/full.npy', allow_pickle = True)

train_rots = False 
num_warps = 100
registered = True

train_parity = 'both'
norm_style = 'std'
test_parity = train_parity

if registered == True:
    
    warped_directory = '/home/fa19/Documents/dHCP_Data_merged/Warped'
    unwarped_directory = '/home/fa19/Documents/dHCP_Data_merged/merged'
    
else:
    
    unwarped_directory = '/home/fa19/Documents/dHCP_Data_merged/derivatives_native_ico6'
    warped_directory = '/data/warped_native'
    

edges = torch.LongTensor(np.load('data/edge_ico_6.npy').T)





train_ds = My_dHCP_Data_Graph(input_arr = train_dataset_arr, 
                   warped_files_directory = warped_directory,
                   unwarped_files_directory = unwarped_directory,
                   edges=edges,
                   rotations= False,
                   number_of_warps = num_warps,
                   parity_choice = train_parity,
                   smoothing = True,
                   normalisation = norm_style,
                   projected = False,
                   sample_only = True, #if false, will go through every warp, not just one random warp
                   output_as_torch = True,
                   )


val_ds = My_dHCP_Data_Graph(input_arr = val_dataset_arr, 
                    warped_files_directory = warped_directory,
                    unwarped_files_directory = unwarped_directory,
                    edges=edges,
                    rotations= False,
                    number_of_warps = 0,
                    parity_choice = test_parity,
                    smoothing = True,
                    normalisation = norm_style,
                    projected = False,
                    sample_only = True, #if false, will go through every warp, not just one random warp
                    output_as_torch = True,
                    )


# val_ds_unwarped = My_dHCP_Data_Graph(input_arr = val_dataset_arr, 
#                    warped_files_directory ='/home/fa19/Documents/dHCP_Data_merged/Warped',
#                    unwarped_files_directory = '/home/fa19/Documents/dHCP_Data_merged/merged',
#                    edges=edges,
#                    rotations= False,
#                    number_of_warps = 0,
#                    parity_choice = test_parity,
#                    smoothing = False,
#                    normalisation = norm_style,
#                    projected = False,
#                    sample_only = True, #if false, will go through every warp, not just one random warp
#                    output_as_torch = True,
#                    )


# test_ds = My_dHCP_Data_Graph(input_arr = test_dataset_arr, 
#                    warped_files_directory = warped_directory,
#                    unwarped_files_directory = unwarped_directory,
#                    edges=edges,
#                    rotations= False,
#                    number_of_warps = 0,
#                    parity_choice = test_parity,
#                    smoothing = False,
#                    normalisation = norm_style,
#                    projected = False,
#                    sample_only = True, #if false, will go through every warp, not just one random warp
#                    output_as_torch = True,
#                    )

# full_ds = My_dHCP_Data_Graph(input_arr = full_dataset_arr, 
#                    warped_files_directory = warped_directory,
#                    unwarped_files_directory = unwarped_directory,
#                    edges=edges,
#                    rotations= False,
#                    number_of_warps = 0,
#                    parity_choice = test_parity,
#                    smoothing = False,
#                    normalisation = norm_style,
#                    projected = False,
#                    sample_only = True, #if false, will go through every warp, not just one random warp
#                    output_as_torch = True,
#                    )


batch_size = 1


weights = np.ones(len(train_ds))
frac_prems = sum(train_dataset_arr[:,1]<37)/len(train_dataset_arr)

weight_prems = 3

positions = [i for i in range(len(train_dataset_arr)) if train_dataset_arr[i,-1]<37]
positions2 = [i for i in range(len(train_dataset_arr), 2*len(train_dataset_arr)) if train_dataset_arr[i//2,-1]<37]

positions = positions + positions2

weights[positions] = weight_prems
train_loader = DataLoader(train_ds, batch_size=batch_size, 
                                            sampler = torch.utils.data.WeightedRandomSampler(weights=weights, num_samples=len(train_ds)),

                                            num_workers = 1)




val_loader = DataLoader(val_ds, batch_size=1, 
                                            shuffle=False, 
                                            num_workers = 2)


# val_unwarped_loader = DataLoader(val_ds_unwarped, batch_size=1, 
#                                            shuffle=False, 
#                                            num_workers = 2)


# test_loader = DataLoader(test_ds, batch_size=1, 
#                                            shuffle=False, 
#                                            num_workers = 2)


# full_loader = DataLoader(full_ds, batch_size=1, 
#                                            shuffle=False, 
#                                            num_workers = 2)




device_number = 0

device = torch.device('cuda:' + str(device_number) if torch.cuda.is_available() else 'cpu')
print('device is ', device)
torch.cuda.set_device(device)


from markovian_models import GraphNet_Regressor
from regressor_model_monet import monet_polar_regression
modality = 'both'

if modality == 'myelination':
    mode = 0
    in_channels = 1

elif modality == 'curvature':
    mode = 1
    in_channels = 1

elif modality == 'sulc':
    mode = [3]
    in_channels = 1

elif modality == 'both':
    mode = [0,3]
    in_channels = 2

elif modality == 'all':
    mode = [0,1,2,3]
    in_channels = 4
    
elif modality == 'triple':
    mode = [0,1,3]
    in_channels = 3
    
elif modality == 'both_curvature':
    mode = [0,1]
    in_channels = 2


def torch_age_to_cardinal(x, L = 30 , minimum = 20):
    # if len(x.shape)==1:
    #     x = x.unsqueeze(0)
    # print(x.shape)
    if x.type() != 'torch.LongTensor' and x.type() != 'torch.cuda.LongTensor':
        x = torch.round(x)
    x = x - minimum
    out = torch.zeros(x.shape[0],L).to(x.device)
    
    for i in range(x.shape[0]):
        b = x[i]
        # print(b)
        b = int(b.item())
        out[i,:b] = 1
        
    return out

    


model = monet_polar_regression([in_channels, 32, 64 , 128, 256])
model = torch.load('/home/fa19/Documents/neurips/regressor_monet_std2')
model = model.to(device)



learning_rate = 5e-4

optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay= 1e-5)

best = 100
num_epochs = 500
print('Starting...')


test_criterion = nn.L1Loss()

criterion = nn.SmoothL1Loss()

val_check_frequency = 1
for epoch in range(num_epochs):
    train_loss = []
    model.train()
    d_losses = []
    g_losses = []
    
    for i, data in enumerate(train_loader):
        model.train()
        im1 =  data['x'][:,mode].to(device) 
        edge = data['edge_index'].to(device)
        
        bs = data.batch


        true_age = data['y'].to(device)
      
        optimizer.zero_grad()

        prediction = model(im1, edge, bs)
   
        loss = criterion(prediction, true_age)

        
      
        loss.backward()
        train_loss.append(loss.item())
        
        optimizer.step()
    print(str(epoch) , np.mean(train_loss))
    
    if epoch % val_check_frequency == 0:
        val_losses = []
        for i, data in enumerate(val_loader):
            model.eval()
            
            
            im1 =  data['x'][:,mode].to(device)
            
            bs = data.batch


            true_age = data['y'].to(device)
          

            prediction = model(im1, edge, bs)
       
            val_loss = test_criterion(prediction, true_age)

            
          
            val_losses.append(loss.item())
        
        
        print('Validation ', np.mean(val_losses))
        

test_dataset_arr = np.load('data/' + str(dataset) + '/test.npy', allow_pickle = True)


test_ds = My_dHCP_Data_Graph(input_arr = test_dataset_arr, 
                    warped_files_directory = warped_directory,
                    unwarped_files_directory = unwarped_directory,
                    edges=edges,
                    rotations= False,
                    number_of_warps = 0,
                    parity_choice = test_parity,
                    smoothing = True,
                    normalisation = norm_style,
                    projected = False,
                    sample_only = True, #if false, will go through every warp, not just one random warp
                    output_as_torch = True,
                    )

tv_ds = My_dHCP_Data_Graph(input_arr = np.vstack([test_dataset_arr, val_dataset_arr]), 
                    warped_files_directory = warped_directory,
                    unwarped_files_directory = unwarped_directory,
                    edges=edges,
                    rotations= False,
                    number_of_warps = 0,
                    parity_choice = test_parity,
                    smoothing = True,
                    normalisation = norm_style,
                    projected = False,
                    sample_only = True, #if false, will go through every warp, not just one random warp
                    output_as_torch = True,
                    )


t_loader = DataLoader(test_ds, batch_size=1, 
                                            shuffle=False, 
                                            num_workers = 2)


test_loader = DataLoader(tv_ds, batch_size=1, 
                                            shuffle=False, 
                                            num_workers = 2)

test_losses = []
all_test_reals = []
all_test_preds = []


for i, data in enumerate(test_loader):
    model.eval()
    
    
    im1 =  data['x'][:,mode].to(device) 
    
    bs = data.batch


    true_age = data['y'].to(device)
    

    prediction = model(im1, edge, bs )
   
    test_loss = test_criterion(prediction, true_age)
    all_test_reals.append(true_age.item())
    all_test_preds.append(prediction.item())

  
    test_losses.append(test_loss.item())


print('Test ', np.mean(test_losses))



import numpy as np

a, b = np.polyfit(all_test_reals, all_test_preds, 1)

import matplotlib.pyplot as plt

fig = plt.figure()
plt.scatter(all_test_reals, all_test_preds)
plt.plot(all_test_reals, a * np.array(all_test_reals) + b)
plt.show()


all_test_preds = np.array(all_test_preds)
all_test_reals = np.array(all_test_reals)

from scipy import stats


y = all_test_preds
x = all_test_reals

a, b = np.polyfit(x, y, 1)

mean_preds = a * x + b

L = np.arange(32, 45)
L_preds = a * L + b


RSE = np.sum(np.square(np.abs(y - x))) / len(y)
r = np.abs(y - x) / RSE

r = np.sqrt(r)


ssx = np.sum(np.square(np.abs(y - x)))


# std_dev_residuals = np.sum(np.square(y-x))/(len(x)-2)

# factor = (1/len(x)) + np.square(x-y)/np.sum(np.square(x-y))

# S_mu = np.sqrt(std_dev_residuals * factor)

# t_factor = stats.t.ppf(0.95,len(L)-2)
# t_factor = 1.98
# S_t = S_mu * t_factor


yr = np.round(all_test_preds)
xr = np.round(x)

# diff = diff / np.sum(np.abs(yr-xr))
S = dict()
C = dict()
for i in range(len(xr)):
    S[xr[i]] = r[i] + S.get(xr[i], 0)
    C[xr[i]] = 1 + C.get(xr[i], 0)

new_list = np.array(list(S.values())) / np.array(list(C.values()))

# sdev_residuals = np.sqrt(new_list * t_factor)


indices = np.argsort(list(S.keys()))
S_t = new_list[indices]

nums = np.array(list(S.keys()))[indices]
nums_filters = nums[3:]
S_t = S_t[3:]

np.save("best_fit_regressor2_std2", S_t)

import matplotlib.pyplot as plt

fig = plt.figure()
plt.scatter(x, y)
plt.plot(all_test_reals, a * np.array(all_test_reals) + b)
plt.plot(L, L_preds + S_t)
plt.plot(L, L_preds - S_t)
plt.show()
# 




# torch.save(model,'/home/fa19/Documents/neurips/regressor_monet_std2')

# torch.save(model.state_dict(),'/home/fa19/Documents/neurips/regressor_monet_std2_sd')


# L = np.arange(32,45)
# np.save('best_fit_regressor2.npy',a*L+b)