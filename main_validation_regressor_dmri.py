#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 15:38:08 2022

@author: fa19
"""

import numpy as np

import torch
import torch.nn as nn



from torch_geometric.data import DataLoader

import copy
import nibabel as nb

from my_utils import save_as_metric

from data_utils.MyDataLoader import My_dHCP_Data_Graph, My_dHCP_Data, My_dmri_Data_Graph

from torch_geometric.data import DataLoader

import copy
import nibabel as nb

from my_utils import save_as_metric
dataset = 'dmri1'

train_dataset_arr = np.load('data/' + str(dataset) + '/train1.npy', allow_pickle = True).astype(object)
val_dataset_arr = np.load('data/' + str(dataset) + '/val1.npy', allow_pickle = True).astype(object)
test_dataset_arr = np.load('data/' + str(dataset) + '/test1.npy', allow_pickle = True).astype(object)

train_dataset_arr[:,1] = train_dataset_arr[:,1].astype(np.float)
train_dataset_arr[:,2] = train_dataset_arr[:,2].astype(np.float)
val_dataset_arr[:,1] = val_dataset_arr[:,1].astype(np.float)
val_dataset_arr[:,2] = val_dataset_arr[:,2].astype(np.float)

test_dataset_arr[:,1] = test_dataset_arr[:,1].astype(np.float)
test_dataset_arr[:,2] = test_dataset_arr[:,2].astype(np.float)

train_dataset_arr = train_dataset_arr[:,:2]
val_dataset_arr = val_dataset_arr[:,:2]
test_dataset_arr = test_dataset_arr[:,:2]


model_dir = '/home/fa19/Documents/neurips/dmri1/'
model_name = '3cycle_dmri_generator_final2'


save_root = '/data/'

save_loc = 'dmri1_images/'
save_dir = save_root + save_loc


regressor_model_location = '/home/fa19/Documents/neurips/dmri1/regressor_monet_dmri_std'



train_rots = False 
num_warps = 0
registered = True

train_parity = 'both'
norm_style = 'std'
test_parity = train_parity


if registered == True:
    
    warped_directory = '/data/dmir3'
    unwarped_directory = '/data/dmri3'
    
else:
    
    unwarped_directory = '/data/dmir3'
    warped_directory = '/data/dmir3'
    
    

edges = torch.LongTensor(np.load('data/edge_ico_6.npy').T)




device_number = 0

device = torch.device('cuda:' + str(device_number) if torch.cuda.is_available() else 'cpu')
print('device is ', device)
torch.cuda.set_device(device)


from markovian_models import GraphNet_Markovian_Generator, GraphNet_Markovian_Discriminator, GraphNet_Markovian_Discriminator_Simple

modality = 'all'

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

from markovian_models import GraphNet_Markovian_Generator, MoNet_Markovian_Generator, GraphNet_Markovian_Discriminator, GraphNet_Markovian_Discriminator_Simple
from regressor_model_monet import monet_polar_regression

model = GraphNet_Markovian_Generator(len(mode), age_dim = 1,device=device)

if 'MoNet' in model_name:
    model = MoNet_Markovian_Generator(len(mode), age_dim = 1,device=device)
model = model.to(device)
model = torch.load(model_dir+model_name)

model.eval()



regressor = monet_polar_regression([in_channels, 32, 64 , 128, 256])
regressor = torch.load(regressor_model_location)

regressor = regressor.to(device)
regressor.eval()

################### finihed loading the model  and the regressor #################


# get only the terms from the test set

test_ds = My_dmri_Data_Graph(input_arr = test_dataset_arr, 
                    warped_files_directory = warped_directory,
                    unwarped_files_directory = unwarped_directory,
                    edges=edges,
                    rotations= False,
                    number_of_warps = 0,
                    parity_choice = 'left',
                    smoothing = False,
                    normalisation = norm_style,
                    projected = False,
                    sample_only = True, #if false, will go through every warp, not just one random warp
                    output_as_torch = True,
                    )

test_loader = DataLoader(test_ds, batch_size=1, 
                                            shuffle=False, 
                                            num_workers = 2)

################### create  longitudinal data / preterms #################

def restandardise(arr):
    for i in range(arr.shape[1]):
        arr[:,i] = (arr[:,i] - torch.mean(arr[:,i])) / torch.std(arr[:,i])
    return arr
    
count = 0
all_l1_losses = []

all_age_predictions = []
for i, data in enumerate(test_loader):
    subject_l1_loss = []
    subject_age_predictions = []
    # row = test_dataset_arr[i % 42]
    name = test_dataset_arr[i,0].split('_')[0]
    
    
    
    im1 = data['x'][:,mode].to(device) 
        
            
                                            
    # im1 =  data['x'][:,mode].to(device) 
        
    bs = data.x.shape[0]//40962

    batch = data.batch
    edge = data.edge_index.to(device)
    true_age = data['y'].to(device)
    
    # true_age_1_v = torch_age_to_cardinal(true_age_1).to(device)
    true_age_1_v = true_age.to(device)
    
    
    for num in np.arange(32,46):
        
        
        
        new_age = torch.Tensor([num]).to(device)
        if len(new_age.shape)==1 :
            new_age.unsqueeze(0)
        # new_age_v = torch_age_to_cardinal(new_age).to(device)
        new_age_v = new_age.to(device)
        
        difference = new_age_v - true_age_1_v
        
        im2 = model(im1, difference.unsqueeze(0))
        # if 'MoNet' in model_name:
            # im2 = restandardise(im2)

        save_as_metric(im2.cpu().detach(), save_dir  + str(name)+ '_'+str(num))
        new_age_prediction = regressor(im2, edge, batch)
        subject_age_predictions.append(new_age_prediction.item())
        l1_loss = nn.L1Loss()(new_age_prediction, new_age_v)
        subject_l1_loss.append(l1_loss.item())
    all_l1_losses.append(subject_l1_loss)
    all_age_predictions.append(subject_age_predictions)
    print(i)


average_losses_per_subject = np.mean(np.array(all_l1_losses), axis=1)

mean_losses_overal = np.mean(average_losses_per_subject)
std_losses_overall = np.std(average_losses_per_subject)


print(mean_losses_overal, std_losses_overall)

all_age_predictions = np.array(all_age_predictions)
mean_predictions = np.mean(all_age_predictions, axis=0)
L = np.arange(32,46)
import matplotlib.pyplot as plt
# a, b = np.polyfit(L, all_age_predictions, 1)

fig = plt.figure()
plt.scatter(L, mean_predictions)
plt.plot(L, L)
plt.show()

