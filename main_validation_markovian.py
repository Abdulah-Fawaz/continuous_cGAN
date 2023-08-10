#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 15:38:08 2022

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


model_dir = '/home/fa19/Documents/neurips/'
model_name = '3cycle_generator_model_final'


save_root = '/home/fa19/Documents/neurips/'

save_loc = '3cycle_sa_final/'
save_dir = save_root + save_loc


train_dataset_arr = np.load('data/' + str(dataset) + '/train.npy', allow_pickle = True)
val_dataset_arr = np.load('data/' + str(dataset) + '/validation.npy', allow_pickle = True)
# test_dataset_arr = np.load('data/' + str(dataset) + '/test.npy', allow_pickle = True)
full_dataset_arr = np.load('data/' + 'full' + '/full.npy', allow_pickle = True)

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




device_number = 0

device = torch.device('cuda:' + str(device_number) if torch.cuda.is_available() else 'cpu')
print('device is ', device)
torch.cuda.set_device(device)


from markovian_models import GraphNet_Markovian_Generator, GraphNet_Markovian_Discriminator, GraphNet_Markovian_Discriminator_Simple

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

    

model = GraphNet_Markovian_Generator(len(mode), age_dim = 1,device=device)

model = model.to(device)
model = torch.load(model_dir+model_name)

model.eval()



################### finihed loading the model #################


test_dataset_arr = np.load('data/' + str(dataset) + '/test.npy', allow_pickle = True)


test_ds = My_dHCP_Data_Graph(input_arr = test_dataset_arr, 
                    warped_files_directory = warped_directory,
                    unwarped_files_directory = unwarped_directory,
                    edges=edges,
                    rotations= False,
                    number_of_warps = 0,
                    parity_choice = test_parity,
                    smoothing = False,
                    normalisation = norm_style,
                    projected = False,
                    sample_only = True, #if false, will go through every warp, not just one random warp
                    output_as_torch = True,
                    )



#
#
dataset = 'full'
everything_arr  = np.load('data/' + str(dataset) + '/full.npy', allow_pickle = True)

everything = My_dHCP_Data_Graph(input_arr = everything_arr, 
                   warped_files_directory = warped_directory,
                   unwarped_files_directory = unwarped_directory,
                   rotations= False,
                   number_of_warps = 0,
                   edges = edges,
                   smoothing = True,
                   normalisation = norm_style,
                   projected = False,
                   sample_only = True, #if false, will go through every warp, not just one random warp
                   output_as_torch = True,
                   )


test_names = [i.split('_')[0] for i in test_dataset_arr[:,0]]
all_shared = []
all_names = [i.split('_')[0] for i in everything_arr[:,0]]
for q in range(len(all_names)):
    n = all_names[q]
    
    shared = [i for i in range(len(everything_arr[:,0])) if n in everything_arr[i,0]]
    if len(shared) == 2:
        print(everything_arr[shared], shared)
        if everything_arr[shared[0]][1] > everything_arr[shared[1]][1]:
            complete = shared[::-1]
        else:
            complete = shared
        complete.append(n)
        all_shared.append(complete)
        

print(all_shared)



################### create  longitudinal data / preterms #################


count = 0

for idxs in all_shared:


    
    subject = idxs[2]
    
    idx1 = idxs[0]
    
    
    
    data = everything.__getitem__(idx1)
    im1 = data['x'][:,mode].to(device) 
        
            
                                            
    # im1 =  data['x'][:,mode].to(device) 
        
    bs = data.x.shape[0]//40962
    name = everything_arr[idx1,0].split('_')[0]
    
    true_age_1 = data['metadata'].to(device)
    
    # true_age_1_v = torch_age_to_cardinal(true_age_1).to(device)
    true_age_1_v = true_age_1.to(device)
    
    
    for num in np.arange(32,45):
        
        
        
        new_age = torch.Tensor([num]).to(device)
        if len(new_age.shape)==1 :
            new_age.unsqueeze(0)
        # new_age_v = torch_age_to_cardinal(new_age).to(device)
        new_age_v = new_age.to(device)
        
        difference = new_age_v - true_age_1_v
        
        im2 = model(im1, difference)
        
        im1_np = im1.reshape(bs,-1,len(mode)).detach().cpu().numpy()
        
        im2 = im2.reshape(bs,-1,len(mode)).detach().cpu().numpy()
         
        save_as_metric(im1_np[0], save_dir + 'val_original'+'_'+str(true_age_1[0].item())+'_'+str(name))
        
        save_as_metric(im2[0], save_dir + 'val_fake'+'_'+str(new_age[0].item())+'_'+str(name))
    
    
    
    
    idx2 = idxs[1]
    
    data2 = everything.__getitem__(idx2)
    im2 = data2['x'][:,mode].to(device) 
        
            
                                            
    # im1 =  data['x'][:,mode].to(device) 
        
    bs = data2.x.shape[0]//40962
    name = everything_arr[idx2,0].split('_')[0]
    
    true_age_2 = data2['metadata'].to(device)
    
    
    
    im2_np = im2.reshape(bs,-1,len(mode)).detach().cpu().numpy()
    
    save_as_metric(im2_np[0], save_dir+'val_original_secondscan'+'_'+str(true_age_2[0].item())+'_'+str(name))
    print(count)
    count+=1


################### quantify longitudinal data / preterms #################
import math
def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = original.max()
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    return psnr

    
    

birth_ages_list = []
second_scan_age_list = []


im_similarities_list = []


count = 0

for idxs in all_shared:


    
    subject = idxs[2]
    
    idx1 = idxs[0]
    
    
    
    data = everything.__getitem__(idx1)
    im1 = data['x'][:,mode]
    im1_np = np.array(im1)
    
    im1 = im1.to(device) 
       
            
    idx2 = idxs[1]
    
    data2 = everything.__getitem__(idx2)
    im2 = data2['x'][:,mode]
    
        
    
    
    
    
    im2_np = np.array(im2)
    im2 = im2.to(device)
                                            
    # im1 =  data['x'][:,mode].to(device) 
        
    bs = data.x.shape[0]//40962
    name = everything_arr[idx1,0].split('_')[0]
    
    true_age_1 = data['metadata'].to(device)
    
    # true_age_1_v = torch_age_to_cardinal(true_age_1).to(device)
    true_age_1_v = true_age_1.to(device)

    true_age_2_v = data2['metadata'].to(device)
    birth_age = data2['y']
    birth_ages_list.append(birth_age.item())
    second_scan_age_list.append(true_age_2_v.item())
    temp_similarity = []
    for num in np.arange(32,45):
        
        
        
        new_age = torch.Tensor([num]).to(device)
        if len(new_age.shape)==1 :
            new_age.unsqueeze(0)
        # new_age_v = torch_age_to_cardinal(new_age).to(device)
        new_age_v = new_age.to(device)
        
        difference = new_age_v - true_age_1_v
        
        imgen = model(im1, difference)
        
        im1_np = im1.reshape(bs,-1,len(mode)).detach().cpu().numpy()
        
        imgen = imgen.reshape(bs,-1,len(mode)).detach().cpu().numpy()
         
        
        temp_similarity.append( PSNR(np.array(imgen), im2_np))
    
        
    im_similarities_list.append(temp_similarity)
        


max_im_similarities = [np.argmax(i) for i in im_similarities_list]


A = np.arange(32,45)



max_age_similarities = np.array(max_im_similarities) + 32
diff_age_similarties = second_scan_age_list - max_age_similarities

import matplotlib.pyplot as plt
fig = plt.figure()
# plt.scatter(np.array(second_scan_age_list)-np.array(birth_ages_list), diff_age_similarties)
# plt.scatter(np.array(birth_ages_list), diff_age_similarties)
# plt.scatter(np.array(birth_ages_list), np.abs(diff_age_similarties))
plt.scatter(np.array(second_scan_age_list)-np.array(birth_ages_list), np.abs(diff_age_similarties))
plt.axhline(y=0, c='orange')
plt.show()




L = list(np.arange(32,45))

import matplotlib.pyplot as plt

ax = plt.subplot()

# Define tick label
# Display graph

# for i in range(len(im_similarities_list)):
for i in range(3):

    plt.plot(L,im_similarities_list[i])
plt.show()

from collections import Counter

C = Counter(np.round(diff_age_similarties))

############################# quantify but with SSIM #################

from quant_utilities import *

simf = final_ssim

  

birth_ages_list = []
second_scan_age_list = []


im_similarities_list = []


count = 0

for idxs in all_shared:


    
    subject = idxs[2]
    
    idx1 = idxs[0]
    
    
    
    data = everything.__getitem__(idx1)
    im1 = data['x'][:,mode]
    im1_np = np.array(im1)
    
    im1 = im1.to(device) 
       
            
    idx2 = idxs[1]
    
    data2 = everything.__getitem__(idx2)
    im2 = data2['x'][:,mode]
    
        
    
    
    
    
    im2_np = np.array(im2)
    im2 = im2.to(device)
                                            
    # im1 =  data['x'][:,mode].to(device) 
        
    bs = data.x.shape[0]//40962
    name = everything_arr[idx1,0].split('_')[0]
    
    true_age_1 = data['metadata'].to(device)
    
    # true_age_1_v = torch_age_to_cardinal(true_age_1).to(device)
    true_age_1_v = true_age_1.to(device)

    true_age_2_v = data2['metadata'].to(device)
    birth_age = data2['y']
    birth_ages_list.append(birth_age.item())
    second_scan_age_list.append(true_age_2_v.item())
    temp_similarity = []
    for num in np.arange(32,45):
        
        
        
        new_age = torch.Tensor([num]).to(device)
        if len(new_age.shape)==1 :
            new_age.unsqueeze(0)
        # new_age_v = torch_age_to_cardinal(new_age).to(device)
        new_age_v = new_age.to(device)
        
        difference = new_age_v - true_age_1_v
        
        imgen = model(im1, difference)
        
        im1_np = im1.reshape(bs,-1,len(mode)).detach().cpu().numpy()
        
        imgen = imgen.reshape(bs,-1,len(mode)).detach().cpu().numpy()
         
        
        temp_similarity.append( simf(np.array(imgen[0]).T, im2_np.T))
    
        
    im_similarities_list.append(temp_similarity)
        


max_im_similarities = [np.argmax(i) for i in im_similarities_list]


A = np.arange(32,45)

import pandas as pd

cognitive_ds = pd.read_csv('/home/fa19/Downloads/DHCPNDH1_DATA_2020-11-23_0903 (2).csv')
cognitive_ds = cognitive_ds[['participationid','cog_comp']]

cog_outcomes = []
for idxs in all_shared:
    name = idxs[2]
    found = cognitive_ds.loc[cognitive_ds['participationid']==name]['cog_comp']
    if len(found)!=0:
        F = found.item()
        if np.isnan(F)==False:
            
            cog_outcomes.append(found.item())
        else:
            cog_outcomes.append(-1)
    else:
        cog_outcomes.append(-1)


qchat_ds = np.load('/home/fa19/Documents/Benchmarking/data/qchat/full.npy',allow_pickle=True)
for row in qchat_ds:
    row[0] = row[0].split('_')[0]
qchat_dict = {qchat_ds[i,0]:qchat_ds[i,-1] for i in range(len(qchat_ds))}

qchat_outcomes = []
for idxs in all_shared:
    name = idxs[2]
    qchat_outcomes.append(qchat_dict.get(name,-1))
qchat_outcomes = np.array(qchat_outcomes)


max_age_similarities = np.array(max_im_similarities) + 32
diff_age_similarties = second_scan_age_list - max_age_similarities
true_age_indices =np.round(np.array(second_scan_age_list))-32

true_age_similarities = []
for i in range(len(im_similarities_list)):
    true_age_similarities.append(im_similarities_list[i][int(true_age_indices[i])])
              
filtered_indices = np.array(cog_outcomes)>10

# filtered_indices = np.array(qchat_outcomes)>11

import matplotlib.pyplot as plt
fig = plt.figure()
# plt.scatter(np.array(second_scan_age_list)-np.array(birth_ages_list), diff_age_similarties)
# plt.scatter(np.array(birth_ages_list), diff_age_similarties)
# plt.scatter(np.array(birth_ages_list), np.abs(diff_age_similarties))
# plt.scatter(np.array(second_scan_age_list)-np.array(birth_ages_list), np.abs(diff_age_similarties))
plt.scatter(np.array(birth_ages_list), np.array(true_age_similarities),s=20, marker='x')
# plt.plot(np.array(birth_ages_list), A*np.array(birth_ages_list)+B, c='orange')
# plt.scatter(np.array(cog_outcomes)[filtered_indices], np.array(diff_age_similarties)[filtered_indices],s=20, marker='x')
# plt.scatter(np.array(qchat_outcomes)[filtered_indices], np.array(true_age_similarities)[filtered_indices],s=20, marker='x')

plt.ylabel('Real Scan Age minus Apparent Age')
plt.xlabel('BayLeys Score')

plt.show()


L = list(np.arange(32,45))

import matplotlib.pyplot as plt
fig = plt.figure()
ax = plt.subplot()

# Define tick label
# Display graph

# for i in range(len(im_similarities_list)):

for i in range(3):

    plt.plot(L,im_similarities_list[i])
plt.show()

#################### incremental generation #########



birth_ages_list = []
second_scan_age_list = []


im_similarities_list = []


count = 0

for idxs in all_shared:


    
    subject = idxs[2]
    
    idx1 = idxs[0]
    
    
    
    data = everything.__getitem__(idx1)
    im1 = data['x'][:,mode]
    im1_np = np.array(im1)
    
    im1 = im1.to(device) 
       
            
    idx2 = idxs[1]
    
    data2 = everything.__getitem__(idx2)
    im2 = data2['x'][:,mode]
    
        
    
    
    
    
    im2_np = np.array(im2)
    im2 = im2.to(device)
                                            
    # im1 =  data['x'][:,mode].to(device) 
        
    bs = data.x.shape[0]//40962
    name = everything_arr[idx1,0].split('_')[0]
    
    true_age_1 = data['metadata'].to(device)
    
    # true_age_1_v = torch_age_to_cardinal(true_age_1).to(device)
    true_age_1_v = true_age_1.to(device)

    true_age_2_v = data2['metadata'].to(device)
    birth_age = data2['y']
    birth_ages_list.append(birth_age.item())
    second_scan_age_list.append(true_age_2_v.item())
    temp_similarity = []
    for num in np.arange(32,45):
        
        
        
        new_age = torch.Tensor([num]).to(device)
        if len(new_age.shape)==1 :
            new_age.unsqueeze(0)
        # new_age_v = torch_age_to_cardinal(new_age).to(device)
        new_age_v = new_age.to(device)
        
        difference = new_age_v - true_age_1_v
        
        imgen = model(im1, difference)
        
        im1_np = im1.reshape(bs,-1,len(mode)).detach().cpu().numpy()
        im1 = imgen.squeeze(0)
        true_age_1_v = new_age_v
        imgen = imgen.reshape(bs,-1,len(mode)).detach().cpu().numpy()
         
        
        temp_similarity.append( PSNR(np.array(imgen), im2_np))
        print('done')
    im_similarities_list.append(temp_similarity)
        


max_im_similarities = [np.argmax(i) for i in im_similarities_list]


A = np.arange(32,45)



max_age_similarities = np.array(max_im_similarities) + 32
diff_age_similarties = second_scan_age_list - max_age_similarities

import matplotlib.pyplot as plt
fig = plt.figure()
# plt.scatter(np.array(second_scan_age_list)-np.array(birth_ages_list), diff_age_similarties)
# plt.scatter(np.array(birth_ages_list), diff_age_similarties)
# plt.scatter(np.array(birth_ages_list), np.abs(diff_age_similarties))
plt.scatter(np.array(second_scan_age_list)-np.array(birth_ages_list), np.abs(diff_age_similarties))
plt.axhline(y=0, c='orange')
plt.show()




L = list(np.arange(32,45))

import matplotlib.pyplot as plt

ax = plt.subplot()

# Define tick label
# Display graph

# for i in range(len(im_similarities_list)):
for i in range(3):

    plt.plot(L,im_similarities_list[i])
    plt.axvline(second_scan_age_list[i])
plt.show()

from collections import Counter

C = Counter(np.round(diff_age_similarties))