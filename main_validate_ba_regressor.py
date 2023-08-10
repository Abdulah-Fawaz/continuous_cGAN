#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 10 18:26:38 2022

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
dataset = 'ba'

train_dataset_arr = np.load('data/' + str(dataset) + '/train.npy', allow_pickle = True)
val_dataset_arr = np.load('data/' + str(dataset) + '/validation.npy', allow_pickle = True)
test_dataset_arr = np.load('data/' + str(dataset) + '/test.npy', allow_pickle = True)



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







everything_arr  = np.load('data/' +'full' + '/full.npy', allow_pickle = True)

everything = My_dHCP_Data_Graph(input_arr = everything_arr, 
                   warped_files_directory = warped_directory,
                   unwarped_files_directory = unwarped_directory,
                   rotations= False,
                   number_of_warps = 0,
                   edges = edges,
                   smoothing = False,
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







device_number = 0

device = torch.device('cuda:' + str(device_number) if torch.cuda.is_available() else 'cpu')
print('device is ', device)
torch.cuda.set_device(device)


from markovian_models import GraphNet_Markovian_Generator, GraphNet_Markovian_Discriminator_Simple_Double, GraphNet_Markovian_Discriminator, GraphNet_Markovian_Discriminator_Simple, GraphNet_Regressor_confounded

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

    



# from scipy.interpolate import griddata


# xy_points = np.load('data/suggested_ico_6_points.npy')
# xy_points[:,0] = (xy_points[:,0] + 0.1)%1
# grid = np.load('data/grid_170_square.npy')


# grid_x, grid_y = np.meshgrid(np.linspace(0, 1, 170), np.linspace(0.00, 1, 170))
# grid[:,0] = grid_x.flatten()
# grid[:,1] = grid_y.flatten()

# resdir = 'results/'+str(model_name)+'/'+str(style)+'/'

### age is 20 to 50


model_loc = '/home/fa19/Documents/neurips/'


scan_age_model = GraphNet_Markovian_Generator(len(mode), age_dim = 1,device=device)
scan_age_model = scan_age_model.to(device)
scan_age_model = torch.load(model_loc + '3cycle_generator_model_final')

birth_age_model = GraphNet_Markovian_Generator(len(mode), age_dim = 2,device=device)
birth_age_model = birth_age_model.to(device)
birth_age_model = torch.load(model_loc + '3cycle_BA_double_generator_model_final3')


sulc_separate = False
if sulc_separate:
        
    birth_age_model1 = GraphNet_Markovian_Generator(1, age_dim = 2,device=device)
    birth_age_model1 = birth_age_model1.to(device)
    birth_age_model1 = torch.load(model_loc + '3cycle_BA_myelin_generator_model_final1')


    
    birth_age_model2 = GraphNet_Markovian_Generator(1, age_dim = 2,device=device)
    birth_age_model2 = birth_age_model2.to(device)
    birth_age_model2 = torch.load(model_loc + '3cycle_BA_sulc_generator_model_final1')


final_save_dir = '/home/fa19/Documents/neurips/full_3cycle_basa/'



# regressor_model = GraphNet_Regressor_confounded(in_channels,device='cuda')
# regressor_model = torch.load('/home/fa19/Documents/neurips/regressor_monet_ba_std1')

from regressor_model_monet import monet_polar_regression_confounded_batched, graphconv_regression_confounded

regressor_model =  monet_polar_regression_confounded_batched([64,128,256,512], in_channels = 2,device='cuda')
regressor_model = torch.load('/home/fa19/Documents/neurips/regressor_monet_ba_std2')
def reg(image, scan_age, D,model=regressor_model, device= 'cuda'):
    D.x = image.to(device)
    D.metadata = scan_age.unsqueeze(1).to(device)
    
    return model(D)
regressor_model = regressor_model.to(device)


import matplotlib.pyplot as plt

fig = plt.figure()



def get_losses(a,b):
    sol = 0
    counter = 0
    a = [item for sublist in a for item in sublist]
    b = [item for sublist in b for item in sublist]

    a = np.array(a)
    b = np.array(b)
    return np.mean(np.abs(a - b)   )



all_losses_final = []
for INDEX in range(len(all_shared)):
    idxs = all_shared[INDEX]    
    print(all_shared[INDEX])
    
    save_loc = '/home/fa19/Documents/neurips/'
    
    
    
    
    subject = idxs[2]
    
    idx2 = idxs[1]
    idx1 = idxs[0]
    
    
    data = everything.__getitem__(idx1)
    im1 = data['x'][:,mode].to(device)  #youngim
        
            
                                            
    # im1 =  data['x'][:,mode].to(device) 
        
    bs = data.x.shape[0]//40962
    name = everything_arr[idx1,0].split('_')[0]
    
    true_age_1 = data['metadata'].to(device) # true scan age 1
    
    # true_age_1_v = torch_age_to_cardinal(true_age_1).to(device)
    true_sa = true_age_1.to(device)
    true_ba =  data['y'].to(device) # true birth age 1
    true_sa_raw = true_sa.item()
    true_ba_raw = true_ba.item()
    
    
    all_losses = []
    all_ba_preds = []
    all_ba_trues = []
    for sa in np.arange(36, 45,3):
        scan_age = torch.Tensor([sa]).to(device)
        
        difference_sa = scan_age - true_sa
        
        scan_age_raw = scan_age.item()
        new_sa_image = scan_age_model(im1, difference_sa)
        
        # save_as_metric(new_sa_image.detach().cpu(), final_save_dir + f'{subject}_sa={scan_age_raw}_ba={true_ba_raw}')
        
        temp_losses = []
        temp_ba_preds = []
        temp_ba_trues = []
        
        for ba in np.arange(min(36, scan_age_raw), scan_age_raw):
            birth_age = torch.Tensor([ba]).to(device)
            birth_age_raw = birth_age.item()
            difference_ba = birth_age - scan_age
            
            D = torch.cat((scan_age.unsqueeze(0), difference_ba.unsqueeze(0)), dim=1)
            
            new_ba_image = birth_age_model(new_sa_image, D)
            if sulc_separate:
                new_ba_image[:,0] = birth_age_model1(new_sa_image[:,0].unsqueeze(1), D).squeeze(1)

                new_ba_image[:,1] = birth_age_model2(new_sa_image[:,1].unsqueeze(1), D).squeeze(1)

            # save_as_metric(new_ba_image.detach().cpu(), final_save_dir + f'{subject}_sa={scan_age_raw}_new_ba={birth_age_raw}')
            data.x = new_ba_image.to(device)
            data.metadata = scan_age.to(device).unsqueeze(1)
            data.batch = torch.zeros(40962).to(device).long()
            # ba_prediction = regressor_model(new_ba_image, scan_age.unsqueeze(1))
            ba_prediction = regressor_model(data.to(device))
            
            temp_ba_preds.append(ba_prediction.item())
            temp_ba_trues.append(birth_age_raw)
        if len(temp_ba_trues)!=0:
            all_ba_preds.append(temp_ba_preds)
            all_ba_trues.append(temp_ba_trues)
        
        
    # print(all_ba_preds)
    
    print(np.round(np.array(all_ba_preds[-1])))
    print(np.array(all_ba_trues[-1]))
    
    all_losses_final.append(get_losses(all_ba_preds, all_ba_trues))
    
    
    m = np.min(all_ba_trues[-1])
    ma = np.max(all_ba_trues[-1])
    
    for k in range(len(all_ba_preds)):
        plt.scatter(np.round(np.array(all_ba_preds[k])), np.array(all_ba_trues[k]))
    plt.xlabel('Predicted Birth Age')
    plt.ylabel('Confounded Birth Age')
    
    plt.plot(np.arange(m,ma), np.arange(m,ma))
plt.show()

print(np.mean(all_losses_final))
print(np.std(all_losses_final))


fig = plt.figure()
for k in range(len(all_ba_preds)):
    plt.scatter(np.round(np.array(all_ba_preds[k])), np.array(all_ba_trues[k]))
plt.xlabel('Predicted Birth Age')
plt.ylabel('Confounded Birth Age')

plt.plot(np.arange(m,ma), np.arange(m,ma))
plt.show()



INDEX = 76


idxs = all_shared[INDEX]    
print(all_shared[INDEX])

save_loc = '/home/fa19/Documents/neurips/'




subject = idxs[2]

idx2 = idxs[1]
idx1 = idxs[0]


data = everything.__getitem__(idx1)
im1 = data['x'][:,mode].to(device)  #youngim


                                
# im1 =  data['x'][:,mode].to(device) 

bs = data.x.shape[0]//40962
name = everything_arr[idx1,0].split('_')[0]

true_age_1 = data['metadata'].to(device) # true scan age 1

# true_age_1_v = torch_age_to_cardinal(true_age_1).to(device)
true_sa = true_age_1.to(device)
true_ba =  data['y'].to(device) # true birth age 1
true_sa_raw = true_sa.item()
true_ba_raw = true_ba.item()


all_losses = []
all_ba_preds = []
all_ba_trues = []
for sa in np.arange(np.round(true_ba_raw), 42,1):
    scan_age = torch.Tensor([sa]).to(device)

    difference_sa = scan_age - true_sa
    
    scan_age_raw = scan_age.item()
    new_sa_image = scan_age_model(im1, difference_sa)
    
    # save_as_metric(new_sa_image.detach().cpu(), final_save_dir + f'{subject}_sa={scan_age_raw}_ba={true_ba_raw}')
    
    temp_losses = []
    temp_ba_preds = []
    temp_ba_trues = []
    
    for ba in np.arange(np.round(true_ba_raw), scan_age_raw+1):
        birth_age = torch.Tensor([ba]).to(device)
        birth_age_raw = birth_age.item()
        difference_ba = birth_age - scan_age
        
        D = torch.cat((scan_age.unsqueeze(0), difference_ba.unsqueeze(0)), dim=1)
        
        new_ba_image = birth_age_model(new_sa_image, D)

        if sulc_separate:
            new_ba_image[:,0] = birth_age_model1(new_sa_image[:,0].unsqueeze(1), D).squeeze(1)

            new_ba_image[:,1] = birth_age_model2(new_sa_image[:,1].unsqueeze(1), D).squeeze(1)

        # save_as_metric(new_ba_image.detach().cpu(), final_save_dir + f'{subject}_sa={scan_age_raw}_new_ba={birth_age_raw}')
        
        # ba_prediction = regressor_model(new_ba_image, scan_age.unsqueeze(1))
        data.x = new_ba_image.to(device)
        data.metadata = scan_age.to(device).unsqueeze(1)
        data.batch = torch.zeros(40962).to(device).long()
        # ba_prediction = regressor_model(new_ba_image, scan_age.unsqueeze(1))
        ba_prediction = regressor_model(data.to(device))
        temp_ba_preds.append(ba_prediction.item())
        temp_ba_trues.append(birth_age_raw)
    if len(temp_ba_trues)!=0:
        all_ba_preds.append(temp_ba_preds)
        all_ba_trues.append(temp_ba_trues)
    

# print(all_ba_preds)

print(np.array(all_ba_preds[-1]))
print(np.array(all_ba_trues[-1]))

all_losses_final.append(get_losses(all_ba_preds, all_ba_trues))
scan_ages_done = [33,36,39,42]

scan_ages_done = list(np.arange(np.round(true_ba_raw),45))
m = np.min(all_ba_trues[-1])
ma = np.max(all_ba_trues[-1])

fig = plt.figure()

for k in range(len(all_ba_preds)):
    plt.scatter(np.array(all_ba_trues[k]), np.array(all_ba_preds[k]), label=scan_ages_done[k], marker='x')
plt.ylabel('Predicted Birth Age')
plt.xlabel('Confounded Birth Age')
plt.legend()

plt.plot(np.arange(34,42), np.arange(34,42))
plt.show()


fig = plt.figure()

for k in range(len(all_ba_preds)):
    plt.plot(np.array(all_ba_trues[k]), np.array(all_ba_preds[k]), label=scan_ages_done[k], marker='x')
plt.ylabel('Predicted Birth Age')
plt.xlabel('Confounded Birth Age')
plt.legend()

plt.plot(np.arange(34,42), np.arange(34,42))
plt.show()

