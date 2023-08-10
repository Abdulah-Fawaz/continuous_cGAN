#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 23:26:15 2022

@author: fa19
"""
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
dataset = 'birth_age_confounded'

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
                   smoothing = False,
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
                    smoothing = False,
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


batch_size = 8

sa_ba_age_differences = train_dataset_arr[:,1] - train_dataset_arr[:,2]

weights = np.clip(sa_ba_age_differences, 0, 1)+0.01

weights = weights.astype(np.float32)


# frac_prems = sum(train_dataset_arr[:,-1]<37)/len(train_dataset_arr)

# weight_prems = 5

# positions = [i for i in range(len(train_dataset_arr)) if np.logical_and(train_dataset_arr[i,-1]<37, train_dataset_arr[i,1]>=37)]
# # positions2 = [i for i in range(len(train_dataset_arr), 2*len(train_dataset_arr)) if np.logical_and(train_dataset_arr[i,-1]<37, train_dataset_arr[i,1]>=37)]

# positions2 =np.array( positions) + len(train_dataset_arr)

# positions = positions + list(positions2)

# weights[positions] = weight_prems
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


from markovian_models import GraphNet_Regressor_confounded
from regressor_model_monet import monet_polar_regression_confounded, graphconv_regression_confounded,monet_polar_regression_confounded_batched
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

    


# model = GraphNet_Regressor_confounded(in_channels,device='cuda')
model = monet_polar_regression_confounded_batched([64,128,256,512], in_channels = in_channels,device='cuda')

# num_features, conv_style=gmmconv,activation_function=nn.ReLU(), in_channels = 4, device='cuda'
# model = torch.load('/home/fa19/Documents/neurips/regressor_monet_ba_std2')

model = model.to(device)



learning_rate = 1e-4

optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay= 1e-5)

best = 100
num_epochs = 4000
print('Starting...')


test_criterion = nn.L1Loss()

criterion = nn.SmoothL1Loss()

val_check_frequency = 10
for epoch in range(num_epochs):
    train_loss = []
    model.train()
    d_losses = []
    g_losses = []
    
    for i, data in enumerate(train_loader):
        model.train()
        im1 =  data['x'][:,mode].to(device) 
        edge = data['edge_index'].to(device)
        scan_age = data['metadata'].to(device)
        bs = data.batch
        data.x = im1

        true_age = data['y'].to(device)
      
        optimizer.zero_grad()
        prediction = model(data.to(device))
        # prediction = model(im1, scan_age).squeeze(1)
   
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
            scan_age = data['metadata'].to(device)
            data.x = im1

            bs = data.batch


            true_age = data['y'].to(device)
          
            prediction = model(data.to(device))

            # prediction = model(im1, scan_age).squeeze(1)
       
            val_loss = test_criterion(prediction, true_age)

            
          
            val_losses.append(loss.item())
        
        
        print('Validation ', np.mean(val_losses))
        
# model = torch.load('/home/fa19/Documents/neurips/regressor_monet_ba_std2')


test_dataset_arr = np.load('data/' + 'ba'+ '/test.npy', allow_pickle = True)
norm_style= 'std'

test_ds = My_dHCP_Data_Graph(input_arr = test_dataset_arr, 
                    warped_files_directory = warped_directory,
                    unwarped_files_directory = unwarped_directory,
                    edges=edges,
                    rotations= False,
                    number_of_warps = 0,
                    parity_choice = 'both',
                    smoothing = False,
                    normalisation = norm_style,
                    projected = False,
                    sample_only = True, #if false, will go through every warp, not just one random warp
                    output_as_torch = True,
                    )



test_loader = DataLoader(test_ds, batch_size=1, 
                                            shuffle=False, 
                                            num_workers = 2)



test_losses = []
test_trues = []
test_preds = []
test_sas = []
for i, data in enumerate(test_loader):
    model.eval()
    
    
    im1 =  data['x'][:,mode].to(device) 
    data.x = im1

    bs = data.batch


    true_age = data['y'].to(device)
    
    scan_age = data['metadata'].to(device)

    # prediction = model(im1, scan_age ).squeeze(1)
    prediction = model(data.to(device))

    test_loss = test_criterion(prediction, true_age)

    
    test_preds.append(prediction.item())
    test_trues.append(true_age.item())
    test_losses.append(test_loss.item())
    test_sas.append(data.metadata.item())

print('Test ', np.mean(test_losses))


import matplotlib.pyplot as plt
fig = plt.figure()
plt.scatter(np.array(test_preds), np.array(test_trues), c= test_sas)

plt.show()

fig = plt.figure()
plt.scatter( np.array(test_sas)-np.array(test_trues),np.abs(np.array(test_trues) - np.array(test_preds)))

plt.show()


# test_losses = []
# for i, data in enumerate(val_loader):
#     model.eval()
    
    
#     im1 =  data['x'][:,mode].to(device) 
    
#     bs = data.batch


#     true_age = data['y'].to(device)
    

#     prediction = model(data.to(device) )
   
#     test_loss = test_criterion(prediction, true_age)

    
  
#     test_losses.append(test_loss.item())


# print('Test ', np.mean(test_losses))



torch.save(model,'/home/fa19/Documents/neurips/regressor_monet_ba_std3')

torch.save(model.state_dict(),'/home/fa19/Documents/neurips/regressor_monet_ba_std_sd3')


# np.save(np.vstack([np.array(test_trues), np.array(test_preds)]),'/home/fa19/Documents/neurips/ba_regressor_predictions1')




