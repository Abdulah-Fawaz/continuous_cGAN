#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 15:05:24 2022

@author: fa19
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 14:57:15 2022

@author: fa19
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 12:28:29 2021

@author: fa19
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 16:17:25 2021

@author: fa19
"""
from scipy.interpolate import griddata

import os

import sys
import numpy as np
from os.path import abspath, dirname
import torch
import torch.nn as nn

from utils import validate_graph, train_graph, pick_criterion, load_optimiser, import_from, load_testing_graph, test_graph

from data_utils.MyDataLoader import My_dHCP_Data_Graph
from data_utils.utils import load_dataloader_graph,load_dataloader_graph_classification, load_dataset_graph,load_dataset_arrays, load_model, make_fig

from torch_geometric.data import DataLoader

import copy
import nibabel as nb

from my_utils import save_as_metric
dataset = 'scan_age'

train_dataset_arr = np.load('data/' + str(dataset) + '/train.npy', allow_pickle = True)
val_dataset_arr = np.load('data/' + str(dataset) + '/validation.npy', allow_pickle = True)
test_dataset_arr = np.load('data/' + str(dataset) + '/test.npy', allow_pickle = True)
full_dataset_arr = np.load('data/' + str(dataset) + '/full.npy', allow_pickle = True)

train_rots = False 
num_warps = 100
registered = True

train_parity = 'both'
norm_style = 'range'
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
                   edges = edges,
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
                   edges = edges,
                   rotations= False,
                   number_of_warps = 0,
                   parity_choice = test_parity,
                   smoothing = False,
                   normalisation = norm_style,
                   projected = False,
                   sample_only = True, #if false, will go through every warp, not just one random warp
                   output_as_torch = True,
                   )


test_ds = My_dHCP_Data_Graph(input_arr = test_dataset_arr, 
                   warped_files_directory = warped_directory,
                   unwarped_files_directory = unwarped_directory,
                   edges = edges,
                   rotations= False,
                   number_of_warps = 0,
                   parity_choice = test_parity,
                   smoothing = False,
                   normalisation = norm_style,
                   projected = False,
                   sample_only = True, #if false, will go through every warp, not just one random warp
                   output_as_torch = True,
                   )

full_ds = My_dHCP_Data_Graph(input_arr = full_dataset_arr, 
                   warped_files_directory = warped_directory,
                   unwarped_files_directory = unwarped_directory,
                   edges = edges,
                   rotations= False,
                   number_of_warps = 0,
                   parity_choice = test_parity,
                   smoothing = False,
                   normalisation = norm_style,
                   projected = False,
                   sample_only = True, #if false, will go through every warp, not just one random warp
                   output_as_torch = True,
                   )


batch_size = 2


train_loader = DataLoader(train_ds, batch_size=batch_size, 
                                           shuffle=True, 
                                           num_workers = 1)


val_loader = DataLoader(val_ds, batch_size=1, 
                                           shuffle=False, 
                                           num_workers = 2)


test_loader = DataLoader(test_ds, batch_size=1, 
                                           shuffle=False, 
                                           num_workers = 2)


full_loader = DataLoader(full_ds, batch_size=1, 
                                           shuffle=False, 
                                           num_workers = 2)




device_number = 0

device = torch.device('cuda:' + str(device_number) if torch.cuda.is_available() else 'cpu')
print('device is ', device)
torch.cuda.set_device(device)

from networks import VAE_gmmconv, VAE_mixed


modality = 'all'

if modality == 'myelination':
    mode = 0
    in_channels = 1

elif modality == 'curvature':
    mode = 1
    in_channels = 1

elif modality == 'both':
    mode = [0,1]
    in_channels = 2

elif modality == 'all':
    mode = [0,1,2,3]
    in_channels = 4
    
    
features = [32,64,128,256,512,1024]

latent_dim = 128



# resdir = 'results/'+str(model_name)+'/'+str(style)+'/'



model = VAE_gmmconv(num_features = features, in_channels= in_channels, latent_dim = latent_dim, device = device)
model = model.to(device)
for layer in model.children():
    if hasattr(layer, 'reset_parameters'):
        layer.reset_parameters() 


learning_rate = 1e-3

optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay= 1e-5)


def criterion(x, y, mean, log_var):

    bceloss = nn.BCELoss(reduction='sum')(x,y)
    l1loss = nn.L1Loss(reduction = 'sum')(x,y)
    KLD   = -0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())
#    print(KLD)
    return bceloss + l1loss + KLD / 1000

    

num_epochs = 100
print('Starting...')
for epoch in range(num_epochs):
    train_loss = []

    for i, data in enumerate(train_loader):
        
        model.train()
        data.x = data.x[:,mode].to(device)
        data.batch = data.batch.to(device)
        data.edge_index = data.edge_index.to(device)
        
        mean, var, reconstruction = model(data)

        loss = criterion(reconstruction, data.x, mean, var)
        
        optimizer.zero_grad()
        train_loss.append(loss.item())
        loss.backward()
        
        optimizer.step()
        # optimizer_D.zero_grad()

        # pred_real = discriminator_model(data)
        # data.x = reconstruction

        # pred_false = discriminator_model(data)

        # lossD_1 = discriminator_loss(pred_real.squeeze(1), torch.Tensor([0.9]).to(device))

        # lossD_2 = discriminator_loss(pred_false.squeeze(1), torch.Tensor([0.1]).to(device))
        
        # lossD = (lossD_1 + lossD_2)        
        # discriminator_losses.append(lossD.item())

        # if lossD > 0.6:
            
        #     lossD.backward()
            
        #     optimizer_D.step()
        torch.cuda.empty_cache()
        
    print('Epoch ' + str(epoch) + ' ', np.mean(train_loss))
    
    model.eval()
    val_loss = []
    
    for i,data in enumerate(val_loader):
        data.x  = data.x[:,mode].to(device)
        data.batch = data.batch.to(device)
        data.edge_index = data.edge_index.to(device)
        mean, var, reconstruction = model(data)

        loss = criterion(reconstruction, data.x, mean, var)
        #        if i ==0:
#            print(torch.max(model.encode(data)[0]))
        
        val_loss.append(loss.item())
        
        if i % 3 == 0:
            save_as_metric(reconstruction.detach().cpu().numpy(), 'test_recon_val_vae'+str(i))
            save_as_metric(data.x.detach().cpu().numpy(), 'original_val' + str(i))


    print(np.mean(val_loss))



#encodings = []
#test_loss = []
#for i,data in enumerate(full_loader):
#    data.x  = data.x[:,mode].unsqueeze(1).to(device)
#    data.batch = data.batch.to(device)
#    data.edge_index = data.edge_index.to(device)
#    reconstruction, mean, var = model(data)
#
#    loss = criterion(reconstruction, data.x, mean, var)
#    
#    test_loss.append(loss.item())
#    encoding = model.encode(data)[0].detach().cpu().numpy()
#    encodings.append(encoding)
#    save_as_metric(reconstruction.detach().cpu().numpy(), 'test_recon_testds_vae')
#    save_as_metric(data.x.detach().cpu().numpy(), 'original_test_vae') 
#
#
#    
#encodings = np.row_stack(encodings)
#
#from sklearn.manifold import TSNE
#if latent_dim >2:
#    
#    embedding = TSNE(n_components=2).fit_transform(encodings)
#else:
#    embedding = encodings
#
#test_ages = test_dataset_arr[:,1].astype(float)
#rounded_test_ages = np.round(test_ages)
#
#if test_parity == 'both':
#    rounded_test_ages = np.hstack([rounded_test_ages, rounded_test_ages])
#
#import matplotlib.pyplot as plt
#fig, ax = plt.subplots()
#
#scatter = ax.scatter(embedding[:,0], embedding[:,1], c=rounded_test_ages)
#
## produce a legend with the unique colors from the scatter
#legend1 = ax.legend(*scatter.legend_elements(),
#                    loc="lower left", title="Scan Ages")
#ax.add_artist(legend1)
#
#plt.show()




#for i in range(50):
#    GEN = model.decode(torch.randn([latent_dim]).cuda())
##    print(GEN)
#    save_as_metric(GEN.detach().cpu().numpy(), 'test_gen_vae'+str(i))
#    
    


# encodings = []
# test_loss = []
# for i,data in enumerate(full_loader):
#     data.x  = data.x[:,mode].unsqueeze(1).to(device)
#     data.batch = data.batch.to(device)
#     data.edge_index = data.edge_index.to(device)
#     reconstruction, mean, var = model(data)

#     loss = criterion(reconstruction, data.x, mean, var)
    
#     test_loss.append(loss.item())
#     encoding = model.encode(data)[0].detach().cpu().numpy()
#     encodings.append(encoding)

# encodings = np.row_stack(encodings)


# from sklearn.manifold import TSNE
# if latent_dim >2:
    
#     embedding = TSNE(n_components=2).fit_transform(encodings)
    
# else:
#     embedding = encodings

# full_ages = full_dataset_arr[:,-1].astype(float)
# rounded_full_ages = np.round(full_ages)

# if test_parity == 'both':
#     rounded_full_ages = np.hstack([rounded_full_ages, rounded_full_ages])
    
# import matplotlib.pyplot as plt
# fig, ax = plt.subplots()
# if latent_dim !=1:
#     scatter = ax.scatter(embedding[:,0], embedding[:,1], c=rounded_full_ages)
# else:
#     scatter = ax.scatter(embedding, rounded_full_ages)
    

# # produce a legend with the unique colors from the scatter
# legend1 = ax.legend(*scatter.legend_elements(),
#                     loc="lower left", title="Scan Ages")
# ax.add_artist(legend1)

# plt.show()


# C = np.zeros(len(rounded_full_ages))
# half = int(np.round(len(C)*0.5))

# C[half:] = 1
# import matplotlib.pyplot as plt
# fig, ax = plt.subplots()

# scatter = ax.scatter(embedding[:,0], embedding[:,1], c=C)

# # produce a legend with the unique colors from the scatter
# legend1 = ax.legend(*scatter.legend_elements(),
#                     loc="lower left", title="Scan Ages")
# ax.add_artist(legend1)

# plt.show()


# sum_prems = np.sum(encodings[rounded_full_ages<=33],axis=0)
# centroid_prems = sum_prems / np.sum(rounded_full_ages<=33)

# sum_terms = np.sum(encodings[rounded_full_ages>=40],axis=0)
# centroid_terms = sum_terms / np.sum(rounded_full_ages>=40)


# diff_vector = centroid_terms - centroid_prems


# for i, data in enumerate(test_loader):
#     if i == 53:
#         data.x  = data.x[:,mode].unsqueeze(1).to(device)
#         data.batch = data.batch.to(device)
#         data.edge_index = data.edge_index.to(device)
#         reconstruction, mean, var = model(data)
        
#         loss = criterion(reconstruction, data.x, mean, var)
        
#         test_loss.append(loss.item())
#         encoding = model.encode(data)[0].detach().cpu().numpy()


# save_dir = '/home/fa19/Documents/Surface-VGAE/results/monet_upconv/' + str(dataset)+'/registered/latent_dim_' + str(latent_dim)+'/' + str(modality)+'/'

# for i in range(15):
     
#     new_encoding = encoding + (i/15 * diff_vector)
    
#     reconstruction = model.decode(torch.Tensor(new_encoding).to(device))
#     save_as_metric(reconstruction.detach().cpu().numpy(), save_dir + 'interpolated_vae_vyoung_'+str(i))
    

# for i, data in enumerate(test_loader):
#     if i == 0:
#         data.x  = data.x[:,mode].unsqueeze(1).to(device)
#         data.batch = data.batch.to(device)
#         data.edge_index = data.edge_index.to(device)
#         reconstruction, mean, var = model(data)
        
#         loss = criterion(reconstruction, data.x, mean, var)
        
#         test_loss.append(loss.item())
#         encoding = model.encode(data)[0].detach().cpu().numpy()


# save_dir = '/home/fa19/Documents/Surface-VGAE/results/monet_upconv/' + str(dataset)+'/registered/latent_dim_' + str(latent_dim)+'/' + str(modality)+'/'

# for i in range(15):
    
#     new_encoding = encoding - (i/15 * diff_vector)
    
#     reconstruction = model.decode(torch.Tensor(new_encoding).to(device))
#     save_as_metric(reconstruction.detach().cpu().numpy(), save_dir + 'interpolated_vae_vold_'+str(i))
    
# model_save_dir = '/home/fa19/Documents/Surface-VGAE/results/monet_upconv/' + str(dataset)+'/registered/latent_dim_' + str(latent_dim)+'/' + str(modality)+'/'
# torch.save(model, model_save_dir+'model.pth')
