#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 21:05:45 2022

@author: fa19
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 19:44:29 2022

@author: fa19
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 14:21:06 2022

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

from data_utils.MyDataLoader import My_dHCP_Data_Graph, My_dHCP_Data
from data_utils.utils import load_dataloader_graph,load_dataloader_graph_classification, load_dataset_graph,load_dataset_arrays, load_model, make_fig

from torch_geometric.data import DataLoader

import copy
import nibabel as nb

from my_utils import save_as_metric
dataset = 'scan_age'

train_dataset_arr = np.load('data/' + str(dataset) + '/train.npy', allow_pickle = True)
val_dataset_arr = np.load('data/' + str(dataset) + '/validation.npy', allow_pickle = True)
test_dataset_arr = np.load('data/' + str(dataset) + '/test.npy', allow_pickle = True)
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


val_ds_unwarped = My_dHCP_Data_Graph(input_arr = val_dataset_arr, 
                   warped_files_directory ='/home/fa19/Documents/dHCP_Data_merged/Warped',
                   unwarped_files_directory = '/home/fa19/Documents/dHCP_Data_merged/merged',
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

full_ds = My_dHCP_Data_Graph(input_arr = full_dataset_arr, 
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


batch_size = 8


train_loader = DataLoader(train_ds, batch_size=batch_size, 
                                           shuffle=True, 
                                           num_workers = 1)


val_loader = DataLoader(val_ds, batch_size=1, 
                                           shuffle=False, 
                                           num_workers = 2)


val_unwarped_loader = DataLoader(val_ds_unwarped, batch_size=1, 
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


from models import GraphNet_VAE_simple, GraphNet_discriminator_simple


modality = 'both'

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
    
    

latent_dim = 2048

# from scipy.interpolate import griddata


# xy_points = np.load('data/suggested_ico_6_points.npy')
# xy_points[:,0] = (xy_points[:,0] + 0.1)%1
# grid = np.load('data/grid_170_square.npy')


# grid_x, grid_y = np.meshgrid(np.linspace(0, 1, 170), np.linspace(0.00, 1, 170))
# grid[:,0] = grid_x.flatten()
# grid[:,1] = grid_y.flatten()

# resdir = 'results/'+str(model_name)+'/'+str(style)+'/'


model = GraphNet_VAE_simple(len(mode),latent_dim, device=device)

model = model.to(device)


discriminator = GraphNet_discriminator_simple(len(mode),device=device).to(device)


for layer in model.children():
    if hasattr(layer, 'reset_parameters'):
        layer.reset_parameters() 


learning_rate = 1e-5
learning_rate_D = 1e-5
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay= 1e-5)

optimizer_D = torch.optim.Adam(discriminator.parameters(), lr = learning_rate_D, weight_decay= 1e-5)

def criterion(x, y, mean, log_var):

    bceloss = nn.BCELoss(reduction='sum')(x,y)
    l1loss = nn.L1Loss(reduction = 'sum')(x,y)
    KLD   = -0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())
#    print(KLD)
    return bceloss + l1loss + KLD / 1000

from torch.autograd import Variable
import torch_geometric 

adversarial_loss = nn.BCELoss()
best = 100
num_epochs = 100
print('Starting...')

for epoch in range(num_epochs):
    train_loss = []
    model.train()
    d_losses = []
    for i, data in enumerate(train_loader):
        
        data.x =  data['x'][:,mode].to(device)
        data.edge_index = data.edge_index.to(device)
        data.batch = data.batch.to(device)
        
        optimizer.zero_grad()

        bs = data.x.shape[0]//40962
        
        valid = Variable(torch.Tensor(bs, 1).fill_(1.0), requires_grad=False).to(device)
        fake = Variable(torch.Tensor(bs, 1).fill_(0.0), requires_grad=False).to(device)

        loss,Output = model(data)    
        gen_data = torch_geometric.data.Data(x = Output[0]+torch.randn_like(Output[0])/10, edge_index=data.edge_index, batch=data.batch)
        # print(Output[0].shape)
        # print(discriminator(gen_data,bs))
        g_loss = adversarial_loss(discriminator(gen_data,bs),valid)
        total_loss = loss + g_loss
        total_loss.backward()
        
        optimizer.step()
        if i% 3 == 0:
            optimizer_D.zero_grad()
            gen_data.x = gen_data.x.detach()
            
            d_loss_fake = adversarial_loss(discriminator(gen_data, bs), fake)
    
            d_loss_real = adversarial_loss(discriminator(data, bs), valid)
            
            d_loss_total = (d_loss_fake + d_loss_real ) /2
            d_loss_total.backward()
            
            optimizer_D.step()
            
            d_losses.append(d_loss_total.item())            
        train_loss.append(total_loss.item())

    print('Epoch ' + str(epoch) + ' ', np.mean(train_loss), 'D Loss', np.mean(d_losses))
    
    model.eval()
    val_loss = []
    
    for i,data in enumerate(val_unwarped_loader):
     
        data.x =  data['x'][:,mode].to(device)
        data.edge_index = data.edge_index.to(device)
        data.batch = data.batch.to(device)

        loss,O= model(data)        
        
     
        val_loss.append(loss.item())
    
        if i % 5 == 0:        
            gen_imgs = O[0].detach().cpu().numpy()
           
            save_as_metric(gen_imgs, 'test_recon_val_vae'+str(i))
            save_as_metric(data.x.detach().cpu().numpy(), 'original_recon_val_vae'+str(i))
            
    print('Validation is ', np.mean(val_loss))


    # if np.mean(val_loss)<= best:
    # #     torch.save(model.state_dict())
    # print(np.mean(val_loss))


    # gen_imgs_np = gen_imgs.detach().cpu().numpy()
    # full_B = np.zeros([40962,len(mode)])
    # for m in range(len(mode)):

    #    B = griddata(grid, gen_imgs_np[0,m,:,:].reshape(28900), xy_points, 'linear')
    #    full_B[:,m] = B
    
    # save_as_metric(full_B, 'results/val_rounded_'+(str(i)))
    
    

model.eval()    
encodings = []
test_loss = []
for i,data in enumerate(full_loader):
    data.x  = data.x[:,mode].unsqueeze(1).to(device)
    data.batch = data.batch.to(device)
    data.edge_index = data.edge_index.to(device)
    loss, Output = model(data)
    
    recon, z, mean,logvar = Output


    encodings.append(z.detach().cpu().numpy())


    
encodings = np.row_stack(encodings)

from sklearn.manifold import TSNE
if latent_dim >2:
    
    embedding = TSNE(n_components=2).fit_transform(encodings)
else:
    embedding = encodings
full_ages = full_dataset_arr[:,-1].astype(float)
rounded_full_ages = np.round(full_ages)

if test_parity == 'both':
    rounded_full_ages = np.hstack([rounded_full_ages, rounded_full_ages])
    
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
if latent_dim !=1:
    scatter = ax.scatter(embedding[:,0], embedding[:,1], c=rounded_full_ages)
else:
    scatter = ax.scatter(embedding, rounded_full_ages)
    

# produce a legend with the unique colors from the scatter
legend1 = ax.legend(*scatter.legend_elements(),
                    loc="lower left", title="Birth Ages")
ax.add_artist(legend1)

plt.show()



full_ages = full_dataset_arr[:,-2].astype(float)
rounded_full_ages = np.round(full_ages)

if test_parity == 'both':
    rounded_full_ages = np.hstack([rounded_full_ages, rounded_full_ages])
    
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
if latent_dim !=1:
    scatter = ax.scatter(embedding[:,0], embedding[:,1], c=rounded_full_ages)
else:
    scatter = ax.scatter(embedding, rounded_full_ages)
    

# produce a legend with the unique colors from the scatter
legend1 = ax.legend(*scatter.legend_elements(),
                    loc="lower left", title="Scan Ages")
ax.add_artist(legend1)

plt.show()



pointA = encodings[212] #CC00135AN12_44900 sa 35.14 ba 34.14


pointB = encodings[136] #CC00135AN12_44900 sa 35.14 ba 34.14


diff = pointB-pointA


model.decode()










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
