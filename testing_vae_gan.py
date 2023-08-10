#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 03:13:33 2022

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
full_dataset_arr = np.load('data/' + 'full' + '/full.npy', allow_pickle = True)

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


batch_size = 1


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

from models import monet_variational_upconv_batched, Discriminator, monet_variational_upsample_batched, Monet_Decoder, Monet_Encoder


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
    
features = [64,128,256,512,1024]

latent_dim = 128



# resdir = 'results/'+str(model_name)+'/'+str(style)+'/'

encoder = Monet_Encoder(num_features = features, in_channels= in_channels, latent_dim = latent_dim).to(device)
decoder = Monet_Decoder(num_features = features, in_channels= in_channels, latent_dim = latent_dim).to(device)


# discriminator_model = Discriminator(num_features = [16,32,64,128], in_channels= in_channels).to(device)


encoder.load_state_dict(torch.load('vae-gan-results/saved_models/encoder_vae_gan_statedict'))
decoder.load_state_dict(torch.load('vae-gan-results/saved_models/decoder_vae_gan_statedict'))




print('Loaded Successfully')

encodings = []

for i,data in enumerate(full_loader):
    data.x  = data.x[:,mode].to(device)
    data.batch = data.batch.to(device)
    data.edge_index = data.edge_index.to(device)
        
        
    Z, mean, var, bs = encoder(data)



    encodings.append(Z.detach().cpu().numpy().flatten())
    
    
    
    
    
    # save_as_metric(reconstruction.detach().cpu().numpy(), 'test_recon_testds_vae')
    # save_as_metric(data.x.detach().cpu().numpy(), 'original_test_vae') 
#
#
#    
encodings = np.row_stack(encodings)

from sklearn.manifold import TSNE
latent_dim = len(Z)
if latent_dim >2:
    
    embedding = TSNE(n_components=2).fit_transform(encodings)
else:
    embedding = encodings

test_ages = full_dataset_arr[:,1].astype(float)
rounded_test_ages = np.round(test_ages)

if test_parity == 'both':
    rounded_test_ages = np.hstack([rounded_test_ages, rounded_test_ages])

import matplotlib.pyplot as plt
fig, ax = plt.subplots()

scatter = ax.scatter(embedding[:,0], embedding[:,1], c=rounded_test_ages)

# produce a legend with the unique colors from the scatter
legend1 = ax.legend(*scatter.legend_elements(),
                    loc="lower left", title="Scan Ages")
ax.add_artist(legend1)

plt.show()




L_encodings = encodings[:encodings.shape[0]//2]


L_encodings_old = L_encodings[full_dataset_arr[:,1]>=41]
centroid_old_L = np.mean(L_encodings_old, axis=0)


L_encodings_young = L_encodings[full_dataset_arr[:,1]<=45]
centroid_young_L = np.mean(L_encodings_young, axis=0)


L_encodings_CC00305XX08 = L_encodings[436]

diff = centroid_old_L - centroid_young_L

for i in range(30):
    new_point = L_encodings_CC00305XX08 + diff*(i/10)
    
    new_image = decoder(torch.Tensor(new_point).cuda(),1)
    new_image[:,1] -= 0.5
    save_as_metric(new_image.detach().cpu(), 'CC00305XX08_diff_move_'+str(i))
    



L_encodings_CC00305XX08_young = L_encodings[436]
L_encodings_CC00305XX08_old = encodings[119]
diff = L_encodings_CC00305XX08_old - L_encodings_CC00305XX08_young
for i in range(10):
    new_point = L_encodings_CC00305XX08 + diff*(i/10)
    
    new_image = decoder(torch.Tensor(new_point).cuda(),1)
    new_image[:,1] -= 0.5
    save_as_metric(new_image.detach().cpu(), 'CC00305XX08_interp_y2o_'+str(i))
    





