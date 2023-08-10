#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 19:26:07 2022

@author: fa19
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 18:29:03 2022

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
from torch.autograd import Variable

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


batch_size = 4


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

from models import Monet_GAN,GraphNet_GAN, Discriminator,Discriminator2, monet_variational_upsample_batched, Monet_Decoder, Monet_Encoder


modality = 'curvature'

if modality == 'myelination':
    mode = 0
    in_channels = 1

elif modality == 'curvature':
    mode = [1]
    in_channels = 1

elif modality == 'both':
    mode = [0,1]
    in_channels = 2
    
features = [64,128,256,512,1024]

latent_dim = 512

recon_loss = torch.nn.SmoothL1Loss()
adversarial_loss = torch.nn.BCELoss()

# resdir = 'results/'+str(model_name)+'/'+str(style)+'/'

generator = GraphNet_GAN(num_features = features, in_channels= in_channels, latent_dim = latent_dim).to(device)

discriminator = Discriminator2(num_features = [64,128,128,256], in_channels= in_channels).to(device)


for layer in generator.children():
       if hasattr(layer, 'reset_parameters'):
           layer.reset_parameters() 

for layer in discriminator.children():
       if hasattr(layer, 'reset_parameters'):
           layer.reset_parameters() 

cuda = True if torch.cuda.is_available() else False

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


learning_rate = 1e-4
learning_rate_D = 1e-5
optimizer_G = torch.optim.Adam(generator.parameters(), lr = learning_rate)

optimizer_D = torch.optim.Adam(discriminator.parameters(), lr = learning_rate_D)



from torch_geometric.data import Data


num_epochs = 600
counter = 0
update_D_freq = 1
print('Starting...')



for epoch in range(num_epochs):
    train_loss = []
    discriminator_losses = []
    generator.train()

    for i, data in enumerate(train_loader):
        

        
        data.x = data.x[:,mode].to(device)

        data.batch = data.batch.to(device)
        data.edge_index = data.edge_index.to(device)
        
        
        # labels = data['label']
    
        # Adversarial ground truths
        bs = data.x.shape[0]//40962
        valid = Variable(Tensor(bs, 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(bs, 1).fill_(0.0), requires_grad=False)

        optimizer_G.zero_grad()

    
        z = Variable(Tensor(np.random.normal(0, 1, (bs, latent_dim))))
    
        # Generate a batch of images
        gen_imgs = generator(z,bs)
    
        # Loss measures generator's ability to fool the discriminator
        gen_data = Data(x = gen_imgs, edge_index = data.edge_index, batch =data.batch)
        g_loss = adversarial_loss(discriminator(gen_data), valid)
    
        g_loss.backward()
        optimizer_G.step()

        if counter % update_D_freq ==0:
    
            optimizer_D.zero_grad()
            # Measure discriminator's ability to classify real from generated samples
            gen_data.x = gen_data.x.detach()
            real_loss = adversarial_loss(discriminator(data), valid)
            real_loss.backward(retain_graph=True)
            
            fake_loss = adversarial_loss(discriminator(gen_data), fake)
            fake_loss.backward()
            # real_loss = adversarial_loss(discriminator(data), valid)
    
            d_loss = (real_loss + fake_loss) / 2
        
            optimizer_D.step()

        # print('done one')
        counter += 1
    print(
        "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
        % (epoch, num_epochs, i, len(train_loader), d_loss.item(), g_loss.item())
    )

    
    if epoch %3 == 0:       
        
        for i in range(20):
            z = Variable(Tensor(np.random.normal(0, 1, (1, latent_dim))))
            gen_imgs = generator(z,1)
            gen_imgs_np = gen_imgs.detach().cpu().numpy()
            
            save_as_metric(gen_imgs_np, 'results/val_rounded_'+(str(i)))
                
torch.save(generator.state_dict(), 'results/generator_GAN_MoNet')
