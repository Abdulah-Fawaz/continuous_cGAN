#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 22:04:48 2022

@author: fa19
"""
import torch_geometric


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 14:46:24 2022

@author: fa19
"""
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
# val_dataset_arr = np.load('data/' + str(dataset) + '/validation.npy', allow_pickle = True)
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


# val_ds = My_dHCP_Data_Graph(input_arr = val_dataset_arr, 
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


batch_size = 2


train_loader = DataLoader(train_ds, batch_size=batch_size, 
                                           shuffle=True, 
                                           num_workers = 1)


# val_loader = DataLoader(val_ds, batch_size=1, 
#                                            shuffle=False, 
#                                            num_workers = 2)


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


from models_icam_mixed import GraphNet_Decoder, GraphNet_Discriminator


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
    


def torch_age_to_cardinal(x, L = 21 , minimum = 25):
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

c_dim = 8


model = GraphNet_Decoder(len(mode), c_dim=c_dim, age_dim = 21,device=device)

model = model.to(device)


discriminator = GraphNet_Discriminator(len(mode),c_dim = c_dim,age_dim = 21,device=device).to(device)


# for layer in model.children():
#     if hasattr(layer, 'reset_parameters'):
#         layer.reset_parameters() 


learning_rate = 1e-5
learning_rate_D = 1e-5

optimizer_G = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay= 1e-5)

optimizer_D = torch.optim.Adam(discriminator.parameters(), lr = learning_rate_D, weight_decay= 1e-5)


from torch.autograd import Variable
import torch_geometric 


best = 100
num_epochs = 10000
print('Starting...')



def weighted_l1loss(im1, im2, agediff):
    l1 = nn.L1Loss()(im1,im2)*torch.exp(-1*agediff/100)
    return l1

def tensor_difference(ims, ages):
    total_loss = 0
    for i in range(len(ims)):
        for j in range(i):
            total_loss += weighted_l1loss(ims[i], ims[j], torch.abs(ages[i]-ages[j]))
        
    return total_loss
    



adversarial_realness_loss = nn.BCELoss()
age_criterion = nn.BCELoss()
c_criterion = nn.L1Loss()
content_criterion = tensor_difference # for same content different age

for epoch in range(num_epochs):
    train_loss = []
    model.train()
    d_losses = []
    total_d_loss = []
    total_g_loss = []
    
    for i, data in enumerate(train_loader):
        
        images =  data['x'][:,mode].to(device) 
        images += torch.randn_like(images)


        real_age = data['y']
        real_age_v = torch_age_to_cardinal(real_age).to(device)
        
        bs = data.x.shape[0]//40962

        optimizer_G.zero_grad()

        
        ### generate random iamges of same age 
        
        random_Cs = torch.randn(bs,c_dim).to(device)
        single_A = torch.randint(21,46, (1,1)).repeat(bs,1).to(device)
        single_A_v = torch_age_to_cardinal(single_A).to(device)
        
        different_As = torch.randint(21,46, (bs,1)).to(device)
        different_A_v = torch_age_to_cardinal(different_As).to(device)
        
        same_Cs = torch.randn(1,c_dim).repeat(bs,1).to(device)
        
             
        
        valid = Variable(torch.Tensor(bs, 1).fill_(1.0), requires_grad=False).to(device)
        fake = Variable(torch.Tensor(bs, 1).fill_(0.0), requires_grad=False).to(device)

        ### same age diffrent content images generated 
        
        
        gen_imgs_sameA = model(random_Cs, single_A_v,bs) 
        gen_imgs_sameA += torch.randn_like(gen_imgs_sameA)/10
        
        
        # same content different age images generated 
        gen_imgs_sameC = model(same_Cs, different_A_v, bs)
        gen_imgs_sameC += torch.randn_like(gen_imgs_sameC)/10
        
        
        
        #impose realness age and content of sameA images
        
        real_sameA, age_sameA, content_sameA = discriminator(gen_imgs_sameA, bs)
        
        gloss1 = age_criterion(age_sameA, single_A_v) #check ages match
     
        gloss2 = adversarial_realness_loss(real_sameA, valid) #check realness matches

        gloss2_5 = c_criterion(content_sameA, random_Cs) # check content encoding matches

        gloss1.backward(retain_graph=True)
        gloss2.backward(retain_graph=True)
        gloss2_5.backward(retain_graph=True)

        
        
        # now again  for same C images
        
        
        real_sameC, age_sameC, content_sameC = discriminator(gen_imgs_sameC, bs)

        gloss3 = content_criterion(gen_imgs_sameC.reshape(bs,-1), different_As.float())
        
     
        gloss4 = adversarial_realness_loss(real_sameC, valid)

        gloss5 = c_criterion(content_sameC, same_Cs)
        gloss6 = age_criterion(age_sameC, different_A_v)

        gloss3.backward(retain_graph=True)
        gloss4.backward(retain_graph=True)
        gloss5.backward(retain_graph=True)
        gloss6.backward(retain_graph=False)
        


        gloss = gloss1 + gloss2 +gloss2_5 + gloss3+ gloss4 + gloss5 + gloss6
        
        # gloss.backward(retain_graph=True)
        
        optimizer_G.step()
        
        total_g_loss.append(gloss.item())
        torch.cuda.empty_cache()
        if i % 2 == 0:
            optimizer_D.zero_grad()
            
            
            real_actual_image, age_actual_image, _ = discriminator(images, bs) # now for real images
            real_sameA, age_sameA, content_sameA = discriminator(gen_imgs_sameA.detach(), bs)

            real_sameC, age_sameC, content_sameC = discriminator(gen_imgs_sameC.detach(), bs)

            d_loss1 = adversarial_realness_loss(real_sameC, fake)
            d_loss1.backward(retain_graph=True)
            
            d_loss2 = adversarial_realness_loss(real_sameA, fake) 
            
            d_loss2.backward(retain_graph=True)

            d_loss3 = adversarial_realness_loss(real_actual_image, valid) 
            d_loss3.backward(retain_graph=True)

            d_loss_age1 = age_criterion(age_actual_image , real_age_v)
            
            d_loss_age1.backward()
            
            d_loss = (d_loss1 + d_loss2 + d_loss3 + d_loss_age1)
            
            
            total_d_loss.append(d_loss.item())
    
            # d_loss.backward()
            optimizer_D.step()
        
        
        
        if i %5 == 0:
            gen_imgs_sameA = gen_imgs_sameA.reshape(bs,-1,2).detach().cpu().numpy()
            gen_imgs_sameC = gen_imgs_sameC.reshape(bs,-1,2).detach().cpu().numpy()
            for b in range(bs):
                
                save_as_metric(gen_imgs_sameA[b], 'sameA_'+str(b)+'_'+str(i))
                save_as_metric(gen_imgs_sameC[b], 'sameC_'+str(b)+'_'+str(i))

        torch.cuda.empty_cache()
        
    print('Epoch is ' +str(epoch)+ ' Gen Loss is ' ,  np.mean(total_g_loss), 'Discriminator Loss ', np.mean(total_d_loss))
    
    
        
      
        
        
        