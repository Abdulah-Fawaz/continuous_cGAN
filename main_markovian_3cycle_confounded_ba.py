#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 11:21:51 2022

@author: fa19
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 14:41:53 2022

@author: fa19
"""
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
dataset = 'ba'

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


weights = np.ones(len(train_ds))
frac_prems = sum(train_dataset_arr[:,-1]<37)/len(train_dataset_arr)

weight_prems = 15

positions = [i for i in range(len(train_dataset_arr)) if np.logical_and(train_dataset_arr[i,-1]<37, train_dataset_arr[i,1]>=37)]
# positions2 = [i for i in range(len(train_dataset_arr), 2*len(train_dataset_arr)) if np.logical_and(train_dataset_arr[i,-1]<37, train_dataset_arr[i,1]>=37)]

positions2 =np.array( positions) + len(train_dataset_arr)

positions = positions + list(positions2)

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


from markovian_models import GraphNet_Markovian_Generator, GraphNet_Markovian_Discriminator_Simple_Double, GraphNet_Markovian_Discriminator, GraphNet_Markovian_Discriminator_Simple

modality = 'myelination'

if modality == 'myelination':
    mode = [0]
    in_channels = 1

elif modality == 'curvature':
    mode = [1]
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

    

def IPloss(x,y,diff,bs):
    L1 = torch.nn.L1Loss()(x,y)
    L1 = torch.nn.L1Loss(reduction='none')(x.view(bs,-1,2),y.view(bs,-1,2)).view(bs,-1)
    L1 = torch.mean(L1,dim=1)
    
    factor = torch.exp(-1* torch.abs(diff))
    
    return torch.mean(L1*factor)

# from scipy.interpolate import griddata


# xy_points = np.load('data/suggested_ico_6_points.npy')
# xy_points[:,0] = (xy_points[:,0] + 0.1)%1
# grid = np.load('data/grid_170_square.npy')


# grid_x, grid_y = np.meshgrid(np.linspace(0, 1, 170), np.linspace(0.00, 1, 170))
# grid[:,0] = grid_x.flatten()
# grid[:,1] = grid_y.flatten()

# resdir = 'results/'+str(model_name)+'/'+str(style)+'/'

### age is 20 to 50

model = GraphNet_Markovian_Generator(len(mode), age_dim = 2,device=device)

model = model.to(device)


discriminator = GraphNet_Markovian_Discriminator_Simple_Double(len(mode),age_dim = 2,device=device).to(device)

# for layer in model.children():
#     if hasattr(layer, 'reset_parameters'):
#         layer.reset_parameters() 


learning_rate = 1e-3
learning_rate_D = 1e-4

optimizer_G = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay= 1e-5)

optimizer_D = torch.optim.Adam(discriminator.parameters(), lr = learning_rate_D, weight_decay= 1e-5)


from torch.autograd import Variable
import torch_geometric 


best = 100
num_epochs = 250
print('Starting...')



def randint_v(high_v, low=22):
    s = torch.zeros_like(high_v)
    for i in range(len(high_v)):
        s[i] = np.random.randint(low,high_v[i].item())
        
    return s

    

adversarial_loss = nn.BCELoss()
recon_loss = nn.SmoothL1Loss()

d_update_freq = 1

for epoch in range(num_epochs):
    train_loss = []
    model.train()
    d_losses = []
    g_losses = []
    
    for i, data in enumerate(train_loader):
        
        im1 =  data['x'][:,mode].to(device) 
        
        bs = data.x.shape[0]//40962


        A1_v = data['metadata'].to(device) # scan age
        A1_v = A1_v.reshape(bs,1).float()
        B1_v = data['y'].to(device) #birth age
        B1_v = B1_v.reshape(bs,1).float()

        # A1_v = torch_age_to_cardinal(A1).to(device)
        
        
        A2_v = A1_v
        
        
        # A2_v = torch_age_to_cardinal(A2).to(device)
        B2_v = randint_v(A1_v).to(device).float()
        
        A3_v = A1_v
        B3_v = randint_v(A3_v).to(device).float()

        # A3_v = torch_age_to_cardinal(A3).to(device)
        
     
        valid = Variable(torch.Tensor(bs, 1).fill_(1.0), requires_grad=False).to(device)
        fake = Variable(torch.Tensor(bs, 1).fill_(0.0), requires_grad=False).to(device)



        diff_ba_1 = B2_v - B1_v
        diff_ba_2 = B3_v - B2_v
        diff_ba_3 = B1_v - B3_v
        
        d1 = torch.cat((A1_v, diff_ba_1),dim=1)
        d2 = torch.cat((A1_v, diff_ba_2),dim=1)
        d3 = torch.cat((A1_v, diff_ba_3),dim=1)
        

        # diff1 = A2 - A1 
        # diff2 = A3 - A2
        
        # diff3 = A1 - A3
        
        optimizer_G.zero_grad()

        im2 = model(im1, d1)
        
        im3 = model(im2, d2)
        
        im4 = model(im3, d3)


        im1_recon_loss = recon_loss(im4, im1)  * 5
        
        im2_real_loss = adversarial_loss(discriminator(im2, A2_v, B2_v), valid)           
        im3_real_loss = adversarial_loss(discriminator(im3, A3_v, B3_v), valid)           
        
        iploss1 = IPloss(im1, im2, diff_ba_1,bs) * 4
        iploss2 = IPloss(im2, im3, diff_ba_2,bs) * 4
        iploss3 = IPloss(im4, im3, diff_ba_3,bs) * 4
        
        gloss = im1_recon_loss + im2_real_loss + im3_real_loss + iploss1 + iploss2 + iploss3
        g_losses.append(gloss.item())
        gloss.backward()
        
        optimizer_G.step()
        
        if i % d_update_freq==0 :

            
            optimizer_D.zero_grad()
            im2 = im2.detach()
            im3 = im3.detach()
            real_image_true = adversarial_loss(discriminator(im1+torch.randn_like(im1)/10,A1_v, B1_v), valid) * 10  
                    
            real_image_fake = adversarial_loss(discriminator(im1+torch.randn_like(im1)/10,A1_v, B2_v), fake) * 5 
            real_image_fake2 = adversarial_loss(discriminator(im1+torch.randn_like(im1)/10,A1_v, B3_v), fake) * 5
                    

            # fake_loss1 = adversarial_loss(discriminator(im2,A1_v), fake) *2
            # fake_loss2 = adversarial_loss(discriminator(im3,A1_v), fake) *2 
    
            fake_loss3 = adversarial_loss(discriminator(im2,A2_v, B2_v), fake)    
            fake_loss4 = adversarial_loss(discriminator(im3,A3_v, B3_v), fake)    
            
            fake_loss5 = adversarial_loss(discriminator(im1,A2_v, B2_v), fake)    
            fake_loss6 = adversarial_loss(discriminator(im1,A3_v, B3_v), fake)     

            fake_loss7 = adversarial_loss(discriminator(im2,A2_v, B1_v), fake)    
            fake_loss8 = adversarial_loss(discriminator(im3,A3_v, B2_v), fake)  

            # d_loss = real_image_true + fake_loss1 + fake_loss2 + fake_loss3 + fake_loss4 + fake_loss5 + fake_loss6

            d_loss = real_image_true  + fake_loss3 + fake_loss4 + fake_loss5 + fake_loss6 + fake_loss7+ fake_loss8 + real_image_fake + real_image_fake2
            d_losses.append(d_loss.item())
            d_loss.backward()
            
            optimizer_D.step()
        if i % 10 == 0:
            im2 = im2.reshape(bs,-1,len(mode)).detach().cpu().numpy()
            im3 = im3.reshape(bs,-1,len(mode)).detach().cpu().numpy()
            im1 = im1.reshape(bs,-1,len(mode)).detach().cpu().numpy()
            for b in range(min(3,bs)):
                save_as_metric(im1[b], 'im1'+'_'+str(i)+'_'+str(b))
                
                save_as_metric(im2[b], 'im2'+'_'+str(i)+'_'+str(b))
                save_as_metric(im3[b], 'im3'+'_'+str(i)+'_'+str(b))
 
        
    print('Epoch is ' +str(epoch)+ ' Gen Loss is ' ,  np.mean(g_losses), 'Discriminator Loss ', np.mean(d_losses))
    

import time
model.eval()    


############## this whole thing needs amending to include brth age!!! ##################

for i, data in enumerate(val_loader):
    
    im1 =  data['x'][:,mode].to(device) 
    
        
    
# im1 =  data['x'][:,mode].to(device) 
    
    bs = data.x.shape[0]//40962
    name = val_dataset_arr[i,0].split('_')[0]
    
    A1 = data['metadata'].to(device)
    print(A1)
    B1 = data['y'].to(device)
    # A1_v = torch_age_to_cardinal(A1).to(device)
    
    for ba in np.arange(32, A1.item()):
        A2 = A1
        B2 = torch.Tensor([ba]).to(device)
        if len(A2.shape)==1 :
            A2.unsqueeze(0)
        # if len(B2.shape) ==1:
        #     B2.unsqueeze(0)
            
        # A2_v = torch_age_to_cardinal(A2).to(device)
        
        
        
        diff_ba = torch.Tensor([ba]).unsqueeze(0).to(device)-B1
        
        D = torch.cat((A1, diff_ba), dim=1)
        
        im2 = model(im1, D)
        
        im1_np = im1.reshape(bs,-1,len(mode)).detach().cpu().numpy()
        
        im2 = im2.reshape(bs,-1,len(mode)).detach().cpu().numpy()
         
        save_as_metric(im1_np[0], 'val_original_2_'+'_'+str(A1[0].item())+'_'+str(B1.item())+'_'+str(name))
        
        save_as_metric(im2[0], 'val_fake_2_'+'_'+str(A1[0].item())+'_'+str(B2.item())+'_'+str(name))
    print(i, ' Complete')
    time.sleep(2)
    if i == 20:
        break
    

torch.save(model, '/home/fa19/Documents/neurips/3cycle_BA_myelin_generator_model_final2')
torch.save(model.state_dict(), '/home/fa19/Documents/neurips/3cycle_BA_myelin_generator_model_final_sd1')


# torch.save(discriminator, '/home/fa19/Documents/neurips/3cycle_BA_double_discriminator_model_final3')        
# torch.save(discriminator.state_dict(), '/home/fa19/Documents/neurips/3cycle_triple_discriminator_model_final_sd3')
        
# model = torch.load('/home/fa19/Documents/neurips/3cycle_BA_myelin_generator_model_final1')





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

save_loc = '/home/fa19/Documents/neurips/'

idxs = [92, 7, 'CC00389XX19'] 


idxs = all_shared[3]
#idxs = [439, 470, '4BN13']

#idxs = [383, 419, '0XX09']

# for idxs in all_shared:
subject = idxs[2]

idx1 = idxs[1]



data = everything.__getitem__(idx1)
im1 = data['x'][:,mode].to(device) 
    
        
                                        
# im1 =  data['x'][:,mode].to(device) 
    
bs = data.x.shape[0]//40962
name = everything_arr[idx1,0].split('_')[0]

true_age_1 = data['metadata'].to(device)

# true_age_1_v = torch_age_to_cardinal(true_age_1).to(device)
true_age_1_v = true_age_1.to(device)
true_ba =  data['y'].to(device)

for num in np.arange(28,true_age_1_v.item()):
    
    
    
    new_age = torch.Tensor([num]).to(device)
    if len(new_age.shape)==1 :
        new_age.unsqueeze(0)
    # new_age_v = torch_age_to_cardinal(new_age).to(device)
    new_age_v = new_age.to(device)
    
    difference = new_age_v - true_ba
    
    D = torch.cat((true_age_1_v, difference.unsqueeze(0)), dim=1)

    im2 = model(im1, D)
    
    im1_np = im1.reshape(bs,-1,len(mode)).detach().cpu().numpy()
    
    im2 = im2.reshape(bs,-1,len(mode)).detach().cpu().numpy()
     
    save_as_metric(im1_np[0], save_loc + '3cycle_confounded/test/val_original'+'_'+str(true_age_1[0].item())+'_'+str(true_ba.item()) + '_'+str(name))
    
    save_as_metric(im2[0], save_loc + '3cycle_confounded/test/val_fake'+'_'+str(new_age[0].item())+'_'+str(name))




idx2 = idxs[1]

data2 = everything.__getitem__(idx2)
im2 = data2['x'][:,mode].to(device) 
    
        
                                        
# im1 =  data['x'][:,mode].to(device) 
    
bs = data2.x.shape[0]//40962
name = everything_arr[idx2,0].split('_')[0]

true_age_2 = data2['metadata'].to(device)



im2_np = im2.reshape(bs,-1,len(mode)).detach().cpu().numpy()

save_as_metric(im2_np[0], '3cycle_confounded/test_again/val_original_secondscan'+'_'+str(true_age_2[0].item())+'_'+str(name))


print(i, ' Complete')

