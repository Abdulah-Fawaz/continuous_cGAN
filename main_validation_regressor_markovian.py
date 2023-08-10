#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 13:09:43 2022

@author: fa19
"""
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

save_loc = '3cycle/'
save_dir = save_root + save_loc


regressor_model_location = '/home/fa19/Documents/neurips/regressor_monet_std2'

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




device_number = 0

device = torch.device('cuda:' + str(device_number) if torch.cuda.is_available() else 'cpu')

print('device is ', device)
torch.cuda.set_device(device)


from markovian_models import GraphNet_Markovian_Generator, MoNet_Markovian_Generator, GraphNet_Markovian_Discriminator, GraphNet_Markovian_Discriminator_Simple
from regressor_model_monet import monet_polar_regression

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


test_dataset_arr = np.load('data/' + str(dataset) + '/test.npy', allow_pickle = True) 
# get only the terms from the test set
test_dataset_arr = test_dataset_arr[test_dataset_arr[:,-1]>=37]

test_ds = My_dHCP_Data_Graph(input_arr = test_dataset_arr, 
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
    
    
    for num in np.arange(32,45):
        
        
        
        new_age = torch.Tensor([num]).to(device)
        if len(new_age.shape)==1 :
            new_age.unsqueeze(0)
        # new_age_v = torch_age_to_cardinal(new_age).to(device)
        new_age_v = new_age.to(device)
        
        difference = new_age_v - true_age_1_v
        
        im2 = model(im1, difference.unsqueeze(0))
        # if 'MoNet' in model_name:
        #     im2 = restandardise(im2)
        print(im2.shape)
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

L = np.arange(32,45)

# import matplotlib.pyplot as plt

# fig = plt.figure()
# for _ in range(32,44):
#     plt.scatter(L, all_age_predictions[_], label = str(test_dataset_arr[_,1]))
# plt.plot(L,L, c='red')
# plt.xlabel('Input Ages')
# plt.ylabel('Predicted Ages')
# plt.legend()
# plt.show()


# import matplotlib.pyplot as plt
# fig = plt.figure()
# for _ in range(0,21):
#     plt.plot(L, all_age_predictions[_], label = str(test_dataset_arr[_,1]))
# plt.plot(L,L, c='red')
# plt.xlabel('Input Ages')
# plt.ylabel('Predicted Ages')

# plt.show()




MEAN = np.mean(all_age_predictions,axis=0)
STD = np.std(all_age_predictions, axis=0)

lower_interval = MEAN-STD
higher_interval = MEAN+STD

# np.save('4cycle_GNN_mean.npy', MEAN)
# np.save('4cycle_GNN_std.npy', STD)

gnn_3cycle_mean = np.load('3cycle_GNN_mean.npy')
gnn_3cycle_std = np.load('3cycle_GNN_std.npy')
gnn_3cycle_lower = gnn_3cycle_mean - gnn_3cycle_std
gnn_3cycle_higher = gnn_3cycle_mean + gnn_3cycle_std

# monet_mean = np.load('3cycle_Monet_mean_sulc_myelin.npy')
# monet_std = np.load('3cycle_Monet_std_sulc_myelin.npy')

gnn_2cycle_mean = np.load('2cycle_GNN_mean.npy')
gnn_2cycle_std = np.load('2cycle_GNN_std.npy')


gnn_4cycle_mean = np.load('4cycle_GNN_mean.npy')
gnn_4cycle_std = np.load('4cycle_GNN_std.npy')
gnn_4cycle_lower = gnn_4cycle_mean - gnn_4cycle_std
gnn_4cycle_higher = gnn_4cycle_mean + gnn_4cycle_std


# best_fit = np.load('best_fit_regressor.npy')
best_fit = np.load('best_fit_regressor2.npy')[:-1]
best_fit_interval =np.load('best_fit_regressor2_std2.npy')

lower_best_fit = best_fit-best_fit_interval
upper_best_fit = best_fit+best_fit_interval

import matplotlib.pyplot as plt
fig = plt.figure()
# Shade the area between y1 and y2
plt.fill_between(L, lower_interval, higher_interval,
                 facecolor="gray", # The fill color
                 color='black',       # The outline color
                 alpha=0.2)          # Transparency of the fill

# plt.plot(L, gnn_4cycle_mean, c='black',linestyle='dashed', label = '4cycle')
# plt.plot(L, MEAN, c='black',linestyle='dotted', label = '2cycle')

plt.xlabel('Target Ages / weeks')
plt.ylabel('Predicted Ages / weeks')
plt.plot(L, MEAN, c = 'black',linestyle='solid',label = 'Our Model')
plt.plot(L,best_fit[:], c='black',linestyle='dashed',label='Regressor Baseline')
plt.plot(L,upper_best_fit[:], c='red',linewidth=0.8,linestyle='dashed')
plt.plot(L,lower_best_fit[:], c='red', linewidth=0.8,linestyle='dashed', label='Regressor 90% Confidence')

# plt.plot(L,L, c='red',linestyle='dotted', label='ground truth')
# plt.scatter(L, all_age_predictions[10], marker='x', label= 'synthetic images')
# plt.scatter(test_ds.__getitem__(10).y.item(), 39.14,marker = '+', s=80, label='original image')
plt.legend()
# Show the plot
plt.show()



num_cycles = [2, 3, 4, 5]
all_means = [2.70, 1.02, 0.80, 1.58]
all_stds = [0.70, 0.28, 0.38, 0.49]

import matplotlib.pyplot as plt
fig = plt.figure()
plt.scatter(num_cycles, all_means)
# Shade the area between y1 and y2
plt.errorbar(num_cycles, all_means,yerr=all_stds, fmt="o", c='dodgerblue', ecolor='orange')
# Show the plot
plt.xlabel('Number of Cycles')
plt.ylabel('Means Absolute Error (MAE) in Predicted Age / Weeks')
plt.xticks(np.arange(min(num_cycles), max(num_cycles)+1, 1.0))

plt.show()

L2_differences = [-14.25, 11.08,-26.62,0]
L2_quantiles = [[-36.179245,  10.38117981] , [-4.02842712, 35.02592468], [-45.19718933 ,  0.56135559], [0,0]]


psnr_differences = [2.163, 2.150, 1.71, 0]
psnr_stds = [2.62, 1.69, 2.81,0]
psnr_quantiles = [[0.23, 3.6], [0.82,3.24], [0.02, 4],[0,0]]

psnr_errors = []
L2_errors = []
L2_first_errors = []

L2_first = [203.4,167.2,231.3,0]

L2_first_quantiles = [[167.4,244.4], [136.3, 195.7],[201.3,264.1]]


ssim_first = [ 0.74, 0.82,0.70, 0.75]


for i,n in enumerate(psnr_quantiles):
    psnr_errors.append([psnr_differences[i]-n[0],-1*psnr_differences[i]+n[1]])

for i,n in enumerate(L2_quantiles):
    L2_errors.append([L2_differences[i]-n[0],-1*L2_differences[i]+n[1]])
    

for i,n in enumerate(L2_first_quantiles):
    L2_first_errors.append([L2_first[i]-n[0],-1*L2_first[i]+n[1]])
    

import matplotlib.pyplot as plt
fig = plt.figure()
for k in range(3):
    
    # plt.scatter(psnr_differences[k], all_means[k], label = str(num_cycles[k])+' cycles', marker='x')
    plt.errorbar(L2_first[k], all_means[k], label = str(num_cycles[k])+' cycles',
                 yerr=all_stds[k],xerr=np.array(L2_first_errors[k])[:,np.newaxis], fmt="o")
# Shade the area between y1 and y2
# Show the plot

plt.xlabel(r'$\longleftarrow\:$ Subject Specificity'
           '\n'  # Newline: the backslash is interpreted as usual
           r'(Δ Image Similarity)')

# plt.xlabel(r'Subject Specificity $\:\longrightarrow$' + '\n' + '(Δ Image Similarity)')
plt.ylabel(r'$\longleftarrow$'+'Age Predicion Accuracy'+'\n'+' (MAE / weeks)')

plt.legend()
plt.show()

cyclegan_l2 = [32.3]
cyclegan_regressor_mae = [7.22]
cyclegan_ssim = [0.824]

baseline_ssim = [0.79]
baseline_mae = [7.4]

import matplotlib.pyplot as plt
fig = plt.figure()
for k in range(4):
    
    # plt.scatter(psnr_differences[k], all_means[k], label = str(num_cycles[k])+' cycles', marker='x')
    plt.scatter(ssim_first[k], all_means[k], label = str(num_cycles[k])+' cycles', marker='x',s=60)
plt.scatter(cyclegan_ssim,cyclegan_regressor_mae, label = 'cyclegan' , marker= 'x',s=60)
# plt.scatter(baseline_ssim,baseline_mae, label = 'baseline' , marker= 'x',s=40)
plt.axhline(y=baseline_mae, color='gray', linestyle='dotted')
# Shade the area between y1 and y2
# Show the plot

plt.xlabel(r'Subject Specificity$\:\longrightarrow$'
           '\n'  # Newline: the backslash is interpreted as usual
           r'(Image Similarity)')

#Δ 
# plt.xlabel(r'Subject Specificity $\:\longrightarrow$' + 'S\n' + '(Δ Image Similarity)')
plt.ylabel(r'$\longleftarrow$'+'Age Predicion Accuracy'+'\n'+' (MAE / weeks)')
plt.text(0.72, baseline_mae[0]-0.3, "Baseline MAE")
plt.axhline(y=0.58, color='gray', linestyle='dashed')
plt.text(0.72, 0.78, "Regressor MAE")


plt.legend()
plt.show()


# ################### quantify longitudinal data / preterms #################
import math
def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = original.max()
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    return psnr

    
    

# birth_ages_list = []
# second_scan_age_list = []


# im_similarities_list = []


# count = 0

# for idxs in all_shared:


    
#     subject = idxs[2]
    
#     idx1 = idxs[0]
    
    
    
#     data = everything.__getitem__(idx1)
#     im1 = data['x'][:,mode]
#     im1_np = np.array(im1)
    
#     im1 = im1.to(device) 
       
            
#     idx2 = idxs[1]

#     data2 = everything.__getitem__(idx2)
#     im2 = data2['x'][:,mode]
    
        
    
    
    
    
#     im2_np = np.array(im2)
#     im2 = im2.to(device)
                                            
#     # im1 =  data['x'][:,mode].to(device) 
        
#     bs = data.x.shape[0]//40962
#     name = everything_arr[idx1,0].split('_')[0]
    
#     true_age_1 = data['metadata'].to(device)
    
#     # true_age_1_v = torch_age_to_cardinal(true_age_1).to(device)
#     true_age_1_v = true_age_1.to(device)

#     true_age_2_v = data2['metadata'].to(device)
#     birth_age = data2['y']
#     birth_ages_list.append(birth_age.item())
#     second_scan_age_list.append(true_age_2_v.item())
#     temp_similarity = []
#     for num in np.arange(32,45):
        
        
        
#         new_age = torch.Tensor([num]).to(device)
#         if len(new_age.shape)==1 :
#             new_age.unsqueeze(0)
#         # new_age_v = torch_age_to_cardinal(new_age).to(device)
#         new_age_v = new_age.to(device)
        
#         difference = new_age_v - true_age_1_v
        
#         imgen = model(im1, difference)
        
#         im1_np = im1.reshape(bs,-1,len(mode)).detach().cpu().numpy()
        
#         imgen = imgen.reshape(bs,-1,len(mode)).detach().cpu().numpy()
         
        
#         temp_similarity.append( PSNR(np.array(imgen), im2_np))
    
        
#     im_similarities_list.append(temp_similarity)
        


# max_im_similarities = [np.argmax(i) for i in im_similarities_list]


# A = np.arange(32,45)



# max_age_similarities = np.array(max_im_similarities) + 32
# diff_age_similarties = second_scan_age_list - max_age_similarities

# import matplotlib.pyplot as plt
# fig = plt.figure()
# # plt.scatter(np.array(second_scan_age_list)-np.array(birth_ages_list), diff_age_similarties)
# # plt.scatter(np.array(birth_ages_list), diff_age_similarties)
# # plt.scatter(np.array(birth_ages_list), np.abs(diff_age_similarties))
# plt.scatter(np.array(second_scan_age_list)-np.array(birth_ages_list), np.abs(diff_age_similarties))
# plt.axhline(y=0, c='orange')
# plt.show()




# L = list(np.arange(32,45))

# import matplotlib.pyplot as plt

# ax = plt.subplot()

# # Define tick label
# # Display graph

# # for i in range(len(im_similarities_list)):
# for i in range(3):

#     plt.plot(L,im_similarities_list[i])
# plt.show()

# from collections import Counter

# C = Counter(np.round(diff_age_similarties))

# ############################# quantify but with SSIM #################

from quant_utilities import *

def mse_numpy(A,B):
    return np.linalg.norm((A-B))

simf = mse_numpy
birth_ages_list = []
second_scan_age_list = []


im_similarities_list = []


count = 0



################### finihed loading the model #################
norm_style=None

test_dataset_arr = np.load('data/' + 'scan_age' + '/test.npy', allow_pickle = True)


test_ds = My_dHCP_Data_Graph(input_arr = test_dataset_arr, 
                    warped_files_directory = warped_directory,
                    unwarped_files_directory = unwarped_directory,
                    edges=edges,
                    rotations= False,
                    number_of_warps = 0,
                    parity_choice = test_parity,
                    smoothing = False,
                    normalisation = 'std',
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
                   smoothing = False,
                   normalisation = 'std',
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
        # print(everything_arr[shared], shared)
        if everything_arr[shared[0]][1] > everything_arr[shared[1]][1]:
            complete = shared[::-1]
        else:
            complete = shared
        complete.append(n)
        all_shared.append(complete)
        

# print(all_shared)

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
         
        
        temp_similarity.append( simf(np.array(imgen[0]), im2_np))
    
        
    im_similarities_list.append(temp_similarity)
        


max_im_similarities = [np.argmax(i) for i in im_similarities_list]


# A = np.arange(32,45)

import pandas as pd

# cognitive_ds = pd.read_csv('/home/fa19/Downloads/DHCPNDH1_DATA_2020-11-23_0903 (2).csv')
# cognitive_ds = cognitive_ds[['participationid','cog_comp']]

# cog_outcomes = []
# for idxs in all_shared:
#     name = idxs[2]
#     found = cognitive_ds.loc[cognitive_ds['participationid']==name]['cog_comp']
#     if len(found)!=0:
#         F = found.item()
#         if np.isnan(F)==False:
            
#             cog_outcomes.append(found.item())
#         else:
#             cog_outcomes.append(-1)
#     else:
#         cog_outcomes.append(-1)



new_cog_ds = pd.read_csv('/home/fa19/Downloads/DHCPNDH1-BayleyDataForNIH_DATA_2022-11-21_1329.csv')
new_cog_ds = new_cog_ds[['participationid','language_comp']]

lang_outcomes = []
for idxs in all_shared:
    name = idxs[2]
    found = new_cog_ds.loc[new_cog_ds['participationid']==name]['language_comp']
    if len(found)!=0:
        F = found.item()
        if np.isnan(F)==False:
            
            lang_outcomes.append(found.item())
        else:
            lang_outcomes.append(-1)
    else:
        lang_outcomes.append(-1)



new_cog_ds = pd.read_csv('/home/fa19/Downloads/DHCPNDH1-BayleyDataForNIH_DATA_2022-11-21_1329.csv')
new_cog_ds = new_cog_ds[['participationid','motor_comp']]

motor_outcomes = []
for idxs in all_shared:
    name = idxs[2]
    found = new_cog_ds.loc[new_cog_ds['participationid']==name]['motor_comp']
    if len(found)!=0:
        F = found.item()
        if np.isnan(F)==False:
            
            motor_outcomes.append(found.item())
        else:
            motor_outcomes.append(-1)
    else:
        motor_outcomes.append(-1)



qchat_ds = np.load('/home/fa19/Documents/Surface-ICAM/data/qchat/full.npy',allow_pickle=True)
for row in qchat_ds:
    row[0] = row[0].split('_')[0]
qchat_dict = {qchat_ds[i,0]:qchat_ds[i,-1] for i in range(len(qchat_ds))}

qchat_outcomes = []
for idxs in all_shared:
    name = idxs[2]
    qchat_outcomes.append(qchat_dict.get(name,-1))
qchat_outcomes = np.array(qchat_outcomes)



cog_ds = np.load('/home/fa19/Documents/Surface-ICAM/data/cogcomp/full.npy',allow_pickle=True)
for row in cog_ds:
    row[0] = row[0].split('_')[0]
cog_dict = {cog_ds[i,0]:cog_ds[i,-1] for i in range(len(cog_ds))}

cog_outcomes = []
for idxs in all_shared:
    name = idxs[2]
    cog_outcomes.append(cog_dict.get(name,-1))
cog_outcomes = np.array(cog_outcomes)

cog_outcomes_filtered =  cog_outcomes[qchat_outcomes.astype(float)>1]
qchat_outcomes_filtered =  qchat_outcomes[qchat_outcomes.astype(float)>1]

qchat_outcomes_filtered =  qchat_outcomes_filtered[cog_outcomes_filtered>1]
cog_outcomes_filtered =  cog_outcomes_filtered[cog_outcomes_filtered>1]





max_age_similarities = np.array(max_im_similarities) + 32
# diff_age_similarties = second_scan_age_list - max_age_similarities
true_age_indices =np.round(np.array(second_scan_age_list))-32

true_age_similarities = []
for i in range(len(im_similarities_list)):
    true_age_similarities.append(im_similarities_list[i][int(true_age_indices[i])])
              
filtered_indices_cog = np.array(cog_outcomes)>10

filtered_indices_qchat = np.array(qchat_outcomes)>11
# filtered_indices_lang = np.array(lang_outcomes)>10
# filtered_indices_motor = np.array(motor_outcomes)>10


best_im_similarity = np.array([max(i) for i in im_similarities_list])


F = np.array(motor_outcomes)>10
# X = np.load('/home/fa19/Documents/neurips/3cycle_GNN_original_ssim_longitudinal_ssim.npy').T

# X = np.load('/home/fa19/Documents/neurips/3cycle_im_sim_array_ssim.npy')
# true_age_similarities = []
# for i in range(len(X)):
#     true_age_similarities.append(X[i][int(true_age_indices[i])])
              

# print(np.corrcoef(np.array(lang_outcomes)[filtered_indices_lang], np.array(true_age_similarities)[filtered_indices_lang]))
print(np.corrcoef(np.array(qchat_outcomes)[filtered_indices_qchat], np.array(true_age_similarities)[filtered_indices_qchat]))
# print(np.corrcoef(np.array(motor_outcomes)[filtered_indices_motor], np.array(true_age_similarities)[filtered_indices_motor]))
print(np.corrcoef(np.array(cog_outcomes)[filtered_indices_cog], np.array(true_age_similarities)[filtered_indices_cog]))



# print(np.corrcoef(np.array(lang_outcomes)[filtered_indices_lang], (np.array(true_age_similarities)/np.array(best_im_similarity))[filtered_indices_lang]))
print(np.corrcoef(np.array(qchat_outcomes)[filtered_indices_qchat], (np.array(true_age_similarities)-np.array(best_im_similarity))[filtered_indices_qchat]))
# print(np.corrcoef(np.array(motor_outcomes)[filtered_indices_motor],(np.array(true_age_similarities)/np.array(best_im_similarity))[filtered_indices_motor]))
print(np.corrcoef(np.array(cog_outcomes)[filtered_indices_cog], (np.array(true_age_similarities)-np.array(best_im_similarity))[filtered_indices_cog]))




import matplotlib.pyplot as plt
fig = plt.figure()
# plt.scatter(np.array(second_scan_age_list)-np.array(birth_ages_list), diff_age_similarties)
# plt.scatter(np.array(birth_ages_list), diff_age_similarties)
# plt.scatter(np.array(birth_ages_list), np.abs(diff_age_similarties))
# plt.scatter(np.array(second_scan_age_list)-np.array(birth_ages_list), np.abs(diff_age_similarties))
# plt.scatter(np.array(birth_ages_list), np.array(true_age_similarities),s=20, marker='x')
# plt.plot(np.array(birth_ages_list), A*np.array(birth_ages_list)+B, c='orange')
# plt.scatter(np.array(cog_outcomes)[filtered_indices], np.array(diff_age_similarties)[filtered_indices],s=20, marker='x')
# plt.scatter(np.array(qchat_outcomes)[filtered_indices], np.array(true_age_similarities)[filtered_indices],s=20, marker='x')
# plt.scatter(np.array(lang_outcomes)[filtered_indices], np.array(true_age_similarities)[filtered_indices],s=20, marker='x')
plt.scatter(np.array(motor_outcomes)[filtered_indices], np.array(true_age_similarities)[filtered_indices],s=20, marker='x')

plt.ylabel('Real Scan Age minus Apparent Age')
plt.xlabel('BayLeys Score')

plt.show()





import matplotlib.pyplot as plt
fig = plt.figure()
# plt.scatter(np.array(second_scan_age_list)-np.array(birth_ages_list), diff_age_similarties)
# plt.scatter(np.array(birth_ages_list), diff_age_similarties)
# plt.scatter(np.array(birth_ages_list), np.abs(diff_age_similarties))
# plt.scatter(np.array(second_scan_age_list)-np.array(birth_ages_list), np.abs(diff_age_similarties))
# plt.scatter(np.array(birth_ages_list), np.array(true_age_similarities),s=20, marker='x')
# plt.plot(np.array(birth_ages_list), A*np.array(birth_ages_list)+B, c='orange')
# plt.scatter(np.array(cog_outcomes)[filtered_indices], np.array(diff_age_similarties)[filtered_indices],s=20, marker='x')
# plt.scatter(np.array(qchat_outcomes)[filtered_indices], np.array(true_age_similarities)[filtered_indices],s=20, marker='x')
# plt.scatter(np.array(lang_outcomes)[filtered_indices], np.array(true_age_similarities)[filtered_indices],s=20, marker='x')
plt.scatter(np.array(motor_outcomes)[F], np.array(lang_outcomes)[F],s=20, marker='x')

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

# #################### incremental generation #########







################### finihed loading the model #################
norm_style=None

test_dataset_arr = np.load('data/' + 'scan_age' + '/test.npy', allow_pickle = True)


test_ds = My_dHCP_Data_Graph(input_arr = test_dataset_arr, 
                    warped_files_directory = warped_directory,
                    unwarped_files_directory = unwarped_directory,
                    edges=edges,
                    rotations= False,
                    number_of_warps = 0,
                    parity_choice = test_parity,
                    smoothing = False,
                    normalisation = 'std',
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
                   smoothing = False,
                   normalisation = 'std',
                   projected = False,
                   sample_only = True, #if false, will go through every warp, not just one random warp
                   output_as_torch = True,
                   )






from quant_utilities import *
def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr
def PSNR_scaled(a,b):
    minimum = min(a.min(), b.min())
    maximum = max(a.max(),b.max())
    
    a = a - minimum
    a = a / (maximum-minimum)
    
    b = b - minimum
    b = b / (maximum-minimum)
    
    a *= 255
    b *= 255
    
    a = np.round(a)
    b = np.round(b)
    
    return PSNR(a,b)


from scipy.signal import correlate
def my_mse_numpy(a,b):
    return mse_numpy(a,b)

def norm(a,b):
    return np.linalg.norm(a-b)


def my_ssim(a,b):
    minimum = min(a.min(), b.min())
    maximum = max(a.max(),b.max())
    
    a = a - minimum
    a = a / (maximum-minimum)
    
    b = b - minimum
    b = b / (maximum-minimum)
    
    a *= 255
    b *= 255
    
    a = np.round(a)
    b = np.round(b)
    return final_ssim_batched(a,b)


sim_func = mse_numpy
# birth_ages_list = []
# second_scan_age_list = []


original_im_sim_list = []
second_im_sim_list = []
first_im_sim_list = []

second_scan_age_list = []
diff_ages = []
indices_used = []

third_im_sim_list = []


xy_points = np.load('data/equirectangular_ico_6_points.npy')
xy_points[:,0] = (xy_points[:,0] + 0.1)%1
grid = np.load('data/grid_170_square.npy')


grid_x, grid_y = np.meshgrid(np.linspace(0.02, 0.98, 170), np.linspace(0.02, 0.98, 170))
grid[:,0] = grid_x.flatten()
grid[:,1] = grid_y.flatten()

from scipy.interpolate import griddata
from skimage.metrics import structural_similarity as ssim


def project_and_normalise(a,b,c):
      
    a = griddata(xy_points, a, grid, 'nearest')
    a = a.reshape(170,170,2)

    b = griddata(xy_points, b, grid, 'nearest')
    b = b.reshape(170,170,2)

    c = griddata(xy_points, c, grid, 'nearest')
    c = c.reshape(170,170,2)
    
    minimums_a = np.min(a, axis=(0,1))
    minimums_b = np.min(b, axis=(0,1))
    minimums_c = np.min(c, axis=(0,1))

    maximums_a = np.max(a, axis=(0,1))
    maximums_b = np.max(b, axis=(0,1))
    maximums_c = np.max(c, axis=(0,1))
    
    minimum = np.minimum(minimums_a, minimums_b, minimums_c)
    maximum = np.maximum(maximums_a, maximums_b,maximums_c)
    

    a = a - minimum
    a = a / (maximum-minimum)
    
    b = b - minimum
    b = b / (maximum-minimum)
    
    c = c - minimum
    c = c / (maximum-minimum)

    a *= 255
    b *= 255
    c *= 255
    
    a = np.round(a)
    b = np.round(b)
    c = np.round(c)
    
    return a,b,c

def ssim2(a,b):
    return ssim(a[:,:,0],b[:,:,0],multichannel=True)
def projected_ssim(a,b,c):

    
    a = griddata(xy_points, a, grid, 'nearest')
    a = a.reshape(170,170,2)

    b = griddata(xy_points, b, grid, 'nearest')
    b = b.reshape(170,170,2)


    minimums_a = np.min(a, axis=(0,1))
    minimums_b = np.min(b, axis=(0,1))
    # minimums_c = np.min(c, axis=(0,1))

    maximums_a = np.max(a, axis=(0,1))
    maximums_b = np.max(b, axis=(0,1))
    # maximums_c = np.max(c, axis=(0,1))
    
    minimum = np.minimum(minimums_a, minimums_b)
    maximum = np.maximum(maximums_a, maximums_b)
    

    a = a - minimum
    a = a / (maximum-minimum)
    
    b = b - minimum
    b = b / (maximum-minimum)
    
    # c = c - minimum
    # c = c / (maximum-minimum)

    a *= 255
    b *= 255
    # c*= 255
    
    a = np.round(a)
    b = np.round(b)
    # c= np.round(c)
    
    

    return ssim(a[:,:,0],b[:,:,0])


    
first_scan_age_list = []


sim_func = mse_numpy
# birth_ages_list = []
# second_scan_age_list = []


original_im_sim_list = []
second_im_sim_list = []
first_im_sim_list = []

second_scan_age_list = []
diff_ages = []
indices_used = []

third_im_sim_list = []
for i, idxs in enumerate(all_shared):


    
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
    first_scan_age_list.append(true_age_1_v.item())

    second_scan_age_list.append(true_age_2_v.item())
    birth_age = data2['y']
    birth_ages_list.append(birth_age.item())
    temp_similarity = []
    
    difference = true_age_2_v - true_age_1_v
    diff_ages.append(difference.item())
    imgen = model(im1, difference)
    # im1_np = im1.reshape(bs,-1,len(mode)).detach().cpu().numpy()
    difference_zero = true_age_1_v - true_age_1_v

    imgen_np = imgen.detach().cpu().numpy()
    
    # im1_np, im2_np, imgen_np = project_and_normalise(im1_np, im2_np, imgen_np)
    
    
    original_im_sim_list.append(sim_func(np.array(im1_np), im2_np))

    second_im_sim_list.append(sim_func(imgen_np, im2_np))
    # imgen2 = model(im1, difference_zero )
    first_im_sim_list.append(sim_func(imgen_np, im1_np))
    third_im_sim_list.append(sim_func(im2_np-imgen_np, imgen_np-im1_np))
    indices_used.append(i)
    if i == 2:
        second_image = im2_np
        first_image = im1_np
        gen_image = imgen.detach().cpu().numpy()
    print('done')


        
X = np.array(second_im_sim_list)
Y = X - np.array(first_im_sim_list)
Z = np.array(third_im_sim_list)
from scipy.stats import linregress

slope, intercept, r, p, se = linregress( np.array(first_im_sim_list), X)
predictions = np.array(slope)*np.array(first_im_sim_list)+np.array(intercept)

x = X - predictions

age_filtered = np.array(second_scan_age_list)>40


filtered_indices_cog = np.array(cog_outcomes)>10
filtered_indices_qchat = np.array(qchat_outcomes)>11


filter_qchat = np.logical_and(filtered_indices_qchat, age_filtered)
filter_cog = np.logical_and(filtered_indices_cog, age_filtered)

# print(np.corrcoef(np.array(lang_outcomes)[filtered_indices_lang], X[filtered_indices_lang]))
print(np.corrcoef(np.array(qchat_outcomes)[filtered_indices_qchat], X[filtered_indices_qchat]))
# print(np.corrcoef(np.array(motor_outcomes)[filtered_indices_motor], X[filtered_indices_motor]))
print(np.corrcoef(np.array(cog_outcomes)[filtered_indices_cog], X[filtered_indices_cog]))



# print(np.corrcoef(np.array(lang_outcomes)[filtered_indices_lang], Y[filtered_indices_lang]))
print(np.corrcoef(np.array(qchat_outcomes)[filtered_indices_qchat], Y[filtered_indices_qchat]))
# print(np.corrcoef(np.array(motor_outcomes)[filtered_indices_motor], Y[filtered_indices_motor]))
print(np.corrcoef(np.array(cog_outcomes)[filtered_indices_cog], Y[filtered_indices_cog]))


# print(np.corrcoef(np.array(lang_outcomes)[filtered_indices_lang], X[filtered_indices_lang]))
print(np.corrcoef(qchat_outcomes[filter_qchat], X[filter_qchat]))
# print(np.corrcoef(np.array(motor_outcomes)[filtered_indices_motor], X[filtered_indices_motor]))
print(np.corrcoef(np.array(cog_outcomes)[filter_cog], X[filter_cog]))



# print(np.corrcoef(np.array(lang_outcomes)[filtered_indices_lang], Y[filtered_indices_lang]))
print(np.corrcoef(np.array(qchat_outcomes)[filter_qchat], Y[filter_qchat]))
# print(np.corrcoef(np.array(motor_outcomes)[filtered_indices_motor], Y[filtered_indices_motor]))
print(np.corrcoef(np.array(cog_outcomes)[filter_cog], Y[filter_cog]))



plt.figure()
plt.scatter(np.array(Y)[filtered_indices_cog],np.array(cog_outcomes)[filtered_indices_cog])
plt.show()   




full_list = []
for row in everything_arr:
    name = row[0].split('_')[0]
    cog_1 = cog_dict.get(name,-1)
    ba_1 = row[1]
    full_list.append([cog_1, ba_1])


# #         new_age = torch.Tensor([num]).to(device)
# #         if len(new_age.shape)==1 :
# #             new_age.unsqueeze(0)
# #         # new_age_v = torch_age_to_cardinal(new_age).to(device)
# #         new_age_v = new_age.to(device)
        
# #         difference = new_age_v - true_age_1_v

# #         imgen = model(im1, difference)
        
# #         im1_np = im1.reshape(bs,-1,len(mode)).detach().cpu().numpy()
# #         im1 = imgen.squeeze(0)
# #         true_age_1_v = new_age_v
#         imgen = imgen.reshape(bs,-1,len(mode)).detach().cpu().numpy()
         
        
#         temp_similarity.append( PSNR(np.array(imgen), im2_np))
#         print('done')
#     im_similarities_list.append(temp_similarity)
        


# max_im_similarities = [np.argmax(i) for i in im_similarities_list]


# A = np.arange(32,45)

second_im_sim_list = []
first_im_sim_list = []
original_im_sim_list = []

sim_func = final_ssim_batched
root_dir = '/home/fa19/Documents/neurips/3cycle_longitudinal_myelin_sulc/'

for i, idxs in enumerate(all_shared):


    
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
    

    second_scan_age_list.append(true_age_2_v.item())
    birth_age = data2['y']
    birth_ages_list.append(birth_age.item())
    temp_similarity = []
    rounded_age = np.round(true_age_2_v.item())
    
    file = f'{root_dir}val_fake_{rounded_age}_{name}.metric.shape.gii'
    imgen_np = np.zeros([40962,2])
    imgen_np[:,0] = nb.load(file).darrays[0].data
    imgen_np[:,1] = nb.load(file).darrays[1].data
    
    
    # difference = true_age_2_v - true_age_1_v
    # diff_ages.append(difference.item())
    # imgen = model(im1, difference)
    # im1_np = im1.reshape(bs,-1,len(mode)).detach().cpu().numpy()
    difference_zero = true_age_1_v - true_age_1_v

    # imgen_np = imgen.detach().cpu().numpy()
    
    # im1_np, im2_np, imgen_np = project_and_normalise(im1_np, im2_np, imgen_np)
    
    
    original_im_sim_list.append(sim_func(np.array(im1_np), im2_np))

    second_im_sim_list.append(sim_func(imgen_np, im2_np))
    # imgen2 = model(im1, difference_zero )
    first_im_sim_list.append(sim_func(imgen_np, im1_np))
    third_im_sim_list.append(sim_func(im2_np-imgen_np, imgen_np-im1_np))
    indices_used.append(i)
    if i == 2:
        second_image = im2_np
        first_image = im1_np
        gen_image = imgen_np
    print('done')


        
X = np.array(second_im_sim_list)
Y = X / np.array(first_im_sim_list)

from scipy.stats import linregress

slope, intercept, r, p, se = linregress( np.array(first_im_sim_list), X)
predictions = np.array(slope)*np.array(first_im_sim_list)+np.array(intercept)

x = X - predictions

age_filtered = np.array(second_scan_age_list)>40


filtered_indices_cog = np.array(cog_outcomes)>10
filtered_indices_qchat = np.array(qchat_outcomes)>11


# filter_qchat = np.logical_and(filtered_indices_qchat, age_filtered)
# filter_cog = np.logical_and(filtered_indices_cog, age_filtered)

# print(np.corrcoef(np.array(lang_outcomes)[filtered_indices_lang], X[filtered_indices_lang]))
print(np.corrcoef(np.array(qchat_outcomes)[filtered_indices_qchat], X[filtered_indices_qchat]))
# print(np.corrcoef(np.array(motor_outcomes)[filtered_indices_motor], X[filtered_indices_motor]))
print(np.corrcoef(np.array(cog_outcomes)[filtered_indices_cog], X[filtered_indices_cog]))\



# print(np.corrcoef(np.array(lang_outcomes)[filtered_indices_lang], Y[filtered_indices_lang]))
print(np.corrcoef(np.array(qchat_outcomes)[filtered_indices_qchat], Y[filtered_indices_qchat]))
# print(np.corrcoef(np.array(motor_outcomes)[filtered_indices_motor], Y[filtered_indices_motor]))
print(np.corrcoef(np.array(cog_outcomes)[filtered_indices_cog], Y[filtered_indices_cog]))


# print(np.corrcoef(np.array(lang_outcomes)[filtered_indices_lang], X[filtered_indices_lang]))
print(np.corrcoef(qchat_outcomes[filter_qchat], X[filter_qchat]))
# print(np.corrcoef(np.array(motor_outcomes)[filtered_indices_motor], X[filtered_indices_motor]))
print(np.corrcoef(np.array(cog_outcomes)[filter_cog], X[filter_cog]))



# print(np.corrcoef(np.array(lang_outcomes)[filtered_indices_lang], Y[filtered_indices_lang]))
print(np.corrcoef(np.array(qchat_outcomes)[filter_qchat], Y[filter_qchat]))
# print(np.corrcoef(np.array(motor_outcomes)[filtered_indices_motor], Y[filtered_indices_motor]))
print(np.corrcoef(np.array(cog_outcomes)[filter_cog], Y[filter_cog]))



# print(np.corrcoef(np.array(lang_outcomes)[filtered_indices_lang], Y[filtered_indices_lang]))
print(np.corrcoef(np.array(qchat_outcomes)[filter_qchat], x[filter_qchat]))
# print(np.corrcoef(np.array(motor_outcomes)[filtered_indices_motor], Y[filtered_indices_motor]))
print(np.corrcoef(np.array(cog_outcomes)[filter_cog], x[filter_cog]))

plt.figure()
plt.scatter(np.array(Y)[filtered_indices_cog],np.array(cog_outcomes)[filtered_indices_cog])
plt.show()   







A = np.vstack([np.array(first_scan_age_list)[:90], np.array(X)])

# from sklearn.linear_model import linregress

#slope, intercept, r, p, se =linregress(A.T,cog_outcomes)

from sklearn import linear_model

regr = linear_model.LinearRegression()

regr.fit(A.T[filtered_indices_cog],cog_outcomes[filtered_indices_cog])

new_y = regr.predict(A.T[filtered_indices_cog])

from sklearn.metrics import r2_score

print(np.corrcoef(new_y, cog_outcomes[filtered_indices_cog]))

print(np.corrcoef(np.array(first_scan_age_list)[:90][filtered_indices_cog], cog_outcomes[filtered_indices_cog]))






A = np.vstack([np.array(first_scan_age_list)[:90], np.array(X)])

# from sklearn.linear_model import linregress

#slope, intercept, r, p, se =linregress(A.T,cog_outcomes)

from sklearn import linear_model

regr = linear_model.LinearRegression()

regr.fit(A.T[filtered_indices_qchat], qchat_outcomes[filtered_indices_qchat])

new_y = regr.predict(A.T[filtered_indices_qchat])

from sklearn.metrics import r2_score

print(np.corrcoef(new_y, qchat_outcomes[filtered_indices_qchat]))

print(np.corrcoef(np.array(first_scan_age_list)[:90][filtered_indices_qchat], qchat_outcomes[filtered_indices_qchat]))



# #         new_age = torch.Tensor([num]).to(device)
# #         if len(new_age.shape)==1 :
# #             new_age.unsqueeze(0)
# #         # new_age_v = torch_age_to_cardinal(new_age).to(device)
# #         new_age_v = new_age.to(device)
        
# #         difference = new_age_v - true_age_1_v

# #         imgen = model(im1, difference)
        
# #         im1_np = im1.reshape(bs,-1,len(mode)).detach().cpu().numpy()
# #         im1 = imgen.squeeze(0)
# #         true_age_1_v = new_age_v
#         imgen = imgen.reshape(bs,-1,len(mode)).detach().cpu().numpy()
         
        


# max_age_similarities = np.array(max_im_similarities) + 32
# diff_age_similarties = second_scan_age_list - max_age_similarities

# import matplotlib.pyplot as plt
# fig = plt.figure()
# # plt.scatter(np.array(second_scan_age_list)-np.array(birth_ages_list), diff_age_similarties)
# # plt.scatter(np.array(birth_ages_list), diff_age_similarties)
# # plt.scatter(np.array(birth_ages_list), np.abs(diff_age_similarties))
# plt.scatter(np.array(second_scan_age_list)-np.array(birth_ages_list), np.abs(diff_age_similarties))
# plt.axhline(y=0, c='orange')
# plt.show()




# L = list(np.arange(32,45))

# import matplotlib.pyplot as plt

# ax = plt.subplot()

# # Define tick label
# # Display graph

# # for i in range(len(im_similarities_list)):
# for i in range(3):

#     plt.plot(L,im_similarities_list[i])
#     plt.axvline(second_scan_age_list[i])
# plt.show()

# from collections import Counter

# C = Counter(np.round(diff_age_similarties))