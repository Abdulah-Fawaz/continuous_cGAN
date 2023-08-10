#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 20:27:24 2022

@author: fa19
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 11:08:19 2022

@author: fa19
"""
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






use_sunet = False

adjust_sa = True


test_ds = My_dHCP_Data_Graph(input_arr = test_dataset_arr, 
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

test_loader = DataLoader(test_ds, 1, False)

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


sulc_separate = True
if sulc_separate:
        
    birth_age_model1 = GraphNet_Markovian_Generator(1, age_dim = 2,device=device)
    birth_age_model1 = birth_age_model1.to(device)
    birth_age_model1 = torch.load(model_loc + '3cycle_BA_myelin_generator_model_final1')


    
    birth_age_model2 = GraphNet_Markovian_Generator(1, age_dim = 2,device=device)
    birth_age_model2 = birth_age_model2.to(device)
    birth_age_model2 = torch.load(model_loc + '3cycle_BA_sulc_generator_model_final1')

if adjust_sa:
    from regressor_model_monet import monet_polar_regression, graphconv_regression_confounded

    regressor_model_sa = monet_polar_regression([in_channels, 32, 64 , 128, 256])
    regressor_model_location_sa = '/home/fa19/Documents/neurips/regressor_monet_std2'
    regressor_model_sa = torch.load(regressor_model_location_sa)
    
    regressor_model_sa = regressor_model_sa.to(device)
    regressor_model_sa.eval()
final_save_dir = '/home/fa19/Documents/neurips/full_3cycle_basa/'



# regressor_model = GraphNet_Regressor_confounded(in_channels,device='cuda')
# regressor_model = torch.load('/home/fa19/Documents/neurips/regressor_monet_ba_std1')

from regressor_model_monet import monet_polar_regression_confounded_batched, graphconv_regression_confounded
from sphericalunet.model import sphericalunet_regression_confounded


if use_sunet:
    
    regressor_model =  sphericalunet_regression_confounded([32,64,128,256,512], in_channels = 2)
    regressor_model = torch.load('/home/fa19/Documents/neurips/regressor_sunet_ba_std')
 
else:
    
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

every_ba_pred = []

every_ba_true = []
every_apparent_scan_age = []


full_M = []


import torch_geometric

for i, data in enumerate(test_loader):
    
    save_loc = '/home/fa19/Documents/neurips/'
    
    
    
    
    
    
    im1 = data['x'][:,mode].to(device)  #youngim
        
            
                                            
    # im1 =  data['x'][:,mode].to(device) 
        
    bs = data.x.shape[0]//40962
    name = test_dataset_arr[i,0].split('_')[0]
    
    true_age_1 = data['metadata'].to(device) # true scan age 1
    
    # true_age_1_v = torch_age_to_cardinal(true_age_1).to(device)
    true_sa = true_age_1.to(device)
    true_ba =  data['y'].to(device) # true birth age 1
    true_sa_raw = true_sa.item()
    true_ba_raw = true_ba.item()
    
    
    all_losses = []
    all_ba_preds = []
    all_ba_trues = []
    temp_apparent_scan_age = []
    
    temp_M = []
    for sa in np.arange(37, 45,1):
        scan_age = torch.Tensor([sa]).to(device)
        
        difference_sa = scan_age - true_sa
        
        scan_age_raw = scan_age.item()
        new_sa_image = scan_age_model(im1, difference_sa)
        
        # save_as_metric(new_sa_image.detach().cpu(), final_save_dir + f'{subject}_sa={scan_age_raw}_ba={true_ba_raw}')
        if adjust_sa:   
            # d2 = torch_geometric.data.Data(x= new_sa_image.to(device), metadata = data.metadata.to(device), edge_index = data.edge_index.to(device), batch=data.batch.to(device))

            apparent_scan_age = regressor_model_sa(new_sa_image.to(device), data.edge_index.to(device), data.batch.to(device)).detach()
          
        else:
            apparent_scan_age = scan_age

        # print(apparent_scan_age)
        temp_losses = []
        temp_ba_preds = []
        temp_ba_trues = []
        temp_apparent_scan_age.append(apparent_scan_age.item())
        # for ba in np.arange(min(37, torch.round(apparent_scan_age).item()), torch.round(apparent_scan_age).item()):
        for ba in np.arange(min(37, torch.round(scan_age).item()), torch.round(scan_age).item()):
            # print(ba,sa)
            # print(ba, apparent_scan_age)
            birth_age = torch.Tensor([ba]).to(device)
            birth_age_raw = birth_age.item()
            difference_ba = birth_age - scan_age
            # print(birth_age.item())
            D = torch.cat((scan_age.unsqueeze(0), difference_ba.unsqueeze(0)), dim=1)
            
            new_ba_image = birth_age_model(new_sa_image, D)
            if sulc_separate:
                # new_ba_image[:,0] = birth_age_model1(new_sa_image[:,0].unsqueeze(1), D).squeeze(1)

                new_ba_image[:,1] = birth_age_model2(new_sa_image[:,1].unsqueeze(1), D).squeeze(1)
            
            # save_as_metric(new_ba_image.detach().cpu(), final_save_dir + f'{subject}_sa={scan_age_raw}_new_ba={birth_age_raw}')
            data.x = new_ba_image.to(device)
            data.metadata = scan_age.to(device).unsqueeze(1)
            data.batch = torch.zeros(40962).to(device).long()
            # ba_prediction = regressor_model(new_ba_image, scan_age.unsqueeze(1))
            if use_sunet:
                ba_prediction = regressor_model(new_ba_image.to(device).unsqueeze(2), scan_age.unsqueeze(1))
            else:
                
                ba_prediction = regressor_model(data.to(device))
            
            temp_ba_preds.append(ba_prediction.item())
            temp_ba_trues.append(birth_age.item())
            temp_M.append([sa.item(),apparent_scan_age.item(),ba.item(), ba_prediction.item()])
            
        if len(temp_ba_trues)!=0:
            all_ba_preds.append(temp_ba_preds)
            all_ba_trues.append(temp_ba_trues)
            every_apparent_scan_age.append(temp_apparent_scan_age)
        
        
    # print(all_ba_preds)
    every_ba_pred.append(all_ba_preds)
    every_ba_true.append(all_ba_trues)
    print(np.round(np.array(all_ba_preds[-1])))
    print(np.array(all_ba_trues[-1]))
    
    all_losses_final.append(get_losses(all_ba_preds, all_ba_trues))
    full_M.append(temp_M)
    
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


# fig = plt.figure()
# for k in range(len(all_ba_preds)):
#     plt.scatter(np.round(np.array(all_ba_preds[k])), np.array(all_ba_trues[k]))
# plt.xlabel('Predicted Birth Age')
# plt.ylabel('Confounded Birth Age')

# plt.plot(np.arange(m,ma), np.arange(m,ma))
# plt.show()



# INDEX = 76


# idxs = all_shared[INDEX]    
# print(all_shared[INDEX])

# save_loc = '/home/fa19/Documents/neurips/'




idx = 39

data = test_ds.__getitem__(idx)
im1 = data['x'][:,mode].to(device)  #youngim


                                
# im1 =  data['x'][:,mode].to(device) 

bs = data.x.shape[0]//40962
name = test_dataset_arr[idx,0].split('_')[0]

true_age_1 = data['metadata'].to(device) # true scan age 1

# true_age_1_v = torch_age_to_cardinal(true_age_1).to(device)
true_sa = true_age_1.to(device)
true_ba =  data['y'].to(device) # true birth age 1
true_sa_raw = true_sa.item()
true_ba_raw = true_ba.item()


all_losses = []
all_ba_preds = []
all_ba_trues = []
for sa in np.arange(37, 45,1):
    scan_age = torch.Tensor([sa]).to(device)

    difference_sa = scan_age - true_sa
    
    scan_age_raw = scan_age.item()
    new_sa_image = scan_age_model(im1, difference_sa)
    
    # save_as_metric(new_sa_image.detach().cpu(), final_save_dir + f'{subject}_sa={scan_age_raw}_ba={true_ba_raw}')
    
    temp_losses = []
    temp_ba_preds = []
    temp_ba_trues = []
    
    for ba in np.arange(37, scan_age_raw+1):
        birth_age = torch.Tensor([ba]).to(device)
        birth_age_raw = birth_age.item()
        difference_ba = birth_age - scan_age
        
        D = torch.cat((scan_age.unsqueeze(0), difference_ba.unsqueeze(0)), dim=1)
        
        new_ba_image = birth_age_model(new_sa_image, D)

        if sulc_separate:
            # new_ba_image[:,0] = birth_age_model1(new_sa_image[:,0].unsqueeze(1), D).squeeze(1)

            new_ba_image[:,1] = birth_age_model2(new_sa_image[:,1].unsqueeze(1), D).squeeze(1)

        # save_as_metric(new_ba_image.detach().cpu(), final_save_dir + f'{subject}_sa={scan_age_raw}_new_ba={birth_age_raw}')
        
        # ba_prediction = regressor_model(new_ba_image, scan_age.unsqueeze(1))
        data.x = new_ba_image.to(device)
        data.metadata = scan_age.to(device).unsqueeze(1)
        data.batch = torch.zeros(40962).to(device).long()
        # ba_prediction = regressor_model(new_ba_image, scan_age.unsqueeze(1))
        if use_sunet:
            ba_prediction = regressor_model(data.x.unsqueeze(2), scan_age.unsqueeze(1))
        else:
            
            ba_prediction = regressor_model(data.to(device))        
        temp_ba_preds.append(ba_prediction.item())
        temp_ba_trues.append(birth_age_raw)
    if len(temp_ba_trues)!=0:
        all_ba_preds.append(temp_ba_preds)
        all_ba_trues.append(temp_ba_trues)
    all_losses.append(temp_losses)

# print(all_ba_preds)

print(np.array(all_ba_preds[-1]))
print(np.array(all_ba_trues[-1]))

all_losses_final.append(get_losses(all_ba_preds, all_ba_trues))
scan_ages_done = [33,36,39,42]

scan_ages_done = list(np.arange(35,45))
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

for k in range(4,len(all_ba_preds)):
    plt.plot(np.array(all_ba_trues[k]), np.array(all_ba_preds[k]), label=scan_ages_done[k], marker='x')
plt.ylabel('Predicted Birth Age')
plt.xlabel('Confounded Birth Age')
plt.legend()

plt.plot(np.arange(37,42), np.arange(37,42))
plt.show()


sol = np.zeros([len(every_ba_pred),len(every_ba_pred[0])])
preds2 = np.zeros([len(every_ba_pred),len(every_ba_pred[0])])
trues2 = np.zeros([len(every_ba_pred),len(every_ba_pred[0])])







full_M = np.array(full_M) # 76 subjects x 28 x 4

# scan age, apparent scan age, target ba, predicted ba


all_means = []

for scan_age_ in range(38,45):
    means = []
    small_M = full_M[full_M[:,:,0]==scan_age_]
    # small_M = small_M.reshape(76,4,-1)
    for birth_age_ in range(37,scan_age_):
        smaller_M = small_M[small_M[:,2]==birth_age_]
        means.append(np.mean(smaller_M[:,3],axis=0))
    all_means.append(means)
            
    # predicteds = small_M[:,[3],:]
    # targets = small_M[:,[2],:]

scan_ages_done = list(np.arange(37,45))


fig = plt.figure()

for k in range(0,len(all_means)):
    plt.plot(np.array(every_ba_true[0][k]), all_means[k], label=scan_ages_done[k], marker='x')
plt.ylabel('Predicted Birth Age')
plt.xlabel('Confounded Birth Age')
plt.legend()

plt.plot(np.arange(37,42), np.arange(37,42))
plt.show()


#### with real scan age #####


A = np.zeros([full_M.shape[0]*full_M.shape[1],3])
A[:,2] = 1

A[:,:2] = full_M.reshape(-1,4)[:,[0,3]]

B =  full_M.reshape(-1,4)[:,[2]]



fit =np.matmul( np.matmul( np.linalg.inv(np.matmul(A.T , A)) , A.T ) , B)
errors = B -np.matmul( A , fit)
reconstructed = np.matmul(A,fit)
residual = np.linalg.norm(errors)

infer_A = []
for i in np.arange(37,45):
    temp_ = []
    for j in np.arange(37,i+1):
        temp_.append([i,j,1])
    infer_A.append(temp_)
    


inference_M = []
for i in np.arange(37,45):
    for j in np.arange(37,i+1):
        inference_M.append([i,j,1])

inference_M=np.array(inference_M)
recon2 = np.matmul(inference_M, fit)
print(errors)


# import matplotlib.pyplot as plt
# fig = plt.figure()

# plt.scatter( inference_M[:,1],np.array(recon2), label=inference_M[:,1], marker='x')
# plt.ylabel('Predicted Birth Age')
# plt.xlabel('Confounded Birth Age')
# plt.legend()

# plt.plot(np.arange(37,45), np.arange(37,45))
# plt.show()

#### with fake scan age #####


import matplotlib.pyplot as plt
fig = plt.figure()

plt.scatter(reconstructed, B, marker='x', s=15)
plt.ylabel('Predicted Birth Age')
plt.xlabel('Confounded Birth Age')

plt.show()


A = np.zeros([full_M.shape[0]*full_M.shape[1],3])
A[:,2] = 1

A[:,:2] = full_M.reshape(-1,4)[:,[0,3]]

B =  full_M.reshape(-1,4)[:,[2]]
C  =  full_M.reshape(-1,4)[:,[0,1]]
diff =  C[:,0]-C[:,1]
B=B[:,0]-diff


fit =np.matmul( np.matmul( np.linalg.inv(np.matmul(A.T , A)) , A.T ) , B)
errors = B -np.matmul( A , fit)
reconstructed = np.matmul(A,fit)
residual = np.linalg.norm(errors)

infer_A = []
for i in np.arange(37,45):
    temp_ = []
    for j in np.arange(37,i+1):
        temp_.append([i,j,1])
    infer_A.append(temp_)
    


inference_M = []
for i in np.arange(37,45):
    for j in np.arange(37,i+1):
        inference_M.append([i,j,1])

inference_M=np.array(inference_M)
recon2 = np.matmul(inference_M, fit)
print(errors)


# import matplotlib.pyplot as plt
# fig = plt.figure()

# plt.scatter( inference_M[:,1]-diff,np.array(recon2), label=inference_M[:,1], marker='x')
# plt.ylabel('Predicted Birth Age')
# plt.xlabel('Confounded Birth Age')
# plt.legend()

# plt.plot(np.arange(37,45), np.arange(37,45))
# plt.show()



import matplotlib.pyplot as plt
fig = plt.figure()

plt.scatter(reconstructed, B,marker='x', c=A[:,0],s=15)
plt.ylabel('Predicted Birth Age')
plt.xlabel('Confounded Birth Age')


plt.plot(np.arange(34,45), np.arange(34,45))
plt.show()

# analysing_index = 0

# mean_collected = []
# std_collected = []

# trues = every_ba_true[0]


# L = np.arange(37,44)

# M = np.zeros([7, 7, len(every_ba_pred)])


# for i, subject in enumerate(every_ba_pred):
    
#     for j, row in enumerate(subject): 
        
#         scan_age_max = max(every_ba_true[i][j])
#         for k,birth_age_prediction in enumerate(row):
            
#             birth_age_true = every_ba_true[i][j][k]
            
        
#             scan_age_position = scan_age_max - 37
#             birth_age_position = birth_age_true - 37
#             M[int(scan_age_position), int(birth_age_position),i] = birth_age_prediction
        


# M_mean = np.mean(M,axis=2)

# T = np.zeros([7,7])


# for i in range(7):
#     for j in range(7):
#         row = M[i,j,:]
#         row_without_zeros = [item for item in row if item !=0]
#         mean = np.mean(row_without_zeros)
        
#         T[i,j] = mean


# L = np.arange(37,44)



# fig = plt.figure()


# counter = 0


# for row in T.T:
#     plt.plot(L[:7-counter], row[counter:])
#     counter+=1
# plt.show()
    
    
    
    

# mean_collected = []
# std_collected = []

# trues = every_ba_true[0]


# L = np.arange(37,44)




        
        

# for analysing_index in range(0,len(every_ba_pred[0])):
#     collated_results = []
  
#     for i,arr in enumerate(every_ba_pred[analysing_index]):
#         collated_results.append(arr[i])
        
#     collated_results = np.array(collated_results)
    
#     mean_collected.append(np.mean(collated_results, axis=0))
    
#     std_collected.append(np.std(collated_results, axis=0))
            

        
# fig = plt.figure()
# L = np.arange(37,45)
# for k in range(len(every_ba_pred[0])):
#     plt.errorbar(trues[k],mean_collected[k], label = str(L[k]))
# plt.plot(L,L, linestyle='dotted', label='ground truth')
# plt.legend(title='Target Scan Age')
# plt.show()



# mean_collected = []
# std_collected = []



# L = np.arange(37,44)

          


trues = every_ba_true[0]


all_results = []


for i in range(len(every_ba_pred[0])):
    temp = []
    for j in range(i+1):
        collated_results = []

        for subject in range(len(every_ba_pred)):
        
            collated_results.append(every_ba_pred[subject][i][j]+max(every_ba_true[0][i])-every_apparent_scan_age[subject][i])
        temp.append(np.mean(collated_results))
    
    all_results.append(temp)






fig = plt.figure()
L = np.arange(37,45)
for k in range(len(every_ba_pred[0])):
    plt.errorbar(trues[k],all_results[k], label = str(L[k]))
plt.plot(L,L, linestyle='dotted', label='ground truth')
plt.legend(title='Target Scan Age')
plt.show()




all_results = []


for i in range(len(every_ba_pred[0])):
    temp = []
    for j in range(i+1):
        collated_results = []

        for subject in range(len(every_ba_pred)):
        
            collated_results.append(every_ba_pred[subject][i][j])#+max(every_ba_true[0][i])-every_apparent_scan_age[subject][i])
        temp.append(np.mean(collated_results))
    
    all_results.append(temp)






fig = plt.figure()
L = np.arange(37,45)
for k in range(len(every_ba_pred[0])):
    plt.errorbar(trues[k],all_results[k], label = str(L[k]))
plt.plot(L,L, linestyle='dotted', label='ground truth')
plt.legend(title='Target Scan Age')
plt.show()



# sa.item(),apparent_scan_age.item(),ba.item(), ba_prediction.item()

# scan age, apparent scan age, target ba, predicted ba


full_M = np.array(full_M)

A = np.zeros([len(full_M),3])
A[:,:,2] = 1

A[:,:,:1] = full_M[:,[]]



# for analysing_index in range(0,len(every_ba_pred[0])):
  
#     for i,arr in enumerate(every_ba_pred[analysing_index]):
#         for k,row in enumerate(arr):
#             collated_results = []
            
            
#             if type(row)==float:
#                 s = row
#                 m = s
#             else:
#                 s = row[k]
#                 m = max(row)
#             collated_results.append(s + every_apparent_scan_age[analysing_index][i] - m)
        
#         collated_results = np.array(collated_results)
        
#         mean_collected.append(np.mean(collated_results, axis=0))
        
#         std_collected.append(np.std(collated_results, axis=0))
                

        