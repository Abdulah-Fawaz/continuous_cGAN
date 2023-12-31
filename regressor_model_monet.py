#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 15:27:08 2022

@author: fa19
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 01:56:38 2021

@author: fa19
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 15:11:43 2020

@author: fa19
"""

import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np


import torch_geometric
import torch_scatter
import torch_geometric.nn as gnn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import degree

import numpy as np


import os

hex_6 = torch.LongTensor(np.load('data/hexagons_6.npy'))
hex_5 = torch.LongTensor(np.load('data/hexagons_5.npy'))
hex_4 = torch.LongTensor(np.load('data/hexagons_4.npy'))
hex_3 = torch.LongTensor(np.load('data/hexagons_3.npy'))
hex_2 = torch.LongTensor(np.load('data/hexagons_2.npy'))
hex_1 = torch.LongTensor(np.load('data/hexagons_1.npy'))



reverse_hex_6 = np.load('data/reverse_hex_6.npy')
reverse_hex_5 = np.load('data/reverse_hex_5.npy')
reverse_hex_4 = np.load('data/reverse_hex_4.npy')
reverse_hex_3 = np.load('data/reverse_hex_3.npy')
reverse_hex_2 = np.load('data/reverse_hex_2.npy')
reverse_hex_1 = np.load('data/reverse_hex_1.npy')



edge_index_6 = torch.LongTensor(np.load('data/edge_index_6.npy'))
edge_index_5 = torch.LongTensor(np.load('data/edge_index_5.npy'))
edge_index_4 = torch.LongTensor(np.load('data/edge_index_4.npy'))
edge_index_3 = torch.LongTensor(np.load('data/edge_index_3.npy'))
edge_index_2 = torch.LongTensor(np.load('data/edge_index_2.npy'))
edge_index_1 = torch.LongTensor(np.load('data/edge_index_1.npy'))

polar_coords = torch.Tensor(np.load('data/ico_6_polar.npy'))

cartesian_cords = torch.Tensor(np.load('data/ico_6_cartesian.npy'))



pseudo_6 = torch.Tensor(np.load('data/relative_coords_polar_6.npy'))
pseudo_5 = torch.Tensor(np.load('data/relative_coords_polar_5.npy'))
pseudo_4 = torch.Tensor(np.load('data/relative_coords_polar_4.npy'))
pseudo_3 = torch.Tensor(np.load('data/relative_coords_polar_3.npy'))
pseudo_2 = torch.Tensor(np.load('data/relative_coords_polar_2.npy'))




#############################################################################################
       
hexes = [hex_6, hex_5, hex_4, hex_3, hex_2, hex_1]
edges_list = [edge_index_6, edge_index_5, edge_index_4, edge_index_3, edge_index_2, edge_index_1]

reverse_hexes = [reverse_hex_6, reverse_hex_5, reverse_hex_4, reverse_hex_3, reverse_hex_2, reverse_hex_1]



def chebconv(inchans, outchans, K = 3):
    return gnn.ChebConv(inchans, outchans, K)

def gcnconv(inchans, outchans):
    return gnn.GCNConv(inchans, outchans)

def gmmconv(inchans, outchans, kernel_size=3):
    return gnn.GMMConv(inchans, outchans, dim=2, kernel_size=kernel_size)

def graphconv(inchans, outchans):
    return gnn.GraphConv(inchans, outchans)

class hex_pooling_2(nn.Module):
    def __init__(self, ico_level, device):
        super(hex_pooling_2, self).__init__()
        self.hex = hexes[ico_level].to(device)
    
    def forward(self, x):
        x = x.reshape(len(self.hex), -1)[self.hex]
        L = int((len(x)+6)/4)
        x = torch.max(x, dim = 1)
        x , indices = x[0][:L], torch.gather(self.hex[:L], 1,x[1][:L])
        return x, indices



def expand_edge_index(start_edge_index, amount_to_expand):
    if amount_to_expand == 0:
        return start_edge_index
    else:
        new_edge_index = start_edge_index
#        step_size = 1+ torch.max(new_edge_index).item()
        for i in range(amount_to_expand):
            step_size = 1+ torch.max(new_edge_index).item()
            new_edge_index=torch.cat([new_edge_index, start_edge_index+step_size], dim=1)
            
        return new_edge_index

def expand_pseudo_index(start_pseudo_index, amount_to_expand):
    if amount_to_expand == 0:
        return start_pseudo_index
    else:
        new_pseudo_index = start_pseudo_index

        for i in range(amount_to_expand):
            
            new_pseudo_index=torch.cat([new_pseudo_index, start_pseudo_index], dim=0)
            
        return new_pseudo_index

def expand_hex(start_hex, amount_to_expand):
    if amount_to_expand == -1:
        return start_hex
    else:
        new_hex = start_hex
        
        for i in range(amount_to_expand):
            step_size = 1+ torch.max(new_hex).item()
            new_hex=torch.cat([new_hex, start_hex+step_size], dim=0)
            
        return new_hex

class hex_pooling_batched(nn.Module):
    def __init__(self, ico_level, amount_to_expand, device):
        super(hex_pooling_2, self).__init__()
        self.hex = hexes[ico_level]
        self.amount_to_expand = amount_to_expand
        self.device = device
    def forward(self, x):
        new_hex = expand_hex(self.hex, self.amount_to_expand).to(self.device)
        x = x.reshape(len(new_hex), -1)[new_hex]
        L = int((len(x)+6)/4)
        x = torch.max(x, dim = 1)
        x , indices = x[0][:L], torch.gather(new_hex[:L], 1,x[1][:L])
        return x, indices


class hex_pooling_batched(nn.Module):
    def __init__(self, ico_level,in_channels,device):
        super(hex_pooling_batched, self).__init__()
        self.hex = hexes[ico_level]
        self.in_channels = in_channels
    def forward(self, x, batch_size):
        x = x.reshape(batch_size, -1, self.in_channels)
        
        x = x[:,self.hex,:]
        
        L = int((x.shape[1]+6)/4)

        x, _ = torch.max(x, dim = 2)

        x = x[:,:L,:]
        x = x.reshape(-1, self.in_channels)
        return x
    
class two_point_interpolate_batched(nn.Module):
    def __init__(self, ico_level,in_channels,device):
        super(two_point_interpolate_batched, self).__init__()
        self.reverse_hex = reverse_hexes[ico_level]
        self.in_channels = in_channels
    def forward(self, x, batch_size):
        x = x.reshape(batch_size, -1, self.in_channels)
        
        x = x[:,self.reverse_hex,:]
        

        x, _ = torch.mean(x, dim = 2)

        x = x.reshape(-1, self.in_channels)
        return x
       
class GraphResidualBlock(nn.Module):
    expansion = 1

    def __init__(self, inchannels, outchannels, conv_style, activation_function, edges_level, downsample=None, device = 'cuda'):
        super(GraphResidualBlock, self).__init__()
        self.conv_style = conv_style
        self.conv1 = conv_style(inchannels, outchannels)
        self.device = device
        #self.bn1 = nn.BatchNorm2d(planes) NO BN FOR NOW
        #self.relu = nn.ReLU(inplace=True)
        
        self.activation_function = activation_function

        self.conv2 = conv_style(outchannels, outchannels) # second one is outchannels becuse exapansion is 1
        
        #self.bn2 = nn.BatchNorm2d(planes)
        
        self.downsample = downsample
        self.edges_level = edges_level
        self.edges = edges_list[self.edges_level].to('cuda')
    def forward(self, x):
        residual = x
        e = self.edges
        out = self.conv1(x, self.edges)

        out = self.activation_function(out)

        out = self.conv2(out, self.edges)
        
        if self.downsample is not None:
            out = hex_pooling(self.edges_level, self.device)(out)

            residual = self.downsample(x, e)
            


        out += residual
        out = self.activation_function(out)

        return out
    

def transform(data):
    row, col = data.edge_index
    deg = degree(col, data.num_nodes)
    data.edge_attr = torch.stack(
        [1 / torch.sqrt(deg[row]), 1 / torch.sqrt(deg[col])], dim=-1)
    return data

def stack_coords(arr, times=1):
    return torch.cat(times*[arr])




class monet_polar_regression(nn.Module):
    def __init__(self, num_features, conv_style=gmmconv,activation_function=nn.ReLU(), in_channels = 4, device='cuda'):
        super(monet_polar_regression, self).__init__()
        self.conv_style = conv_style
        self.device = device
        self.in_channels = 4
        self.conv1 = conv_style(self.in_channels, num_features[0])
        self.conv2 = conv_style(num_features[0], num_features[1])
        self.conv3 = conv_style(num_features[1], num_features[2])
        self.conv4 = conv_style(num_features[2], num_features[3])
        self.pool1 = hex_pooling(0, self.device)
        self.pool2 = hex_pooling(1, self.device)
        self.pool3 = hex_pooling(2, self.device)
        self.pool4 = hex_pooling(3, self.device)
        
        
        self.activation_function = activation_function
        
        self.fc = nn.Linear(num_features[3] * 2, num_features[3])
        self.fc2 = nn.Linear(num_features[3], 1)

        

        #print "block.expansion=",block.expansion
#        self.fc = nn.Linear(512 * block.expansion, num_classes)

#        for m in self.modules():
#            if isinstance(m, nn.Conv2d):
#                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                m.weight.data.normal_(0, math.sqrt(2. / n))
#            elif isinstance(m, nn.BatchNorm2d):
#                m.weight.data.fill_(1)
#                m.bias.data.zero_()


    def forward(self, x, e, batch):

      

        x = self.conv1(x,e, pseudo_6.to(self.device))
        x = self.activation_function(x)
        x = self.pool1(x)


        x = self.conv2(x,edge_index_5.to(self.device), pseudo_5.to(self.device))
        x = self.activation_function(x)
        x = self.pool2(x)
        
        
        x = self.conv3(x,edge_index_4.to(self.device), pseudo_4.to(self.device))
        x = self.activation_function(x)
        x = self.pool3(x)
        
        
        x = self.conv4(x,edge_index_3.to(self.device), pseudo_3.to(self.device))
        x = self.activation_function(x)
        x = self.pool4(x)
        
        
        
        x_max = gnn.global_max_pool(x, batch[:162].to(self.device))
        x_mean = gnn.global_mean_pool(x, batch[:162].to(self.device))
        
        x_c = torch.cat([x_max, x_mean], dim = 1)
        


        x_out = self.fc(x_c)

        x_out = self.activation_function(x_out)
        x_out = self.fc2(x_out)
        
        return x_out.squeeze(1)
    
    

   
    
    
class monet_polar_segmentation(nn.Module):
    def __init__(self, num_features, conv_style=gmmconv,activation_function=nn.ReLU(), in_channels = 4, device='cuda'):
        super(monet_polar_segmentation, self).__init__()
        self.conv_style = conv_style
        self.device = device
        self.in_channels = 4
        self.conv1 = conv_style(self.in_channels, num_features[0])
        self.conv2 = conv_style(num_features[0], num_features[1])
        self.conv3 = conv_style(num_features[1], num_features[2])
        self.conv4 = conv_style(num_features[2], num_features[3])
        self.pool1 = hex_pooling_2(0, self.device)
        self.pool2 = hex_pooling_2(1, self.device)
        self.pool3 = hex_pooling_2(2, self.device)
        self.pool4 = hex_pooling_2(3, self.device)



        

        self.unpool1 = hex_unpooling(self.device)
        self.unpool2 = hex_unpooling(self.device)
        self.unpool3 = hex_unpooling(self.device)
        self.unpool4 = hex_unpooling(self.device)

        self.conv9 = conv_style(num_features[0] + self.in_channels, 21)

        self.conv8 = conv_style(num_features[1]+num_features[0], num_features[0])
        self.conv7 = conv_style(num_features[2]+num_features[1], num_features[1])
        self.conv6 = conv_style(num_features[3] + num_features[2], num_features[2])
        self.conv5 = conv_style(num_features[3], num_features[3])


        self.activation_function = activation_function
        
        self.outac = nn.Softmax(dim=1)

        


    def forward(self, data):

        x = data.x

        e = data.edge_index


        batch = data.batch

        x = self.conv1(x,e, pseudo_6.to(self.device)) #ico6
        x0 = self.activation_function(x)
        x1, i1 = self.pool1(x0)

        #ico5
        x = self.conv2(x1,edge_index_5.to(self.device), pseudo_5.to(self.device))
        x = self.activation_function(x)
        x2, i2 = self.pool2(x)
        #ico4
        
        x = self.conv3(x2,edge_index_4.to(self.device), pseudo_4.to(self.device))
        x = self.activation_function(x)
        x3, i3 = self.pool3(x)
        #ico3
        
        x = self.conv4(x3,edge_index_3.to(self.device), pseudo_3.to(self.device))
        x = self.activation_function(x)
        x4, i4 = self.pool4(x)
        
        #ico2

        x = self.conv5(x4,edge_index_2.to(self.device), pseudo_2.to(self.device))
        x = self.activation_function(x)
        #ico2
        x = self.unpool3(x, i4)
        #ico3
        x = torch.cat([x,x3], dim=1)

        
        x = self.conv6(x,edge_index_3.to(self.device), pseudo_3.to(self.device))
        
        
        x = self.activation_function(x)
        x = self.unpool2(x, i3)
        #ico4
        
        x = torch.cat([x,x2], dim = 1)
        
        x = self.conv7(x,edge_index_4.to(self.device), pseudo_4.to(self.device))
        x = self.activation_function(x)
        x = self.unpool3(x, i2)
        
        #ico5
        
        x = torch.cat([x,x1], dim = 1)
        
        x = self.conv8(x,edge_index_5.to(self.device), pseudo_5.to(self.device))
        x = self.activation_function(x)
        x = self.unpool4(x, i1)
        
        #ico6 
        
        x = torch.cat([x,x0], dim = 1)
        x = self.conv9(x,e, pseudo_6.to(self.device))

        x_out = self.outac(x)
        
        return x_out
        
    
    


class monet_polar_classification(nn.Module):
    def __init__(self, num_features, conv_style=gmmconv,activation_function=nn.ReLU(), in_channels = 9, device='cuda'):
        super(monet_polar_classification, self).__init__()
        self.conv_style = conv_style
        self.device = device
        self.in_channels = 9
        self.conv1 = conv_style(self.in_channels, num_features[0])
        self.conv2 = conv_style(num_features[0], num_features[1])
        self.conv3 = conv_style(num_features[1], num_features[2])
        self.conv4 = conv_style(num_features[2], num_features[3])
        self.pool1 = hex_pooling(0, self.device)
        self.pool2 = hex_pooling(1, self.device)
        self.pool3 = hex_pooling(2, self.device)
        self.pool4 = hex_pooling(3, self.device)
        
        
        self.activation_function = activation_function
        
        self.fc = nn.Linear(num_features[3] * 2, num_features[3])
        self.fc2 = nn.Linear(num_features[3], 2)
        self.outac = nn.LogSoftmax(dim=1)
        

        #print "block.expansion=",block.expansion
#        self.fc = nn.Linear(512 * block.expansion, num_classes)

#        for m in self.modules():
#            if isinstance(m, nn.Conv2d):
#                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                m.weight.data.normal_(0, math.sqrt(2. / n))
#            elif isinstance(m, nn.BatchNorm2d):
#                m.weight.data.fill_(1)
#                m.bias.data.zero_()


    def forward(self, data):

        x = data.x

        e = data.edge_index


        batch=data.batch

        x = self.conv1(x,e, pseudo_6.to(self.device))
        x = self.activation_function(x)
        x = self.pool1(x)


        x = self.conv2(x,edge_index_5.to(self.device), pseudo_5.to(self.device))
        x = self.activation_function(x)
        x = self.pool2(x)
        
        
        x = self.conv3(x,edge_index_4.to(self.device), pseudo_4.to(self.device))
        x = self.activation_function(x)
        x = self.pool3(x)
        
        
        x = self.conv4(x,edge_index_3.to(self.device), pseudo_3.to(self.device))
        x = self.activation_function(x)
        x = self.pool4(x)
        
        
        
        x_max = gnn.global_max_pool(x, batch[:162].to(self.device))
        x_mean = gnn.global_mean_pool(x, batch[:162].to(self.device))
        
        x_c = torch.cat([x_max, x_mean], dim = 1)
        


        x_out = self.fc(x_c)

        x_out = self.activation_function(x_out)
        x_out = self.fc2(x_out)
        x_out = self.outac(x_out)
        return x_out.squeeze(1)
    
class monet_polar_regression_confounded(nn.Module):
    def __init__(self, num_features, conv_style=gmmconv,activation_function=nn.ReLU(), in_channels=2, device='cuda'):
        super(monet_polar_regression_confounded, self).__init__()
        self.conv_style = conv_style
        self.device = device
        self.in_channels = in_channels
        self.conv1 = conv_style(self.in_channels, num_features[0])
        self.conv2 = conv_style(num_features[0], num_features[1])
        self.conv3 = conv_style(num_features[1], num_features[2])
        self.conv4 = conv_style(num_features[2], num_features[3])
        self.pool1 = hex_pooling(0, self.device)
        self.pool2 = hex_pooling(1, self.device)
        self.pool3 = hex_pooling(2, self.device)
        self.pool4 = hex_pooling(3, self.device)
        self.convm = nn.Conv1d(1,4, kernel_size=1)
        self.dropout = nn.Dropout(0.25)
        
        self.activation_function = activation_function
        
        self.fc = nn.Linear((num_features[3] * 2 )+ 4, num_features[3])
        self.fc2 = nn.Linear(num_features[3], 1)
#        self.dropout = nn.Dropout(0.5)
        

        #print "block.expansion=",block.expansion
#        self.fc = nn.Linear(512 * block.expansion, num_classes)

#        for m in self.modules():
#            if isinstance(m, nn.Conv2d):
#                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                m.weight.data.normal_(0, math.sqrt(2. / n))
#            elif isinstance(m, nn.BatchNorm2d):
#                m.weight.data.fill_(1)
#                m.bias.data.zero_()


    def forward(self, data):

        x = data.x

        e = data.edge_index

        m = data.metadata.to(self.device)
        batch=data.batch

        x = self.conv1(x,e, pseudo_6.to(self.device))
        x = self.activation_function(x)
        x = self.pool1(x)


        x = self.conv2(x,edge_index_5.to(self.device), pseudo_5.to(self.device))
        x = self.activation_function(x)
        x = self.pool2(x)
        
        
        x = self.conv3(x,edge_index_4.to(self.device), pseudo_4.to(self.device))
        x = self.activation_function(x)
        x = self.pool3(x)
        
        
        x = self.conv4(x,edge_index_3.to(self.device), pseudo_3.to(self.device))
        x = self.activation_function(x)
        x = self.pool4(x)
        
        
        
        x_max = gnn.global_max_pool(x, batch[:162].to(self.device))
        x_mean = gnn.global_mean_pool(x, batch[:162].to(self.device))
        
        x_c = torch.cat([x_max, x_mean], dim = 1)
        
        m = self.convm(m.unsqueeze(1))
        # m = nn.ReLU()(m)
        m = m.reshape(m.shape[0],-1)

        x_c = torch.cat([x_c, m], dim=1)

        x_out = self.fc(x_c)
        x_out = self.dropout(x_out)
#        x_out = self.dropout(x_out)
        x_out = self.activation_function(x_out)
        x_out = self.fc2(x_out)
        
        return x_out.squeeze(1)






class monet_polar_regression_confounded_batched(nn.Module):
    def __init__(self, num_features, conv_style=gmmconv,activation_function=nn.ReLU(), in_channels=2, device='cuda'):
        super(monet_polar_regression_confounded_batched, self).__init__()
        self.conv_style = conv_style
        self.device = device
        self.in_channels = in_channels
        self.conv1 = conv_style(self.in_channels, num_features[0])
        self.conv2 = conv_style(num_features[0], num_features[1])
        self.conv3 = conv_style(num_features[1], num_features[2])
        self.conv4 = conv_style(num_features[2], num_features[3])
        self.pool1 = hex_pooling_batched(0,num_features[0], self.device)
        self.pool2 = hex_pooling_batched(1,num_features[1], self.device)
        self.pool3 = hex_pooling_batched(2,num_features[2], self.device)
        self.pool4 = hex_pooling_batched(3,num_features[3], self.device)
        self.convm = nn.Conv1d(1,4, kernel_size=1)
        self.dropout = nn.Dropout(0.2)
        
        self.activation_function = activation_function
        
        self.fc = nn.Linear((num_features[3] * 162 )+ 4, num_features[3])
        self.fc2 = nn.Linear(num_features[3], 1)
#        self.dropout = nn.Dropout(0.5)
        

        #print "block.expansion=",block.expansion
#        self.fc = nn.Linear(512 * block.expansion, num_classes)

#        for m in self.modules():
#            if isinstance(m, nn.Conv2d):
#                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                m.weight.data.normal_(0, math.sqrt(2. / n))
#            elif isinstance(m, nn.BatchNorm2d):
#                m.weight.data.fill_(1)
#                m.bias.data.zero_()


    def forward(self, data):

        x = data.x

        e = data.edge_index

        m = data.metadata.to(self.device)
        batch=data.batch
        batch_size = 1+int(torch.max(batch).item())
#        x = self.fc_start(x)
#        x = self.activation_function(x)

        x = self.conv1(x,e, expand_pseudo_index(pseudo_6, batch_size-1).to(self.device))

        x = self.activation_function(x)
        x = self.pool1(x, batch_size)


        x = self.conv2(x,expand_edge_index(edge_index_5, batch_size-1).to(self.device), expand_pseudo_index(pseudo_5, batch_size-1).to(self.device))
        x = self.activation_function(x)
        x = self.pool2(x, batch_size)
        
        
        x = self.conv3(x,expand_edge_index(edge_index_4, batch_size-1).to(self.device), expand_pseudo_index(pseudo_4, batch_size-1).to(self.device))
         
        x = self.activation_function(x)
        x = self.pool3(x, batch_size)
        
        
        x = self.conv4(x,expand_edge_index(edge_index_3, batch_size-1).to(self.device), expand_pseudo_index(pseudo_3, batch_size-1).to(self.device))
         
        x = self.activation_function(x)
        x = self.pool4(x, batch_size)
        
        
               
        m = self.convm(m.unsqueeze(1))
        # m = nn.ReLU()(m)
        m = m.reshape(m.shape[0],-1)

        x = x.reshape(batch_size,-1)

        x_c = torch.cat([x, m], dim=1)

        x_out = self.fc(x_c)
        x_out = self.dropout(x_out)
#        x_out = self.dropout(x_out)
        x_out = self.activation_function(x_out)
        x_out = self.fc2(x_out)
        
        return x_out.squeeze(1)

 
        
class hex_pooling(nn.Module):
    def __init__(self, ico_level, device):
        super(hex_pooling, self).__init__()
        self.hex = hexes[ico_level].to(device)
    
    def forward(self, x):
        x = x.reshape(len(self.hex), -1)[self.hex]
        L = int((len(x)+6)/4)
        x = torch.max(x, dim = 1)[0][: L]
        
        return x


class hex_pooling_2(nn.Module):
    def __init__(self, ico_level, device):
        super(hex_pooling_2, self).__init__()
        self.hex = hexes[ico_level].to(device)
    
    def forward(self, x):
        x = x.reshape(len(self.hex), -1)[self.hex]
        L = int((len(x)+6)/4)
        x = torch.max(x, dim = 1)
        x , indices = x[0][:L], torch.gather(self.hex[:L], 1,x[1][:L])
        return x, indices

class hex_unpooling(nn.Module):
    def __init__(self, device):
        super(hex_unpooling, self).__init__()
        self.device=device
        
    def forward(self, x, indices):        

        other_indices = torch.arange(indices.shape[1])
        
        other_indices = other_indices.repeat(len(indices),1).to(self.device)
        
        
        L = int( (len(x) * 4 ) - 6)

        y = torch.zeros(L, x.shape[1]).to(self.device)
        
        y[indices, other_indices] = x
        
        return y
    
class do_downsample(nn.Module):

    def __init__(self, inchans, outchans, conv_style, ico_level, device):
        super(do_downsample, self).__init__()

        self.conv_style = conv_style
        self.inchans = inchans
        self.outchans = outchans
        self.ico_level = ico_level
        self.device = device
        self.conv1 = conv_style(self.inchans, self.outchans)
        self.pooling_method = hex_pooling(self.ico_level, self.device)
        
        
    def forward(self,x,e):

        
        x = self.conv1(x,e)
        
        x = self.pooling_method(x)





class graphconv_regression_confounded(nn.Module):

    def __init__(self, num_features, conv_style=graphconv,activation_function=nn.ReLU(), in_channels = 4, device='cuda'):
        super(graphconv_regression_confounded, self).__init__()
        self.conv_style = conv_style
        self.device = device
        self.in_channels = 4
        self.conv1 = conv_style(self.in_channels, num_features[0])
        self.conv2 = conv_style(num_features[0], num_features[1])
        self.conv3 = conv_style(num_features[1], num_features[2])
        self.conv4 = conv_style(num_features[2], num_features[3])
   

        self.convm = nn.Conv1d(1,4, kernel_size=1)
        self.activation_function = activation_function
        
        self.fc = nn.Linear((num_features[3] * 2) + 4, num_features[3])
        self.fc2 = nn.Linear(num_features[3], 1)

        

        #print "block.expansion=",block.expansion
#        self.fc = nn.Linear(512 * block.expansion, num_classes)

#        for m in self.modules():
#            if isinstance(m, nn.Conv2d):
#                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                m.weight.data.normal_(0, math.sqrt(2. / n))
#            elif isinstance(m, nn.BatchNorm2d):
#                m.weight.data.fill_(1)
#                m.bias.data.zero_()


    def forward(self, data):
        x = data.x
        e = data.edge_index
        batch = data.batch.to(self.device)
        m = data.metadata.to(self.device)
        
        
        x = self.conv1(x,e)
        x = self.activation_function(x)


        x = self.conv2(x,e)
        x = self.activation_function(x)
       
        
        
        x = self.conv3(x,e)
        x = self.activation_function(x)
        
        
        x = self.conv4(x,e)
        x = self.activation_function(x)
        
        
        x_max = gnn.global_max_pool(x, batch)
        x_mean = gnn.global_mean_pool(x, batch)
        
        x_c = torch.cat([x_max, x_mean], dim = 1)
#        #print "view: ",x.data.shape        
        m = self.convm(m.unsqueeze(1))
        m = nn.ReLU()(m)
        m = m.reshape(m.shape[0],-1)

        x_c = torch.cat([x_c, m], dim=1)
        
        x_out = self.fc(x_c)
        x_out = self.activation_function(x_out)
        x_out = self.fc2(x_out)
        
        return x_out.squeeze(1)
    