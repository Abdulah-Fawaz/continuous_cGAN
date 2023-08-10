#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 11:45:48 2022

@author: fa19
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 14:56:32 2022

@author: fa19
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 22:58:21 2020

@author: fa19
"""



import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np
import torch_geometric
import torch_scatter
import torch_geometric.nn as gnn

from torch_geometric.data import Data

#from utils import *
from layers import *
from python_scripts_for_filters_orders.matlab_equivalent_functions import *


#from torch_geometric.utils import degree



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
reverse_hexes = [reverse_hex_6, reverse_hex_5, reverse_hex_4, reverse_hex_3, reverse_hex_2, reverse_hex_1]

edges_list = [edge_index_6, edge_index_5, edge_index_4, edge_index_3, edge_index_2, edge_index_1]

upsample_6 = torch.LongTensor(np.load('data/upsample_to_ico6.npy'))
upsample_5 = torch.LongTensor(np.load('data/upsample_to_ico5.npy'))
upsample_4 = torch.LongTensor(np.load('data/upsample_to_ico4.npy'))
upsample_3 = torch.LongTensor(np.load('data/upsample_to_ico3.npy'))
upsample_2 = torch.LongTensor(np.load('data/upsample_to_ico2.npy'))
upsample_1 = torch.LongTensor(np.load('data/upsample_to_ico1.npy'))

#############################################################################################

upsamples = [upsample_6, upsample_5, upsample_4, upsample_3, upsample_2, upsample_1]

def chebconv(inchans, outchans, K = 3):
    return gnn.ChebConv(inchans, outchans, K, normalization='sym')

def graphconv(inchans, outchans):
    return gnn.GraphConv(inchans, outchans)

def gatconv(inchans, outchans):
    return gnn.GATConv(inchans, outchans)

def gcnconv(inchans, outchans):
    return gnn.GCNConv(inchans, outchans)
def gmmconv(inchans, outchans, kernel_size=5):
    return gnn.GMMConv(inchans, outchans, dim=2, kernel_size=kernel_size)

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

# class hex_pooling_batched(nn.Module):
#     def __init__(self, ico_level, amount_to_expand, device):
#         super(hex_pooling_2, self).__init__()
#         self.hex = hexes[ico_level]
    
#     def forward(self, x):
#         new_hex = expand_hex(self.hex, amount_to_expand).to(device)
#         x = x.reshape(len(new_hex), -1)[new_hex]
#         L = int((len(x)+6)/4)
#         x = torch.max(x, dim = 1)
#         x , indices = x[0][:L], torch.gather(new_hex[:L], 1,x[1][:L])
#         return x, indices

class hex_upsample(nn.Module):
    """
    NB ico level 0 goes to 40962
    ico level 1 goes to 10242 
    ico level 2 to 2562
    ico level 3 to 642
    ico level 4 to 162
    ico level 5 to 42

    """
    
    
    def __init__(self,ico_level, device='cpu'):
        super(hex_upsample, self).__init__()
        self.upsample = upsamples[ico_level].to(device)
        self.hex = hexes[ico_level].to(device)
        self.device = device
    def forward(self, x, batch_size):
        x = x.reshape(batch_size,-1,x.size(1))
        
        limit = int(x.shape[1])
        new_x = torch.zeros(batch_size, self.hex.shape[0],x.shape[2]).to(self.device)
        new_x[:,:limit,:] = x
        new_x[:,limit:,:] = torch.mean(x[:,self.upsample,:],dim=2)
        
        return new_x.reshape(-1, x.size(2))
    
class hex_pooling_batched(nn.Module):
    """
    
    NB ico level 0 goes to 10242 size
    ico level 1 to 2562
    ico level 2 to 642
    ico level 3 to 162
    ico level 4 to 42
    ico level 5 to 12
    
    """
    
    def __init__(self, ico_level,in_channels):
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
       


class ResBlock_gmmconv_identity(nn.Module):

    def __init__(self, inchans, conv_style=gmmconv,activation_function=nn.LeakyReLU()):
        super(ResBlock_gmmconv_identity, self).__init__()
        self.conv1 = conv_style(inchans, inchans)
        self.conv2 = conv_style(inchans, inchans)
        
        self.norm1 = gnn.InstanceNorm(inchans)
        self.norm2 = gnn.InstanceNorm(inchans)

        self.activation = activation_function

    def forward(self,x,e,p):

        x1 = self.conv1(x,e,p)
        x1 = self.norm1(x1)
        x1 = self.activation(x1)

        x1 = self.conv2(x1,e,p)
        x1 = self.norm2(x1)        

        
        out = x1+x
        out = self.activation(out)

        return out
        
class ResBlock_gmmconv_identity_no_outac(nn.Module):

    def __init__(self, inchans, conv_style=gmmconv,activation_function=nn.LeakyReLU()):
        super(ResBlock_gmmconv_identity_no_outac, self).__init__()
        self.conv1 = conv_style(inchans, inchans)
        self.conv2 = conv_style(inchans, inchans)
        
        self.norm1 = gnn.InstanceNorm(inchans)
        self.norm2 = gnn.InstanceNorm(inchans)

        self.activation = activation_function

    def forward(self,x,e,p):

        x1 = self.conv1(x,e,p)
        x1 = self.norm1(x1)
        x1 = self.activation(x1)

        x1 = self.conv2(x1,e,p)
        x1 = self.norm2(x1)        

        
        out = x1+x

        return out
class ResBlock_gmmconv_basic(nn.Module):

    def __init__(self, inchans, outchans, conv_style=gmmconv,activation_function=nn.ReLU()):
        super(ResBlock_gmmconv_basic, self).__init__()
        self.conv1 = conv_style(inchans, inchans)
        self.conv2 = conv_style(inchans, outchans)
        self.conv3 = conv_style(inchans, outchans)

        self.norm1 = gnn.InstanceNorm(inchans)
        self.norm2 = gnn.InstanceNorm(outchans)
        self.norm3 = gnn.InstanceNorm(outchans)

        self.activation = activation_function

    def forward(self,x,e,p):

        x1 = self.conv1(x,e,p)
        x1 = self.norm1(x1)
        x1 = self.activation(x1)

        x1 = self.conv2(x1,e,p)
        x1 = self.norm2(x1)        
        x2 = self.conv3(x,e,p)
        x2 = self.norm3(x2)
        
        out = x1+x2
        out = self.activation(out)

        return out
        

class ResBlock_gmmconv_downblock(nn.Module):

    def __init__(self, inchans, outchans, ico_level, conv_style=gmmconv,activation_function=nn.ReLU()):
        super(ResBlock_gmmconv_downblock, self).__init__()
        self.conv1 = conv_style(inchans, inchans)
        self.conv2 = conv_style(inchans, inchans)
        self.conv3 = conv_style(inchans, outchans)
        self.conv4 = conv_style(inchans, outchans)

        self.pool = hex_pooling_batched(ico_level, inchans)
        self.pool2 = hex_pooling_batched(ico_level, outchans)

        self.norm1 = gnn.InstanceNorm(inchans)
        self.norm2 = gnn.InstanceNorm(inchans)
        self.norm3 = gnn.InstanceNorm(outchans)
        self.norm4 = gnn.InstanceNorm(outchans)
        
        self.activation = activation_function


    def forward(self,x,e,p, e2, p2, batch_size):

        x1 = self.conv1(x,e,p)        
        x1 = self.norm1(x1)
        x1 = self.activation(x1)
        
        x1 = self.conv2(x,e,p)        
        x1 = self.norm2(x1)
        x1 = self.activation(x1)     
        
        x1 = self.pool(x1, batch_size)
        
        x1 = self.conv3(x1, e2,p2)
        x1 = self.norm3(x1)
        
        
        
        x2 = self.conv4(x, e, p)
        x2 = self.norm4(x2)
        x2 = self.pool2(x2, batch_size)
        
        
        out = x1 + x2

        out = self.activation(out)
        
        
        return out
        


class ResBlock_gmmconv_upblock_upsample(nn.Module):

    def __init__(self, inchans, outchans, ico_level, device, conv_style=gmmconv,activation_function=nn.ReLU()):
        super(ResBlock_gmmconv_upblock_upsample, self).__init__()
        self.conv1 = conv_style(inchans, inchans)
        self.conv2 = conv_style(inchans, inchans)
        self.conv3 = conv_style(inchans, outchans)
        self.conv4 = conv_style(inchans, outchans)

        self.unpool = hex_upsample(ico_level, device)
        self.unpool2 = hex_upsample(ico_level, device)

        self.norm1 = gnn.InstanceNorm(inchans)
        self.norm2 = gnn.InstanceNorm(inchans)
        self.norm3 = gnn.InstanceNorm(outchans)
        self.norm4 = gnn.InstanceNorm(outchans)
        
        self.activation = activation_function
        self.device = device

    def forward(self,x,e,p, e2, p2, batch_size):

        x1 = self.conv1(x,e,p)        
        x1 = self.norm1(x1)
        x1 = self.activation(x1)
        
        x1 = self.conv2(x,e,p)        
        x1 = self.norm2(x1)
        x1 = self.activation(x1)     
        
        x1 = self.unpool(x1, batch_size)
        
        x1 = self.conv3(x1, e2,p2)
        x1 = self.norm3(x1)
        
        
        
        x2 = self.conv4(x, e, p)
        x2 = self.norm4(x2)
        x2 = self.unpool2(x2,  batch_size)
        
        
        out = x1 + x2

        out = self.activation(out)
        
        
        return out





####################################################################################

class VAE_Encoder(nn.Module):
    def __init__(self, num_features, conv_style=gmmconv,in_channels = 4, latent_dim = 100, device='cuda'):
        super(VAE_Encoder, self).__init__()

        self.residentity = ResBlock_gmmconv_identity(in_channels,conv_style=conv_style)
        
        self.resbasic1 = ResBlock_gmmconv_basic(in_channels,num_features[0], conv_style=conv_style)
        
        self.resdown1 = ResBlock_gmmconv_downblock(num_features[0], num_features[1], 0,conv_style=conv_style)
        self.resdown2 = ResBlock_gmmconv_downblock(num_features[1], num_features[2], 1,conv_style=conv_style)
        self.resdown3 = ResBlock_gmmconv_downblock(num_features[2], num_features[3], 2,conv_style=conv_style)
        self.resdown4 = ResBlock_gmmconv_downblock(num_features[3], num_features[4], 3,conv_style=conv_style)
        
        self.fc1 = nn.Linear(num_features[4] * 162, 2048)
        self.normfc1 = gnn.InstanceNorm(2048)
        
        self.fc_mean = nn.Linear(2048, latent_dim)
        self.fc_logvar = nn.Linear(2048, latent_dim)
        
        
        self.device = device
        

    def forward(self, data):

        x = data.x
           
        e = data.edge_index
                   
        batch=data.batch.to(self.device)
        batch_size = 1+int(torch.max(batch).item())
        
        
        p6 = expand_pseudo_index(pseudo_6, batch_size-1).to(self.device)
        p5 = expand_pseudo_index(pseudo_5, batch_size-1).to(self.device)
        p4 = expand_pseudo_index(pseudo_4, batch_size-1).to(self.device)
        p3 = expand_pseudo_index(pseudo_3, batch_size-1).to(self.device)
        p2 = expand_pseudo_index(pseudo_2, batch_size-1).to(self.device)
        
        
        
        e6 = expand_edge_index(edge_index_6, batch_size-1).to(self.device)
        e5 = expand_edge_index(edge_index_5, batch_size-1).to(self.device)
        e4 = expand_edge_index(edge_index_4, batch_size-1).to(self.device)
        e3 = expand_edge_index(edge_index_3, batch_size-1).to(self.device)
        e2 = expand_edge_index(edge_index_2, batch_size-1).to(self.device)
        
        out = self.residentity(x, e6, p6)
        
        out = self.resbasic1(out, e6, p6)
        
        out = self.resdown1(out, e6, p6, e5, p5, batch_size)
        out = self.resdown2(out, e5, p5, e4, p4, batch_size)
        out = self.resdown3(out, e4, p4, e3, p3, batch_size)
        out = self.resdown4(out, e3, p3, e2, p2, batch_size)

        out = out.view(batch_size, -1)
        out = self.fc1(out)
        out = self.normfc1(out)     
        
        
        mean = self.fc_mean(out)
        
        var = self.fc_logvar(out)
        
        return mean, var


class VAE_Decoder(nn.Module):
    def __init__(self, num_features, conv_style=gmmconv,in_channels = 4, latent_dim = 100, device='cuda'):
        super(VAE_Decoder, self).__init__()

        self.fc1 = nn.Linear(latent_dim, num_features[4] * 162)
        self.normfc1 = gnn.InstanceNorm(num_features[4] * 162)
        self.activation_function1 = nn.LeakyReLU(0.2)
        
        self.resup1 = ResBlock_gmmconv_upblock_upsample(num_features[4], num_features[3], 3,device, conv_style=conv_style)
        self.resup2 = ResBlock_gmmconv_upblock_upsample(num_features[3], num_features[2], 2,device,conv_style=conv_style)
        self.resup3 = ResBlock_gmmconv_upblock_upsample(num_features[2], num_features[1], 1,device,conv_style=conv_style)
        self.resup4 = ResBlock_gmmconv_upblock_upsample(num_features[1], num_features[0], 0,device,conv_style=conv_style)
        
        self.resbasic1 = ResBlock_gmmconv_basic(num_features[0],in_channels, conv_style=conv_style)

        self.residentity = ResBlock_gmmconv_identity_no_outac(in_channels,conv_style=conv_style)
        
        
        self.device = device
        
        self.out_ac = nn.Sigmoid()
    def forward(self, x, batch_size):

        
        p6 = expand_pseudo_index(pseudo_6, batch_size-1).to(self.device)
        p5 = expand_pseudo_index(pseudo_5, batch_size-1).to(self.device)
        p4 = expand_pseudo_index(pseudo_4, batch_size-1).to(self.device)
        p3 = expand_pseudo_index(pseudo_3, batch_size-1).to(self.device)
        p2 = expand_pseudo_index(pseudo_2, batch_size-1).to(self.device)
        
        
        
        e6 = expand_edge_index(edge_index_6, batch_size-1).to(self.device)
        e5 = expand_edge_index(edge_index_5, batch_size-1).to(self.device)
        e4 = expand_edge_index(edge_index_4, batch_size-1).to(self.device)
        e3 = expand_edge_index(edge_index_3, batch_size-1).to(self.device)
        e2 = expand_edge_index(edge_index_2, batch_size-1).to(self.device)
        
        
        out = self.fc1(x)
        out = self.normfc1(out)
        out = self.activation_function1(out)
        out = out.view(162*batch_size, -1)
        

        out = self.resup1(out, e2, p2, e3, p3, batch_size)
        out = self.resup2(out, e3, p3, e4, p4, batch_size)
        out = self.resup3(out, e4, p4, e5, p5, batch_size)
        out = self.resup4(out, e5, p5, e6, p6, batch_size)        
        
        out = self.resbasic1(out, e6, p6)

        x_tilde = self.residentity(out, e6, p6)
        

        return self.out_ac(x_tilde)

        # upconv_top_index_40962, upconv_down_index_40962, upconv_top_index_10242, upconv_down_index_10242,  upconv_top_index_2562, upconv_down_index_2562,  upconv_top_index_642, upconv_down_index_642, upconv_top_index_162, upconv_down_index_162, upconv_top_index_42, upconv_down_index_42, upconv_top_index_12, upconv_down_index_12 = my_mat_Get_2ring_upconv_index()
        
        # self.activation_function = activation_function
        
        # self.fc1 = nn.Linear(num_features[3] * 162, num_features[3])
        # self.fc_encode_mu = nn.Linear(num_features[3]+1 , latent_dim)
        # self.fc_encode_logvar = nn.Linear(num_features[3]+1 , latent_dim)

        # self.fc_decode = nn.Linear(latent_dim+1, num_features[3] * 162)
        

        # self.upconv1 = upconv_layer(num_features[3], num_features[2], upconv_top_index_642, upconv_down_index_642 )
        # self.upconv2 = upconv_layer(num_features[2], num_features[1], upconv_top_index_2562, upconv_down_index_2562 )
        # self.upconv3 = upconv_layer(num_features[1], num_features[0], upconv_top_index_10242, upconv_down_index_10242 )
        # self.upconv4 = upconv_layer(num_features[0], self.in_channels, upconv_top_index_40962, upconv_down_index_40962)


class VAE_Decoder_sunet(nn.Module):
    def __init__(self, num_features, conv_style=gmmconv,in_channels = 4, latent_dim = 100, device='cuda'):
        super(VAE_Decoder_sunet, self).__init__()

        self.fc1 = nn.Linear(latent_dim, num_features[3] * 162)
        self.normfc1 = gnn.InstanceNorm(num_features[3] * 162)
        self.activation_function1 = nn.LeakyReLU(0.2)
        
        upconv_top_index_40962, upconv_down_index_40962, upconv_top_index_10242, upconv_down_index_10242,  upconv_top_index_2562, upconv_down_index_2562,  upconv_top_index_642, upconv_down_index_642, upconv_top_index_162, upconv_down_index_162, upconv_top_index_42, upconv_down_index_42, upconv_top_index_12, upconv_down_index_12 = my_mat_Get_2ring_upconv_index()

        self.upconv1 = upconv_layer(num_features[3], num_features[2], upconv_top_index_642, upconv_down_index_642 )
        self.upconv2 = upconv_layer(num_features[2], num_features[1], upconv_top_index_2562, upconv_down_index_2562 )
        self.upconv3 = upconv_layer(num_features[1], num_features[0], upconv_top_index_10242, upconv_down_index_10242 )
        self.upconv4 = upconv_layer(num_features[0], in_channels, upconv_top_index_40962, upconv_down_index_40962)
        

        
        self.activation = nn.ReLU()
        self.device = device
        
        self.out_ac = nn.Sigmoid()
        
    def forward(self, x):
        
        
        out = self.fc1(x)
        out = self.activation_function1(out)
        out = out.reshape(out.shape[0],162,-1).permute(1,2,0)

        out = self.upconv1(out)
        out=self.activation(out)

        out = self.upconv2(out)
        out=self.activation(out)

        out = self.upconv3(out)
        out=self.activation(out)

        out = self.upconv4(out)
        
        out = self.out_ac(out)
        
        return out.view(out.shape[0]*out.shape[2], out.shape[1])


class VAE_gmmconv(nn.Module):
    def __init__(self, num_features, conv_style=gmmconv,in_channels = 4, latent_dim = 100, device='cuda'):
        super(VAE_gmmconv, self).__init__()
        self.encoder = VAE_Encoder(num_features, in_channels=in_channels, device=device, latent_dim = latent_dim)
        self.decoder = VAE_Decoder(num_features, in_channels=in_channels, device=device, latent_dim = latent_dim)
        self.device = device
        self.latent_dim = latent_dim
    def forward(self,data):
        x = data.x
           
        e = data.edge_index
                   
        batch=data.batch.to(self.device)
        batch_size = 1+int(torch.max(batch).item())
        

        z_mean,z_logvar=self.encoder(data)
        std = z_logvar.mul(0.5).exp_()
        
        #sampling epsilon from normal distribution
        epsilon=torch.randn(batch_size,self.latent_dim).to(self.device)
        
        z=z_mean+std*epsilon
        
        x_tilda=self.decoder(z, batch_size)
          
        return z_mean,z_logvar,x_tilda
    
    
class VAE_mixed(nn.Module):
    def __init__(self, num_features, conv_style=gmmconv,in_channels = 4, latent_dim = 100, device='cuda'):
        super(VAE_mixed, self).__init__()
        self.encoder = VAE_Encoder(num_features, in_channels=in_channels, device=device, latent_dim = latent_dim)
        self.decoder = VAE_Decoder_sunet(num_features, in_channels=in_channels, device=device, latent_dim = latent_dim)
        self.device = device
        self.latent_dim = latent_dim
    def forward(self,data):
        x = data.x
           
        e = data.edge_index
                   
        batch=data.batch.to(self.device)
        batch_size = 1+int(torch.max(batch).item())
        

        z_mean,z_logvar=self.encoder(data)
        std = z_logvar.mul(0.5).exp_()
        
        #sampling epsilon from normal distribution
        epsilon=torch.randn(batch_size,self.latent_dim).to(self.device)
        
        z=z_mean+std*epsilon
        
        x_tilda=self.decoder(z)
          
        return z_mean,z_logvar,x_tilda

#####################################################################################


#####################################################################################


#######################################################################################









    
# class monet_variational_upconv_official(nn.Module):

#     def __init__(self, num_features, conv_style=gmmconv,activation_function=nn.ReLU(0.2), in_channels = 4, latent_dim = 100, device='cuda'):
#         super(monet_variational_upconv_official, self).__init__()
#         self.conv_style = conv_style
#         self.device = device
#         self.in_channels = in_channels
# #        self.fc_start = nn.Linear(self.in_channels, num_features[0])
# #        self.fc_end = nn.Linear(num_features[0], self.in_channels)
        
#         self.conv1 = conv_style(self.in_channels, num_features[0])
#         self.bn1 = gnn.norm.BatchNorm(num_features[0])
#         self.conv2 = conv_style(num_features[0], num_features[1])
#         self.bn2 = gnn.norm.BatchNorm(num_features[1])

#         self.conv3 = conv_style(num_features[1], num_features[2])
#         self.bn3 = gnn.norm.BatchNorm(num_features[2])

#         self.conv4 = conv_style(num_features[2], num_features[3])
#         self.bn4 = gnn.norm.BatchNorm(num_features[3])

#         self.pool1 = hex_pooling_batched(0,num_features[0], self.device)
#         self.pool2 = hex_pooling_batched(1,num_features[1], self.device)
#         self.pool3 = hex_pooling_batched(2,num_features[2], self.device)
#         self.pool4 = hex_pooling_batched(3,num_features[3], self.device)

# #        upconv_top_index_163842, upconv_down_index_163842, upconv_top_index_40962, upconv_down_index_40962, upconv_top_index_10242, upconv_down_index_10242,  upconv_top_index_2562, upconv_down_index_2562,  upconv_top_index_642, upconv_down_index_642, upconv_top_index_162, upconv_down_index_162 = Get_upconv_index() 
#         upconv_top_index_40962, upconv_down_index_40962, upconv_top_index_10242, upconv_down_index_10242,  upconv_top_index_2562, upconv_down_index_2562,  upconv_top_index_642, upconv_down_index_642, upconv_top_index_162, upconv_down_index_162, upconv_top_index_42, upconv_down_index_42, upconv_top_index_12, upconv_down_index_12 = my_mat_Get_2ring_upconv_index()
        
#         self.activation_function = activation_function
        
#         self.fc1 = nn.Linear(num_features[3] * 162, num_features[3]*16)
#         self.fc2 = nn.Linear(num_features[3] * 16, num_features[3])
#         self.fc3 = nn.Linear(num_features[3], 512)
        
        
#         self.fc_encode_mu = nn.Linear(512+1 , 256)
#         self.fc_encode_mu2 = nn.Linear(256,latent_dim)
#         self.fc_encode_logvar = nn.Linear(512+1 , 256)
#         self.fc_encode_logvar2 = nn.Linear(256,latent_dim)
     
        
        
#         self.fc_decode1 = nn.Linear(latent_dim+1, num_features[3])
#         self.fc_decode2 = nn.Linear(num_features[3], num_features[3] * 162)



#         self.upconv1 = upconv_layer(num_features[3], num_features[2], upconv_top_index_642, upconv_down_index_642 )
#         self.upconv2 = upconv_layer(num_features[2], num_features[1], upconv_top_index_2562, upconv_down_index_2562 )
#         self.upconv3 = upconv_layer(num_features[1], num_features[0], upconv_top_index_10242, upconv_down_index_10242 )
#         self.upconv4 = upconv_layer(num_features[0], self.in_channels, upconv_top_index_40962, upconv_down_index_40962)


#         self.conv8 = conv_style(num_features[1], num_features[1])
#         self.conv7 = conv_style(num_features[2], num_features[2])
# #        self.conv6 = conv_style(num_features[3], num_features[2])
#         self.conv9 = conv_style(num_features[0], num_features[0])
        
# #        self.bn1 = nn.BatchNorm1d(num_features[0],momentum=0.15, track_running_stats=False)
# #        self.bn2 = nn.BatchNorm1d(num_features[1],momentum=0.15, track_running_stats=False)
# #        self.bn3 = nn.BatchNorm1d(num_features[2],momentum=0.15, track_running_stats=False)
# #        self.bn4 = nn.BatchNorm1d(num_features[3],momentum=0.15, track_running_stats=False)
        
        
#         self.out_ac = nn.Sigmoid()

    
#     def encode(self, data):
#         x = data.x

#         e = data.edge_index

#         confound = data.metadata.to(self.device)
        
#         batch=data.batch.to(self.device)
#         batch_size = 1+int(torch.max(batch).item())
# #        x = self.fc_start(x)
# #        x = self.activation_function(x)

#         x = self.conv1(x,e, expand_pseudo_index(pseudo_6, batch_size-1).to(self.device))
#         x = self.bn1(x)
#         x = self.activation_function(x)

#         x = self.pool1(x, batch_size)


#         x = self.conv2(x,expand_edge_index(edge_index_5, batch_size-1).to(self.device), expand_pseudo_index(pseudo_5, batch_size-1).to(self.device))
#         x = self.bn2(x)

#         x = self.activation_function(x)
#         x = self.pool2(x, batch_size)
        

#         x = self.conv3(x,expand_edge_index(edge_index_4, batch_size-1).to(self.device), expand_pseudo_index(pseudo_4, batch_size-1).to(self.device))
#         x = self.bn3(x)

#         x = self.activation_function(x)
#         x  = self.pool3(x, batch_size)
        

#         x = self.conv4(x,expand_edge_index(edge_index_3, batch_size-1).to(self.device), expand_pseudo_index(pseudo_3, batch_size-1).to(self.device))
#         x = self.bn4(x)

#         x = self.activation_function(x)
#         x = self.pool4(x, batch_size)
        

# #        x = self.bn4(x)

# #        x = self.fc1(x)

        
#         x = x.reshape(batch_size, -1)

#         x = self.fc1(x)
#         x = self.activation_function(x)

#         x = self.fc2(x)
#         x = self.activation_function(x)
        
#         x = self.fc3(x)
#         x = self.activation_function(x)
        
        


# #        x_out = self.bn0(x_out)
# #        print(x_f.shape)
# #        print('Decoded shape: ', print(x_f.shape))
#         x = torch.cat([x, confound], dim=1)
#         encoding_mu = self.fc_encode_mu(x)
#         encoding_mu = self.activation_function(encoding_mu)

#         encoding_mu = self.fc_encode_mu2(encoding_mu)
#         encoding_mu = self.activation_function(encoding_mu)

#         encoding_var = self.fc_encode_logvar(x)
#         encoding_var = self.activation_function(encoding_var)

#         encoding_var = self.fc_encode_logvar2(encoding_var)
#         encoding_var = self.activation_function(encoding_var)

        


#         return encoding_mu, encoding_var, batch_size
    
#     def decode(self, confound, encoding ,batch_size):

#         encoding = torch.cat([encoding, confound], dim=1)
#         y = self.fc_decode1(encoding)
#         y = self.activation_function(y)
#         y = self.fc_decode2(y)
# #        print(y.shape)
#         y = y.view(162,-1,batch_size)

#         #ico2
#         y = self.upconv1(y)
#         #ico3

        
#         y = self.activation_function(y)   
#         y = y.view(y.shape[0]*batch_size, -1)

#         y = self.conv7(y,expand_edge_index(edge_index_3, batch_size-1).to(self.device), expand_pseudo_index(pseudo_3, batch_size-1).to(self.device))

#         y = self.activation_function(y)        
#         y = y.view(642,-1,batch_size)
#         y = self.upconv2(y)
#         #ico4


#         y = self.activation_function(y)
#         y = y.view(y.shape[0]*batch_size, -1)
                
#         y = self.conv8(y,edge_index_4.to(self.device), pseudo_4.to(self.device))
#         y = self.activation_function(y) 
        
        
#         y = y.view(2562,-1,batch_size)
        
        
        
#         y = self.upconv3(y)

#         y = y.view(y.shape[0]*batch_size, -1)

#         y = self.activation_function(y)
#         y = self.conv9(y,edge_index_5.to(self.device), pseudo_5.to(self.device))
#         y = self.activation_function(y)  
        
        
#         y = y.view(10242,-1,batch_size)
#         y = self.upconv4(y)

#         y = y.flatten()
        
#         y = self.out_ac(y)
        
#         y = y.view(40962*batch_size,-1)
        
        
#         return y
    
#     def reparametrize(self, mean, var):
#         epsilon = torch.randn_like(var)       # sampling epsilon        
#         z = mean + torch.exp(0.5*var)*epsilon                          # reparameterization trick
#         return z
    
#     def forward(self, data):
#         encoding_mu, encoding_var, batch_size = self.encode(data)
#         encoding = self.reparametrize(encoding_mu, encoding_var)
#         confound = data.metadata.to(self.device)
#         reconstruction = self.decode(confound, encoding, batch_size)
# #        print('reooding_var)
#         return reconstruction, encoding_mu, encoding_var   

# class ResBlock_gmmconv(nn.Module):

#     def __init__(self, channels, conv_style=gmmconv,activation_function=nn.LeakyReLU(), device='cuda'):
#         super(ResBlock_gmmconv, self).__init__()
#         self.conv1 = conv_style(channels, channels)
#         self.conv2 = conv_style(channels, channels)
        
#         self.device = device
#         self.norm1 = gnn.InstanceNorm(channels)
#         self.norm2 = gnn.InstanceNorm(channels)
#         self.activation = activation_function

#     def forward(self,x,e,p):

#         x1 = self.conv1(x,e,p)
#         x1 = self.activation(x1)
#         x1 = self.norm1(x1)
#         x1 = self.conv2(x1,e,p)
#         x1 = self.activation(x1)
#         x1 = self.norm2(x1)        
        
#         out = x1+x
#         return out
        
# class MoNetEncoder(nn.Module):

#     def __init__(self, num_features, conv_style=gmmconv,activation_function=nn.LeakyReLU(), in_channels = 4, latent_dim = 100, device='cuda'):
#         super(MoNetEncoder, self).__init__()
#         self.conv_style = conv_style
#         self.device = device
#         self.in_channels = in_channels
        
#         self.conv1 = conv_style(self.in_channels, num_features[0])
#         self.conv2 = conv_style(num_features[0], num_features[1])
#         self.conv3 = conv_style(num_features[1], num_features[2])
#         self.conv4 = conv_style(num_features[2], num_features[3])
#         self.res1 = ResBlock_gmmconv(num_features[3])
#         self.res2 = ResBlock_gmmconv(num_features[3])
#         self.res3 = ResBlock_gmmconv(num_features[3])
#         self.res4 = ResBlock_gmmconv(num_features[3])

#         self.pool1 = hex_pooling_batched(0,num_features[0], self.device)
#         self.pool2 = hex_pooling_batched(1,num_features[1], self.device)
#         self.pool3 = hex_pooling_batched(2,num_features[2], self.device)
#         self.pool4 = hex_pooling_batched(3,num_features[3], self.device)

# #        upconv_top_index_163842, upconv_down_index_163842, upconv_top_index_40962, upconv_down_index_40962, upconv_top_index_10242, upconv_down_index_10242,  upconv_top_index_2562, upconv_down_index_2562,  upconv_top_index_642, upconv_down_index_642, upconv_top_index_162, upconv_down_index_162 = Get_upconv_index() 
#         upconv_top_index_40962, upconv_down_index_40962, upconv_top_index_10242, upconv_down_index_10242,  upconv_top_index_2562, upconv_down_index_2562,  upconv_top_index_642, upconv_down_index_642, upconv_top_index_162, upconv_down_index_162, upconv_top_index_42, upconv_down_index_42, upconv_top_index_12, upconv_down_index_12 = my_mat_Get_2ring_upconv_index()
        
#         self.activation_function = activation_function
        
        
        
        
#     def forward(data):
#         x = data.x

#         e = data.edge_index

        
#         batch=data.batch.to(self.device)
#         batch_size = 1+int(torch.max(batch).item())


#         x = self.conv1(x,e, expand_pseudo_index(pseudo_6, batch_size-1).to(self.device))
#         x = self.norm1(x)
#         x = self.activation_function(x)

#         x = self.pool1(x, batch_size)


#         x = self.conv2(x,expand_edge_index(edge_index_5, batch_size-1).to(self.device), expand_pseudo_index(pseudo_5, batch_size-1).to(self.device))
#         x = self.norm2(x)

#         x = self.activation_function(x)
#         x = self.pool2(x, batch_size)
        

#         x = self.conv3(x,expand_edge_index(edge_index_4, batch_size-1).to(self.device), expand_pseudo_index(pseudo_4, batch_size-1).to(self.device))
#         x = self.norm3(x)

#         x = self.activation_function(x)
#         x  = self.pool3(x, batch_size)
        

#         x = self.conv4(x,expand_edge_index(edge_index_3, batch_size-1).to(self.device), expand_pseudo_index(pseudo_3, batch_size-1).to(self.device))
#         x = self.norm4(x)
#         x = self.activation_function(x)
#         x = self.pool4(x, batch_size)
        
#         x = self.res1(x,expand_edge_index(edge_index_2, batch_size-1).to(self.device), expand_pseudo_index(pseudo_2, batch_size-1).to(self.device))
#         x = self.res2(x,expand_edge_index(edge_index_2, batch_size-1).to(self.device), expand_pseudo_index(pseudo_2, batch_size-1).to(self.device))
#         x = self.res3(x,expand_edge_index(edge_index_2, batch_size-1).to(self.device), expand_pseudo_index(pseudo_2, batch_size-1).to(self.device))
#         x = self.res4(x,expand_edge_index(edge_index_2, batch_size-1).to(self.device), expand_pseudo_index(pseudo_2, batch_size-1).to(self.device))

# #        x = self.bn4(x)

# #        x = self.fc1(x)

        
#         x = x.reshape(batch_size, -1)

#         x = self.fc1(x)
#         x = self.activation_function(x)

# #        x_out = self.bn0(x_out)
# #        print(x_f.shape)
# #        print('Decoded shape: ', print(x_f.shape))

#         encoding_mu = self.fc_encode_mu(x)
#         encoding_var = self.fc_encode_logvar(x)

#         return encoding_mu, encoding_var, batch_size        
        

        
# class hex_upsample(nn.Module):
#     def __init__(self,ico_level, device):
#         super(hex_upsample, self).__init__()
#         self.upsample = upsamples[ico_level].to(device)
#         self.hex = hexes[ico_level].to(device)
        
#     def forward(self, x,device, batch_size):
#         x = x.reshape(batch_size,-1,x.size(1))
        
#         limit = int(x.shape[1])
#         new_x = torch.zeros(batch_size, self.hex.shape[0],x.shape[2]).to(device)
#         new_x[:,:limit,:] = x
#         new_x[:,limit:,:] = torch.mean(x[:,self.upsample,:],dim=2)
        
#         return new_x.reshape(-1, x.size(2))

# class monet_variational_upsample_batched(nn.Module):

#     def __init__(self, num_features, conv_style=gmmconv,activation_function=nn.LeakyReLU(), in_channels = 4, latent_dim = 100, device='cuda'):
#         super(monet_variational_upsample_batched, self).__init__()
#         self.conv_style = conv_style
#         self.device = device
#         self.in_channels = in_channels
        
#         self.conv1 = conv_style(self.in_channels, num_features[0])
#         self.norm1 = gnn.InstanceNorm(num_features[0])
#         self.conv2 = conv_style(num_features[0], num_features[1])
#         self.norm2 = gnn.InstanceNorm(num_features[1])

#         self.conv3 = conv_style(num_features[1], num_features[2])
#         self.norm3 = gnn.InstanceNorm(num_features[2])

#         self.conv4 = conv_style(num_features[2], num_features[3])
#         self.norm4 = gnn.InstanceNorm(num_features[3])

#         self.res1 = ResBlock_gmmconv(num_features[3])
#         self.res2 = ResBlock_gmmconv(num_features[3])
#         self.res3 = ResBlock_gmmconv(num_features[3])
#         self.res4 = ResBlock_gmmconv(num_features[3])

#         self.pool1 = hex_pooling_batched(0,num_features[0], self.device)
#         self.pool2 = hex_pooling_batched(1,num_features[1], self.device)
#         self.pool3 = hex_pooling_batched(2,num_features[2], self.device)
#         self.pool4 = hex_pooling_batched(3,num_features[3], self.device)

# #        upconv_top_index_163842, upconv_down_index_163842, upconv_top_index_40962, upconv_down_index_40962, upconv_top_index_10242, upconv_down_index_10242,  upconv_top_index_2562, upconv_down_index_2562,  upconv_top_index_642, upconv_down_index_642, upconv_top_index_162, upconv_down_index_162 = Get_upconv_index() 
#         upconv_top_index_40962, upconv_down_index_40962, upconv_top_index_10242, upconv_down_index_10242,  upconv_top_index_2562, upconv_down_index_2562,  upconv_top_index_642, upconv_down_index_642, upconv_top_index_162, upconv_down_index_162, upconv_top_index_42, upconv_down_index_42, upconv_top_index_12, upconv_down_index_12 = my_mat_Get_2ring_upconv_index()
        
#         self.activation_function = activation_function
        
#         self.fc1 = nn.Linear(num_features[3] * 162, num_features[3])
#         self.fc_encode_mu = nn.Linear(num_features[3] , latent_dim)
#         self.fc_encode_logvar = nn.Linear(num_features[3] , latent_dim)

#         self.fc_decode = nn.Linear(latent_dim, num_features[3] * 162)
        
#         self.upsample2 = hex_upsample(4,self.device)
#         self.upsample3 = hex_upsample(3,self.device)
#         self.upsample4 = hex_upsample(2,self.device)
#         self.upsample5 = hex_upsample(1,self.device)
#         self.upsample6 = hex_upsample(0,self.device)


#         self.res5 = ResBlock_gmmconv(num_features[3])
#         self.res6 = ResBlock_gmmconv(num_features[3])
#         self.res7 = ResBlock_gmmconv(num_features[3])
#         self.res8 = ResBlock_gmmconv(num_features[3])
#         self.res9 = ResBlock_gmmconv(num_features[0])

#         self.conv8 = conv_style(num_features[2], num_features[1])
#         self.norm8 = gnn.InstanceNorm(num_features[1])

#         self.conv7 = conv_style(num_features[3], num_features[2])
#         self.norm7 = gnn.InstanceNorm(num_features[2])

# #        self.conv6 = conv_style(num_features[3], num_features[2])
#         self.conv9 = conv_style(num_features[1], num_features[0])
#         self.norm9 = gnn.InstanceNorm(num_features[0])

#         self.conv10 =conv_style(num_features[0], self.in_channels)


        
# #        self.bn1 = nn.BatchNorm1d(num_features[0],momentum=0.15, track_running_stats=False)
# #        self.bn2 = nn.BatchNorm1d(num_features[1],momentum=0.15, track_running_stats=False)
# #        self.bn3 = nn.BatchNorm1d(num_features[2],momentum=0.15, track_running_stats=False)
# #        self.bn4 = nn.BatchNorm1d(num_features[3],momentum=0.15, track_running_stats=False)
        
        
#         self.out_ac = nn.Sigmoid()

    
#     def encode(self, data):
#         x = data.x

#         e = data.edge_index

        
#         batch=data.batch.to(self.device)
#         batch_size = 1+int(torch.max(batch).item())


#         x = self.conv1(x,e, expand_pseudo_index(pseudo_6, batch_size-1).to(self.device))
#         x = self.norm1(x)
#         x = self.activation_function(x)

#         x = self.pool1(x, batch_size)


#         x = self.conv2(x,expand_edge_index(edge_index_5, batch_size-1).to(self.device), expand_pseudo_index(pseudo_5, batch_size-1).to(self.device))
#         x = self.norm2(x)

#         x = self.activation_function(x)
#         x = self.pool2(x, batch_size)
        

#         x = self.conv3(x,expand_edge_index(edge_index_4, batch_size-1).to(self.device), expand_pseudo_index(pseudo_4, batch_size-1).to(self.device))
#         x = self.norm3(x)

#         x = self.activation_function(x)
#         x  = self.pool3(x, batch_size)
        

#         x = self.conv4(x,expand_edge_index(edge_index_3, batch_size-1).to(self.device), expand_pseudo_index(pseudo_3, batch_size-1).to(self.device))
#         x = self.norm4(x)
#         x = self.activation_function(x)
#         x = self.pool4(x, batch_size)
        
#         x = self.res1(x,expand_edge_index(edge_index_2, batch_size-1).to(self.device), expand_pseudo_index(pseudo_2, batch_size-1).to(self.device))
#         x = self.res2(x,expand_edge_index(edge_index_2, batch_size-1).to(self.device), expand_pseudo_index(pseudo_2, batch_size-1).to(self.device))
#         x = self.res3(x,expand_edge_index(edge_index_2, batch_size-1).to(self.device), expand_pseudo_index(pseudo_2, batch_size-1).to(self.device))
#         x = self.res4(x,expand_edge_index(edge_index_2, batch_size-1).to(self.device), expand_pseudo_index(pseudo_2, batch_size-1).to(self.device))

# #        x = self.bn4(x)

# #        x = self.fc1(x)

        
#         x = x.reshape(batch_size, -1)

#         x = self.fc1(x)
#         x = self.activation_function(x)

# #        x_out = self.bn0(x_out)
# #        print(x_f.shape)
# #        print('Decoded shape: ', print(x_f.shape))

#         encoding_mu = self.fc_encode_mu(x)
#         encoding_var = self.fc_encode_logvar(x)

#         return encoding_mu, encoding_var, batch_size
    
#     def decode(self, encoding ,batch_size):

#         y = self.fc_decode(encoding)

# #        print(y.shape)
#         y = y.view(162,-1)


#         y = self.res5(y,expand_edge_index(edge_index_2, batch_size-1).to(self.device), expand_pseudo_index(pseudo_2, batch_size-1).to(self.device))
#         y = self.res6(y,expand_edge_index(edge_index_2, batch_size-1).to(self.device), expand_pseudo_index(pseudo_2, batch_size-1).to(self.device))
#         y = self.res7(y,expand_edge_index(edge_index_2, batch_size-1).to(self.device), expand_pseudo_index(pseudo_2, batch_size-1).to(self.device))
#         y = self.res8(y,expand_edge_index(edge_index_2, batch_size-1).to(self.device), expand_pseudo_index(pseudo_2, batch_size-1).to(self.device))
        
#         # y = y.view(162,-1,batch_size)

#         #ico2
#         y = self.upsample3(y, self.device, batch_size)
#         #ico3

        
#         y = self.activation_function(y)   
#         # y = y.view(y.shape[0]*batch_size, -1)

#         y = self.conv7(y,expand_edge_index(edge_index_3, batch_size-1).to(self.device), expand_pseudo_index(pseudo_3, batch_size-1).to(self.device))
#         y = self.norm7(y)

#         y = self.activation_function(y)        
#         # y = y.view(642,-1,batch_size)
#         y = self.upsample4(y, self.device, batch_size)
#         #ico4


#         y = self.activation_function(y)
#         # y = y.view(y.shape[0]*batch_size, -1)
                
#         y = self.conv8(y,edge_index_4.to(self.device), pseudo_4.to(self.device))
#         y = self.norm8(y)

#         y = self.activation_function(y) 
        
        
#         # y = y.view(2562,-1,batch_size)
        
        
        
#         y = self.upsample5(y, self.device, batch_size)

#         # y = y.view(y.shape[0]*batch_size, -1)

#         y = self.activation_function(y)
#         y = self.conv9(y,edge_index_5.to(self.device), pseudo_5.to(self.device))
#         y = self.norm9(y)

#         y = self.activation_function(y)  
        
        

#         y = self.upsample6(y, self.device, batch_size)
#         y = self.activation_function(y)
#         # y = y.view(y.shape[0]*batch_size, -1)
#         y = self.res9(y,expand_edge_index(edge_index_6, batch_size-1).to(self.device), expand_pseudo_index(pseudo_6, batch_size-1).to(self.device))
        
#         y= self.conv10(y,expand_edge_index(edge_index_6, batch_size-1).to(self.device), expand_pseudo_index(pseudo_6, batch_size-1).to(self.device))
#         y = self.out_ac(y)

        
        
#         y = y.view(40962*batch_size,-1)
        
        
#         return y
    
#     def reparametrize(self, mean, var):
#         mean = mean.reshape(2,mean.shape[1]//2)
#         var = var.reshape(2,var.shape[1]//2)
        
#         std = torch.exp(0.5 * var)
#         epsilon = torch.randn_like(std) 
#         # sampling epsilon        
#         z = mean + std*epsilon           

#         return z.flatten()
    
#     def forward(self, data):
#         encoding_mu, encoding_var, batch_size = self.encode(data)
#         encoding = self.reparametrize(encoding_mu, encoding_var)

#         reconstruction = self.decode( encoding, batch_size)
# #        print('reooding_var)
#         return reconstruction, encoding_mu, encoding_var   
        
# class monet_variational_upconv_batched(nn.Module):

#     def __init__(self, num_features, conv_style=gmmconv,activation_function=nn.LeakyReLU(), in_channels = 4, latent_dim = 100, device='cuda'):
#         super(monet_variational_upconv_batched, self).__init__()
#         self.conv_style = conv_style
#         self.device = device
#         self.in_channels = in_channels
        
#         self.conv1 = conv_style(self.in_channels, num_features[0])
#         self.norm1 = gnn.InstanceNorm(num_features[0])
#         self.conv2 = conv_style(num_features[0], num_features[1])
#         self.norm2 = gnn.InstanceNorm(num_features[1])

#         self.conv3 = conv_style(num_features[1], num_features[2])
#         self.norm3 = gnn.InstanceNorm(num_features[2])

#         self.conv4 = conv_style(num_features[2], num_features[3])
#         self.norm4 = gnn.InstanceNorm(num_features[3])

#         self.res1 = ResBlock_gmmconv(num_features[3])
#         self.res2 = ResBlock_gmmconv(num_features[3])
#         self.res3 = ResBlock_gmmconv(num_features[3])
#         self.res4 = ResBlock_gmmconv(num_features[3])

#         self.pool1 = hex_pooling_batched(0,num_features[0], self.device)
#         self.pool2 = hex_pooling_batched(1,num_features[1], self.device)
#         self.pool3 = hex_pooling_batched(2,num_features[2], self.device)
#         self.pool4 = hex_pooling_batched(3,num_features[3], self.device)

# #        upconv_top_index_163842, upconv_down_index_163842, upconv_top_index_40962, upconv_down_index_40962, upconv_top_index_10242, upconv_down_index_10242,  upconv_top_index_2562, upconv_down_index_2562,  upconv_top_index_642, upconv_down_index_642, upconv_top_index_162, upconv_down_index_162 = Get_upconv_index() 
#         upconv_top_index_40962, upconv_down_index_40962, upconv_top_index_10242, upconv_down_index_10242,  upconv_top_index_2562, upconv_down_index_2562,  upconv_top_index_642, upconv_down_index_642, upconv_top_index_162, upconv_down_index_162, upconv_top_index_42, upconv_down_index_42, upconv_top_index_12, upconv_down_index_12 = my_mat_Get_2ring_upconv_index()
        
#         self.activation_function = activation_function
        
#         self.fc1 = nn.Linear(num_features[3] * 162, num_features[3])
#         self.fc_encode_mu = nn.Linear(num_features[3] , latent_dim)
#         self.fc_encode_logvar = nn.Linear(num_features[3] , latent_dim)

#         self.fc_decode = nn.Linear(latent_dim, num_features[3] * 162)
        

#         self.upconv1 = upconv_layer(num_features[3], num_features[2], upconv_top_index_642, upconv_down_index_642 )
#         self.upconv2 = upconv_layer(num_features[2], num_features[1], upconv_top_index_2562, upconv_down_index_2562 )
#         self.upconv3 = upconv_layer(num_features[1], num_features[0], upconv_top_index_10242, upconv_down_index_10242 )
#         self.upconv4 = upconv_layer(num_features[0], num_features[0], upconv_top_index_40962, upconv_down_index_40962)

#         self.res5 = ResBlock_gmmconv(num_features[3])
#         self.res6 = ResBlock_gmmconv(num_features[3])
#         self.res7 = ResBlock_gmmconv(num_features[3])
#         self.res8 = ResBlock_gmmconv(num_features[3])
#         self.res9 = ResBlock_gmmconv(num_features[0])

#         self.conv8 = conv_style(num_features[1], num_features[1])
#         self.norm8 = gnn.InstanceNorm(num_features[1])

#         self.conv7 = conv_style(num_features[2], num_features[2])
#         self.norm7 = gnn.InstanceNorm(num_features[2])

# #        self.conv6 = conv_style(num_features[3], num_features[2])
#         self.conv9 = conv_style(num_features[0], num_features[0])
#         self.norm9 = gnn.InstanceNorm(num_features[0])

#         self.conv10 =conv_style(num_features[0], self.in_channels)


        
# #        self.bn1 = nn.BatchNorm1d(num_features[0],momentum=0.15, track_running_stats=False)
# #        self.bn2 = nn.BatchNorm1d(num_features[1],momentum=0.15, track_running_stats=False)
# #        self.bn3 = nn.BatchNorm1d(num_features[2],momentum=0.15, track_running_stats=False)
# #        self.bn4 = nn.BatchNorm1d(num_features[3],momentum=0.15, track_running_stats=False)
        
        
#         self.out_ac = nn.Sigmoid()

    
#     def encode(self, data):
#         x = data.x

#         e = data.edge_index

        
#         batch=data.batch.to(self.device)
#         batch_size = 1+int(torch.max(batch).item())


#         x = self.conv1(x,e, expand_pseudo_index(pseudo_6, batch_size-1).to(self.device))
#         x = self.norm1(x)
#         x = self.activation_function(x)

#         x = self.pool1(x, batch_size)


#         x = self.conv2(x,expand_edge_index(edge_index_5, batch_size-1).to(self.device), expand_pseudo_index(pseudo_5, batch_size-1).to(self.device))
#         x = self.norm2(x)

#         x = self.activation_function(x)
#         x = self.pool2(x, batch_size)
        

#         x = self.conv3(x,expand_edge_index(edge_index_4, batch_size-1).to(self.device), expand_pseudo_index(pseudo_4, batch_size-1).to(self.device))
#         x = self.norm3(x)

#         x = self.activation_function(x)
#         x  = self.pool3(x, batch_size)
        

#         x = self.conv4(x,expand_edge_index(edge_index_3, batch_size-1).to(self.device), expand_pseudo_index(pseudo_3, batch_size-1).to(self.device))
#         x = self.norm4(x)
#         x = self.activation_function(x)
#         x = self.pool4(x, batch_size)
        
#         x = self.res1(x,expand_edge_index(edge_index_2, batch_size-1).to(self.device), expand_pseudo_index(pseudo_2, batch_size-1).to(self.device))
#         x = self.res2(x,expand_edge_index(edge_index_2, batch_size-1).to(self.device), expand_pseudo_index(pseudo_2, batch_size-1).to(self.device))
#         x = self.res3(x,expand_edge_index(edge_index_2, batch_size-1).to(self.device), expand_pseudo_index(pseudo_2, batch_size-1).to(self.device))
#         x = self.res4(x,expand_edge_index(edge_index_2, batch_size-1).to(self.device), expand_pseudo_index(pseudo_2, batch_size-1).to(self.device))

# #        x = self.bn4(x)

# #        x = self.fc1(x)

        
#         x = x.reshape(batch_size, -1)

#         x = self.fc1(x)
#         x = self.activation_function(x)

# #        x_out = self.bn0(x_out)
# #        print(x_f.shape)
# #        print('Decoded shape: ', print(x_f.shape))

#         encoding_mu = self.fc_encode_mu(x)
#         encoding_var = self.fc_encode_logvar(x)

#         return encoding_mu, encoding_var, batch_size
    
#     def decode(self, encoding ,batch_size):

#         y = self.fc_decode(encoding)

# #        print(y.shape)
#         y = y.view(162,-1)


#         y = self.res5(y,expand_edge_index(edge_index_2, batch_size-1).to(self.device), expand_pseudo_index(pseudo_2, batch_size-1).to(self.device))
#         y = self.res6(y,expand_edge_index(edge_index_2, batch_size-1).to(self.device), expand_pseudo_index(pseudo_2, batch_size-1).to(self.device))
#         y = self.res7(y,expand_edge_index(edge_index_2, batch_size-1).to(self.device), expand_pseudo_index(pseudo_2, batch_size-1).to(self.device))
#         y = self.res8(y,expand_edge_index(edge_index_2, batch_size-1).to(self.device), expand_pseudo_index(pseudo_2, batch_size-1).to(self.device))
        
#         y = y.view(162,-1,batch_size)

#         #ico2
#         y = self.upconv1(y)
#         #ico3

        
#         y = self.activation_function(y)   
#         y = y.view(y.shape[0]*batch_size, -1)

#         y = self.conv7(y,expand_edge_index(edge_index_3, batch_size-1).to(self.device), expand_pseudo_index(pseudo_3, batch_size-1).to(self.device))
#         y = self.norm7(y)

#         y = self.activation_function(y)        
#         y = y.view(642,-1,batch_size)
#         y = self.upconv2(y)
#         #ico4


#         y = self.activation_function(y)
#         y = y.view(y.shape[0]*batch_size, -1)
                
#         y = self.conv8(y,edge_index_4.to(self.device), pseudo_4.to(self.device))
#         y = self.norm8(y)

#         y = self.activation_function(y) 
        
        
#         y = y.view(2562,-1,batch_size)
        
        
        
#         y = self.upconv3(y)

#         y = y.view(y.shape[0]*batch_size, -1)

#         y = self.activation_function(y)
#         y = self.conv9(y,edge_index_5.to(self.device), pseudo_5.to(self.device))
#         y = self.norm9(y)

#         y = self.activation_function(y)  
        
        
#         y = y.view(10242,-1,batch_size)
#         y = self.upconv4(y)
#         y = self.activation_function(y)
#         y = y.view(y.shape[0]*batch_size, -1)
#         y = self.res9(y,expand_edge_index(edge_index_6, batch_size-1).to(self.device), expand_pseudo_index(pseudo_6, batch_size-1).to(self.device))
        
#         y= self.conv10(y,expand_edge_index(edge_index_6, batch_size-1).to(self.device), expand_pseudo_index(pseudo_6, batch_size-1).to(self.device))
#         y = self.out_ac(y)

        
        
#         y = y.view(40962*batch_size,-1)
        
        
#         return y
    
#     def reparametrize(self, mean, var):
#         mean = mean.reshape(2,mean.shape[1]//2)
#         var = var.reshape(2,var.shape[1]//2)
        
#         std = torch.exp(0.5 * var)
#         epsilon = torch.randn_like(std) 
#         # sampling epsilon        
#         z = mean + std*epsilon           

#         return z.flatten()
    
#     def forward(self, data):
#         encoding_mu, encoding_var, batch_size = self.encode(data)
#         encoding = self.reparametrize(encoding_mu, encoding_var)

#         reconstruction = self.decode( encoding, batch_size)
# #        print('reooding_var)
#         return reconstruction, encoding_mu, encoding_var   

# class Discriminator(nn.Module):

#     def __init__(self, num_features, conv_style=gmmconv,activation_function=nn.ReLU(0.2), in_channels = 4, device='cuda'):
#         super(Discriminator, self).__init__()
#         self.conv_style = conv_style
#         self.device = device
#         self.in_channels = in_channels
        
#         self.conv1 = conv_style(self.in_channels, num_features[0])
#         self.conv2 = conv_style(num_features[0], num_features[1])
#         self.conv3 = conv_style(num_features[1], num_features[2])
#         self.conv4 = conv_style(num_features[2], num_features[3])
#         self.pool1 = hex_pooling_batched(0,num_features[0], self.device)
#         self.pool2 = hex_pooling_batched(1,num_features[1], self.device)
#         self.pool3 = hex_pooling_batched(2,num_features[2], self.device)
#         self.pool4 = hex_pooling_batched(3,num_features[3], self.device)

# #        upconv_top_index_163842, upconv_down_index_163842, upconv_top_index_40962, upconv_down_index_40962, upconv_top_index_10242, upconv_down_index_10242,  upconv_top_index_2562, upconv_down_index_2562,  upconv_top_index_642, upconv_down_index_642, upconv_top_index_162, upconv_down_index_162 = Get_upconv_index() 
#         upconv_top_index_40962, upconv_down_index_40962, upconv_top_index_10242, upconv_down_index_10242,  upconv_top_index_2562, upconv_down_index_2562,  upconv_top_index_642, upconv_down_index_642, upconv_top_index_162, upconv_down_index_162, upconv_top_index_42, upconv_down_index_42, upconv_top_index_12, upconv_down_index_12 = my_mat_Get_2ring_upconv_index()
        
#         self.activation_function = activation_function
        
#         self.fc1 = nn.Linear(num_features[3] * 162, num_features[3])
#         self.fc2 = nn.Linear(num_features[3] , 1)


        
#         self.out_ac = nn.Sigmoid()

    
#     def forward(self, data):
#         x = data.x

#         e = data.edge_index
        
#         batch=data.batch.to(self.device)
#         batch_size = 1+int(torch.max(batch).item())

        

#         x = self.conv1(x,e, expand_pseudo_index(pseudo_6, batch_size-1).to(self.device))

#         x = self.activation_function(x)

#         x = self.pool1(x, batch_size)


#         x = self.conv2(x,expand_edge_index(edge_index_5, batch_size-1).to(self.device), expand_pseudo_index(pseudo_5, batch_size-1).to(self.device))

#         x = self.activation_function(x)
#         x = self.pool2(x, batch_size)
        

#         x = self.conv3(x,expand_edge_index(edge_index_4, batch_size-1).to(self.device), expand_pseudo_index(pseudo_4, batch_size-1).to(self.device))
#         x = self.activation_function(x)
#         x  = self.pool3(x, batch_size)
        

#         x = self.conv4(x,expand_edge_index(edge_index_3, batch_size-1).to(self.device), expand_pseudo_index(pseudo_3, batch_size-1).to(self.device))
#         x = self.activation_function(x)
#         x = self.pool4(x, batch_size)
        

# #        x = self.bn4(x)

# #        x = self.fc1(x)

        
#         x = x.reshape(batch_size, -1)

#         x = self.fc1(x)
#         x = self.activation_function(x)

# #        x_out = self.bn0(x_out)
# #        print(x_f.shape)
# #        print('Decoded shape: ', print(x_f.shape))

#         pred = self.fc2(x)


#         return pred
    






# class monet_variational_upconv(nn.Module):

#     def __init__(self, num_features, conv_style=gmmconv,activation_function=nn.ReLU(0.2), in_channels = 4, latent_dim = 100, device='cuda'):
#         super(monet_variational_upconv, self).__init__()
#         self.conv_style = conv_style
#         self.device = device
#         self.in_channels = in_channels
#         self.fc_start = nn.Linear(self.in_channels, num_features[0])
#         self.fc_end = nn.Linear(num_features[0], self.in_channels)
        
#         self.conv1 = conv_style(num_features[0], num_features[0])
#         self.conv2 = conv_style(num_features[0], num_features[1])
#         self.conv3 = conv_style(num_features[1], num_features[2])
#         self.conv4 = conv_style(num_features[2], num_features[3])
#         self.pool1 = hex_pooling_2(0, self.device)
#         self.pool2 = hex_pooling_2(1, self.device)
#         self.pool3 = hex_pooling_2(2, self.device)
#         self.pool4 = hex_pooling_2(3, self.device)

# #        upconv_top_index_163842, upconv_down_index_163842, upconv_top_index_40962, upconv_down_index_40962, upconv_top_index_10242, upconv_down_index_10242,  upconv_top_index_2562, upconv_down_index_2562,  upconv_top_index_642, upconv_down_index_642, upconv_top_index_162, upconv_down_index_162 = Get_upconv_index() 
#         upconv_top_index_40962, upconv_down_index_40962, upconv_top_index_10242, upconv_down_index_10242,  upconv_top_index_2562, upconv_down_index_2562,  upconv_top_index_642, upconv_down_index_642, upconv_top_index_162, upconv_down_index_162, upconv_top_index_42, upconv_down_index_42, upconv_top_index_12, upconv_down_index_12 = my_mat_Get_2ring_upconv_index()
        
#         self.activation_function = activation_function
        
#         self.fc1 = nn.Linear(num_features[3] * 162, num_features[3])
#         self.fc_encode_mu = nn.Linear(num_features[3] , latent_dim)
#         self.fc_encode_logvar = nn.Linear(num_features[3] , latent_dim)

#         self.fc_decode = nn.Linear(latent_dim, num_features[3] * 162)
        

#         self.upconv1 = upconv_layer(num_features[3], num_features[2], upconv_top_index_642, upconv_down_index_642 )
#         self.upconv2 = upconv_layer(num_features[2], num_features[1], upconv_top_index_2562, upconv_down_index_2562 )
#         self.upconv3 = upconv_layer(num_features[1], num_features[0], upconv_top_index_10242, upconv_down_index_10242 )
#         self.upconv4 = upconv_layer(num_features[0], self.in_channels, upconv_top_index_40962, upconv_down_index_40962)


#         self.conv8 = conv_style(num_features[1], num_features[1])
#         self.conv7 = conv_style(num_features[2], num_features[2])
# #        self.conv6 = conv_style(num_features[3], num_features[2])
#         self.conv9 = conv_style(num_features[0], num_features[0])
        
# #        self.bn1 = nn.BatchNorm1d(num_features[0],momentum=0.15, track_running_stats=False)
# #        self.bn2 = nn.BatchNorm1d(num_features[1],momentum=0.15, track_running_stats=False)
# #        self.bn3 = nn.BatchNorm1d(num_features[2],momentum=0.15, track_running_stats=False)
# #        self.bn4 = nn.BatchNorm1d(num_features[3],momentum=0.15, track_running_stats=False)
        
        
#         self.out_ac = nn.Sigmoid()
        
    
#     def encode(self, data):
#         x = data.x

#         e = data.edge_index


#         batch=data.batch.to(self.device)
#         x = self.fc_start(x)
#         x = self.activation_function(x)

#         x = self.conv1(x,e, pseudo_6.to(self.device))
#         x = self.activation_function(x)
#         x, i1 = self.pool1(x)


#         x = self.conv2(x,edge_index_5.to(self.device), pseudo_5.to(self.device))
#         x = self.activation_function(x)
#         x, i2 = self.pool2(x)
        
        
#         x = self.conv3(x,edge_index_4.to(self.device), pseudo_4.to(self.device))
#         x = self.activation_function(x)
#         x, i3 = self.pool3(x)
        
        
#         x = self.conv4(x,edge_index_3.to(self.device), pseudo_3.to(self.device))
#         x = self.activation_function(x)
#         x, i4 = self.pool4(x)
# #        x = self.bn4(x)

# #        x = self.fc1(x)

        
#         x = x.flatten()
#         x_f = self.fc1(x)
# #        x_out = self.bn0(x_out)

#         encoding_mu = self.fc_encode_mu(x_f)
#         encoding_var = self.fc_encode_logvar(x_f)
#         indices_list = [i1,i2,i3,i4]
#         return encoding_mu, encoding_var, indices_list
    
#     def decode(self, encoding):

#         y = self.fc_decode(encoding)

#         y = y.view(162,-1,1)


#         #ico2
#         y = self.upconv1(y)
#         #ico3


#         y = self.activation_function(y)        
#         y = self.conv7(y.squeeze(2),edge_index_3.to(self.device), pseudo_3.to(self.device)).unsqueeze(2)

#         y = self.activation_function(y)        

#         y = self.upconv2(y)
#         #ico4

        
        
#         y = self.activation_function(y)
        
#         y = self.conv8(y.squeeze(2),edge_index_4.to(self.device), pseudo_4.to(self.device)).unsqueeze(2)
#         y = self.activation_function(y)  
#         y = self.upconv3(y)

#         #ico5
                
#         y = self.activation_function(y)
#         y = self.conv9(y.squeeze(2),edge_index_5.to(self.device), pseudo_5.to(self.device)).unsqueeze(2)
#         y = self.activation_function(y)  
#         y = self.upconv4(y)

#         y = y.flatten()
        
#         y = self.out_ac(y)
        
#         y = y.view(40962,-1)
        
        
#         return y
    
#     def reparametrize(self, mean, var):
#         epsilon = torch.randn_like(var)       # sampling epsilon        
#         z = mean + var*epsilon                          # reparameterization trick
#         return z
    
#     def forward(self, data):
#         encoding_mu, encoding_var, indices = self.encode(data)
#         encoding = self.reparametrize(encoding_mu, encoding_var)
        
#         reconstruction = self.decode(encoding)
# #        print('reooding_var)
#         return reconstruction, encoding_mu, encoding_var   



# class monet_variational_upconv_confounded_channel(nn.Module):

#     def __init__(self, num_features, conv_style=gmmconv,activation_function=nn.LeakyReLU(), in_channels = 4, latent_dim = 100, device='cuda'):
#         super(monet_variational_upconv_confounded_channel, self).__init__()
#         self.conv_style = conv_style
#         self.device = device
#         self.in_channels = in_channels


        
#         self.conv1 = conv_style(self.in_channels + 1, num_features[0])
#         self.bn1 = gnn.norm.InstanceNorm(num_features[0])
#         self.conv2 = conv_style(num_features[0], num_features[1])
#         self.bn2 = gnn.norm.InstanceNorm(num_features[1])

#         self.conv3 = conv_style(num_features[1], num_features[2])
#         self.bn3 = gnn.norm.InstanceNorm(num_features[2])

#         self.conv4 = conv_style(num_features[2], num_features[3])
#         self.bn4 = gnn.norm.InstanceNorm(num_features[3])



#         self.pool1 = hex_pooling_2(0, self.device)
#         self.pool2 = hex_pooling_2(1, self.device)
#         self.pool3 = hex_pooling_2(2, self.device)
#         self.pool4 = hex_pooling_2(3, self.device)

# #        upconv_top_index_163842, upconv_down_index_163842, upconv_top_index_40962, upconv_down_index_40962, upconv_top_index_10242, upconv_down_index_10242,  upconv_top_index_2562, upconv_down_index_2562,  upconv_top_index_642, upconv_down_index_642, upconv_top_index_162, upconv_down_index_162 = Get_upconv_index() 
#         upconv_top_index_40962, upconv_down_index_40962, upconv_top_index_10242, upconv_down_index_10242,  upconv_top_index_2562, upconv_down_index_2562,  upconv_top_index_642, upconv_down_index_642, upconv_top_index_162, upconv_down_index_162, upconv_top_index_42, upconv_down_index_42, upconv_top_index_12, upconv_down_index_12 = my_mat_Get_2ring_upconv_index()
        
#         self.activation_function = activation_function
        
#         self.fc1 = nn.Linear(num_features[3] * 162, num_features[3])
# #        self.bnfc1 =gnn.norm.InstanceNorm(num_features[3])
#         self.fc_encode_mu = nn.Linear(num_features[3], latent_dim)
#         self.fc_encode_logvar = nn.Linear(num_features[3] , latent_dim)

#         self.fc_decode = nn.Linear(latent_dim+1, num_features[3] * 162)
        

#         self.upconv1 = upconv_layer(num_features[3], num_features[2], upconv_top_index_642, upconv_down_index_642 )
#         self.upconv2 = upconv_layer(num_features[2], num_features[1], upconv_top_index_2562, upconv_down_index_2562 )
#         self.upconv3 = upconv_layer(num_features[1], num_features[0], upconv_top_index_10242, upconv_down_index_10242 )
#         self.upconv4 = upconv_layer(num_features[0], self.in_channels, upconv_top_index_40962, upconv_down_index_40962)

# #
# #        self.conv8 = conv_style(num_features[1], num_features[1])
# #        self.conv7 = conv_style(num_features[2], num_features[2])
# ##        self.conv6 = conv_style(num_features[3], num_features[2])
# #        self.conv9 = conv_style(num_features[0], num_features[0])
        
#         self.bn7 = gnn.norm.InstanceNorm(num_features[2])        
#         self.bn8 = gnn.norm.InstanceNorm(num_features[1])   
#         self.bn9 = gnn.norm.InstanceNorm(num_features[0])        
#         self.bn10 = gnn.norm.InstanceNorm(self.in_channels)    
# #        self.bn1 = nn.BatchNorm1d(num_features[0],momentum=0.15, track_running_stats=False)
# #        self.bn2 = nn.BatchNorm1d(num_features[1],momentum=0.15, track_running_stats=False)
# #        self.bn3 = nn.BatchNorm1d(num_features[2],momentum=0.15, track_running_stats=False)
# #        self.bn4 = nn.BatchNorm1d(num_features[3],momentum=0.15, track_running_stats=False)
#         self.final_conv= nn.Conv1d(self.in_channels, self.in_channels, kernel_size=1)
        
        
#         self.out_ac = nn.Sigmoid()
        
    
#     def encode(self, data):
#         x = data.x

#         e = data.edge_index


#         batch=data.batch.to(self.device)
#         confound = data.metadata.to(self.device)
#         x = torch.cat([x, torch.ones(40962,1).to(self.device)*confound], dim=1)
#         x = self.conv1(x,e, pseudo_6.to(self.device))
#         x = self.bn1(x)
#         x = self.activation_function(x)
#         x, i1 = self.pool1(x)


#         x = self.conv2(x,edge_index_5.to(self.device), pseudo_5.to(self.device))
#         x = self.bn2(x)

#         x = self.activation_function(x)
#         x, i2 = self.pool2(x)
        
        
#         x = self.conv3(x,edge_index_4.to(self.device), pseudo_4.to(self.device))
#         x = self.bn3(x)

#         x = self.activation_function(x)
#         x, i3 = self.pool3(x)
        
        
#         x = self.conv4(x,edge_index_3.to(self.device), pseudo_3.to(self.device))
#         x = self.bn4(x)

#         x = self.activation_function(x)
#         x, i4 = self.pool4(x)
# #        x = self.bn4(x)

# #        x = self.fc1(x)

        
#         x = x.flatten()
#         x_f = self.fc1(x)
#         x_f = self.activation_function(x_f)
#         #        x_out = self.bn0(x_out)

#         encoding_mu = self.fc_encode_mu(x_f)
#         encoding_var = self.fc_encode_logvar(x_f)
#         indices_list = [i1,i2,i3,i4]
#         return encoding_mu, encoding_var, indices_list
    
#     def decode(self, confound, encoding):

#         encoding = torch.cat([encoding, confound.squeeze(1)])
#         y = self.fc_decode(encoding)

#         y = y.view(162,-1,1)


#         #ico2
#         y = self.upconv1(y)
#         #ico3
#         y = self.bn7(y)

#         y = self.activation_function(y)        
# #        y = self.conv7(y.squeeze(2),edge_index_3.to(self.device), pseudo_3.to(self.device)).unsqueeze(2)
# ##        y = self.bn7(y)
# #        y = self.activation_function(y)        

#         y = self.upconv2(y)
#         #ico4

#         y = self.bn8(y)
        
#         y = self.activation_function(y)
        
# #        y = self.conv8(y.squeeze(2),edge_index_4.to(self.device), pseudo_4.to(self.device)).unsqueeze(2)
# ##        y = self.bn8(y)
# #
# #        y = self.activation_function(y)  
#         y = self.upconv3(y)
#         y = self.bn9(y)
#         #ico5
                
#         y = self.activation_function(y)
# #        y = self.conv9(y.squeeze(2),edge_index_5.to(self.device), pseudo_5.to(self.device)).unsqueeze(2)
# ##        y = self.bn9(y)
# #
# #        y = self.activation_function(y)  
#         y = self.upconv4(y)
#         y = self.bn10(y)
        
#         y = y.permute(2,1,0)

#         y =self.final_conv(y)

        
#         y = self.out_ac(y)
        
#         y = y.view(40962,-1)
        
        
#         return y
    
#     def reparametrize(self, mean, var):
#         std = torch.exp(0.5 * var)
#         eps = torch.randn_like(std)
#         return eps * std + mean
    
#     def forward(self, data):
#         encoding_mu, encoding_var, indices = self.encode(data)
#         encoding = self.reparametrize(encoding_mu, encoding_var)
#         confound = data.metadata.to(self.device)
#         reconstruction = self.decode(confound, encoding)
# #        print('reooding_var)
#         return reconstruction, encoding_mu, encoding_var   




# class monet_variational_upconv_confounded(nn.Module):

#     def __init__(self, num_features, conv_style=gmmconv,activation_function=nn.LeakyReLU(), in_channels = 4, latent_dim = 100, confound_dim = 19,device='cuda'):
#         super(monet_variational_upconv_confounded, self).__init__()
#         self.conv_style = conv_style
#         self.device = device
#         self.in_channels = in_channels

#         self.confound_dim = confound_dim
        
#         self.conv1 = conv_style(self.in_channels, num_features[0])
#         self.bn1 = gnn.norm.BatchNorm(num_features[0])
#         self.conv2 = conv_style(num_features[0], num_features[1])
#         self.bn2 = gnn.norm.BatchNorm(num_features[1])

#         self.conv3 = conv_style(num_features[1], num_features[2])
#         self.bn3 = gnn.norm.BatchNorm(num_features[2])

#         self.conv4 = conv_style(num_features[2], num_features[3])
#         self.bn4 = gnn.norm.BatchNorm(num_features[3])



#         self.pool1 = hex_pooling_2(0, self.device)
#         self.pool2 = hex_pooling_2(1, self.device)
#         self.pool3 = hex_pooling_2(2, self.device)
#         self.pool4 = hex_pooling_2(3, self.device)

# #        upconv_top_index_163842, upconv_down_index_163842, upconv_top_index_40962, upconv_down_index_40962, upconv_top_index_10242, upconv_down_index_10242,  upconv_top_index_2562, upconv_down_index_2562,  upconv_top_index_642, upconv_down_index_642, upconv_top_index_162, upconv_down_index_162 = Get_upconv_index() 
#         upconv_top_index_40962, upconv_down_index_40962, upconv_top_index_10242, upconv_down_index_10242,  upconv_top_index_2562, upconv_down_index_2562,  upconv_top_index_642, upconv_down_index_642, upconv_top_index_162, upconv_down_index_162, upconv_top_index_42, upconv_down_index_42, upconv_top_index_12, upconv_down_index_12 = my_mat_Get_2ring_upconv_index()
        
#         self.activation_function = activation_function
        
#         self.fc1 = nn.Linear(num_features[3] * 162, num_features[3])
#         self.bnfc1 = nn.BatchNorm1d(num_features[3])
#         self.fc_encode_mu = nn.Linear(num_features[3] +confound_dim, latent_dim)
#         self.fc_encode_logvar = nn.Linear(num_features[3]+confound_dim , latent_dim)

#         self.fc_decode = nn.Linear(latent_dim+confound_dim, num_features[3] * 162)
        

#         self.upconv1 = upconv_layer(num_features[3], num_features[2], upconv_top_index_642, upconv_down_index_642 )
#         self.upconv2 = upconv_layer(num_features[2], num_features[1], upconv_top_index_2562, upconv_down_index_2562 )
#         self.upconv3 = upconv_layer(num_features[1], num_features[0], upconv_top_index_10242, upconv_down_index_10242 )
#         self.upconv4 = upconv_layer(num_features[0], self.in_channels, upconv_top_index_40962, upconv_down_index_40962)

# #
# #        self.conv8 = conv_style(num_features[1], num_features[1])
# #        self.conv7 = conv_style(num_features[2], num_features[2])
# ##        self.conv6 = conv_style(num_features[3], num_features[2])
# #        self.conv9 = conv_style(num_features[0], num_features[0])
        
#         self.bn7 = gnn.norm.BatchNorm(num_features[2])        
#         self.bn8 = gnn.norm.BatchNorm(num_features[1])   
#         self.bn9 = gnn.norm.BatchNorm(num_features[0])        

# #        self.bn1 = nn.BatchNorm1d(num_features[0],momentum=0.15, track_running_stats=False)
# #        self.bn2 = nn.BatchNorm1d(num_features[1],momentum=0.15, track_running_stats=False)
# #        self.bn3 = nn.BatchNorm1d(num_features[2],momentum=0.15, track_running_stats=False)
# #        self.bn4 = nn.BatchNorm1d(num_features[3],momentum=0.15, track_running_stats=False)
#         self.final_conv= nn.Conv1d(self.in_channels, self.in_channels, kernel_size=1)
        
        
#         self.out_ac = nn.Sigmoid()
        
    
#     def encode(self, data):

#         x = data.x
#         e = data.edge_index


#         batch=data.batch.to(self.device)
#         confound = data.y.unsqueeze(1).to(self.device)

#         x = self.conv1(x,e, pseudo_6.to(self.device))
# #        x = self.bn1(x)
#         x = self.activation_function(x)
#         x, i1 = self.pool1(x)


#         x = self.conv2(x,edge_index_5.to(self.device), pseudo_5.to(self.device))
# #        x = self.bn2(x)

#         x = self.activation_function(x)
#         x, i2 = self.pool2(x)
        
        
#         x = self.conv3(x,edge_index_4.to(self.device), pseudo_4.to(self.device))
# #        x = self.bn3(x)

#         x = self.activation_function(x)
#         x, i3 = self.pool3(x)
        
        
#         x = self.conv4(x,edge_index_3.to(self.device), pseudo_3.to(self.device))
# #        x = self.bn4(x)

#         x = self.activation_function(x)
#         x, i4 = self.pool4(x)
# #        x = self.bn4(x)

# #        x = self.fc1(x)

        
#         x = x.flatten()
#         x_f = self.fc1(x)
# #        x_out = self.bn0(x_out)

#         x_f = torch.cat([x_f, data.confound.squeeze(1)], dim=0)
#         encoding_mu = self.fc_encode_mu(x_f)
#         encoding_var = self.fc_encode_logvar(x_f)
#         indices_list = [i1,i2,i3,i4]
#         return encoding_mu, encoding_var, indices_list
    
#     def decode(self, confound, encoding):

#         encoding = torch.cat([encoding, confound.squeeze(1)], dim=0)
#         y = self.fc_decode(encoding)

#         y = y.view(162,-1,1)


#         #ico2
#         y = self.upconv1(y)
#         #ico3


#         y = self.activation_function(y)        
# #        y = self.conv7(y.squeeze(2),edge_index_3.to(self.device), pseudo_3.to(self.device)).unsqueeze(2)
# ##        y = self.bn7(y)
# #        y = self.activation_function(y)        

#         y = self.upconv2(y)
#         #ico4

        
        
#         y = self.activation_function(y)
        
# #        y = self.conv8(y.squeeze(2),edge_index_4.to(self.device), pseudo_4.to(self.device)).unsqueeze(2)
# ##        y = self.bn8(y)
# #
# #        y = self.activation_function(y)  
#         y = self.upconv3(y)

#         #ico5
                
#         y = self.activation_function(y)
# #        y = self.conv9(y.squeeze(2),edge_index_5.to(self.device), pseudo_5.to(self.device)).unsqueeze(2)
# ##        y = self.bn9(y)
# #
# #        y = self.activation_function(y)  
#         y = self.upconv4(y)

        
#         y = y.permute(2,1,0)

#         y =self.final_conv(y)

        
#         y = self.out_ac(y)
        
#         y = y.view(40962,-1)
        
        
#         return y
    
#     def reparametrize(self, mean, var):
#         std = torch.exp(0.5 * var) 
#         q = torch.distributions.Normal(mean, std)
        
#         z = q.rsample()                      # reparameterization trick
#         return z

        
        
        
#     def forward(self, data):
#         encoding_mu, encoding_var, indices = self.encode(data)
#         encoding = self.reparametrize(encoding_mu, encoding_var)
#         confound = data.confound
#         reconstruction = self.decode(confound, encoding)
# #        print('reooding_var)
#         return reconstruction, encoding_mu, encoding_var   


# class monet_variational_upconv_confounded2(nn.Module):

#     def __init__(self, num_features, conv_style=gmmconv,activation_function=nn.LeakyReLU(), in_channels = 4, latent_dim = 100, confound_dim = 19,device='cuda'):
#         super(monet_variational_upconv_confounded2, self).__init__()
#         self.conv_style = conv_style
#         self.device = device
#         self.in_channels = in_channels

#         self.confound_dim = confound_dim
        
#         self.conv1 = conv_style(self.in_channels, num_features[0])
#         self.bn1 = gnn.norm.BatchNorm(num_features[0])
#         self.conv2 = conv_style(num_features[0], num_features[1])
#         self.bn2 = gnn.norm.BatchNorm(num_features[1])

#         self.conv3 = conv_style(num_features[1], num_features[2])
#         self.bn3 = gnn.norm.BatchNorm(num_features[2])

#         self.conv4 = conv_style(num_features[2], num_features[3])
#         self.bn4 = gnn.norm.BatchNorm(num_features[3])



#         self.pool1 = hex_pooling_2(0, self.device)
#         self.pool2 = hex_pooling_2(1, self.device)
#         self.pool3 = hex_pooling_2(2, self.device)
#         self.pool4 = hex_pooling_2(3, self.device)

# #        upconv_top_index_163842, upconv_down_index_163842, upconv_top_index_40962, upconv_down_index_40962, upconv_top_index_10242, upconv_down_index_10242,  upconv_top_index_2562, upconv_down_index_2562,  upconv_top_index_642, upconv_down_index_642, upconv_top_index_162, upconv_down_index_162 = Get_upconv_index() 
#         upconv_top_index_40962, upconv_down_index_40962, upconv_top_index_10242, upconv_down_index_10242,  upconv_top_index_2562, upconv_down_index_2562,  upconv_top_index_642, upconv_down_index_642, upconv_top_index_162, upconv_down_index_162, upconv_top_index_42, upconv_down_index_42, upconv_top_index_12, upconv_down_index_12 = my_mat_Get_2ring_upconv_index()
        
#         self.activation_function = activation_function
        
#         self.fc1 = nn.Linear(num_features[3] * 162, num_features[3])
#         self.bnfc1 = nn.BatchNorm1d(num_features[3])
#         self.fc_encode_mu = nn.Linear(num_features[3] +confound_dim, latent_dim)
#         self.fc_encode_logvar = nn.Linear(num_features[3]+confound_dim , latent_dim)

#         self.fc_decode = nn.Linear(latent_dim+confound_dim, num_features[3] * 162)
        

#         self.upconv1 = upconv_layer(num_features[3], num_features[2], upconv_top_index_642, upconv_down_index_642 )
#         self.upconv2 = upconv_layer(num_features[2], num_features[1], upconv_top_index_2562, upconv_down_index_2562 )
#         self.upconv3 = upconv_layer(num_features[1], num_features[0], upconv_top_index_10242, upconv_down_index_10242 )
#         self.upconv4 = upconv_layer(num_features[0], self.in_channels, upconv_top_index_40962, upconv_down_index_40962)

# #
# #        self.conv8 = conv_style(num_features[1], num_features[1])
# #        self.conv7 = conv_style(num_features[2], num_features[2])
# ##        self.conv6 = conv_style(num_features[3], num_features[2])
# #        self.conv9 = conv_style(num_features[0], num_features[0])
        
#         self.bn7 = gnn.norm.BatchNorm(num_features[2])        
#         self.bn8 = gnn.norm.BatchNorm(num_features[1])   
#         self.bn9 = gnn.norm.BatchNorm(num_features[0])        

# #        self.bn1 = nn.BatchNorm1d(num_features[0],momentum=0.15, track_running_stats=False)
# #        self.bn2 = nn.BatchNorm1d(num_features[1],momentum=0.15, track_running_stats=False)
# #        self.bn3 = nn.BatchNorm1d(num_features[2],momentum=0.15, track_running_stats=False)
# #        self.bn4 = nn.BatchNorm1d(num_features[3],momentum=0.15, track_running_stats=False)
#         self.final_conv= conv_style(self.in_channels, self.in_channels)
        
        
#         self.out_ac = nn.Sigmoid()
        
    
#     def encode(self, data):

#         x = data.x
#         e = data.edge_index


#         batch=data.batch.to(self.device)
#         confound = data.y.unsqueeze(1).to(self.device)

#         x = self.conv1(x,e, pseudo_6.to(self.device))
# #        x = self.bn1(x)
#         x = self.activation_function(x)
#         x, i1 = self.pool1(x)


#         x = self.conv2(x,edge_index_5.to(self.device), pseudo_5.to(self.device))
# #        x = self.bn2(x)

#         x = self.activation_function(x)
#         x, i2 = self.pool2(x)
        
        
#         x = self.conv3(x,edge_index_4.to(self.device), pseudo_4.to(self.device))
# #        x = self.bn3(x)

#         x = self.activation_function(x)
#         x, i3 = self.pool3(x)
        
        
#         x = self.conv4(x,edge_index_3.to(self.device), pseudo_3.to(self.device))
# #        x = self.bn4(x)

#         x = self.activation_function(x)
#         x, i4 = self.pool4(x)
# #        x = self.bn4(x)

# #        x = self.fc1(x)

        
#         x = x.flatten()
#         x_f = self.fc1(x)
# #        x_out = self.bn0(x_out)

#         x_f = torch.cat([x_f, data.confound.squeeze(1)], dim=0)
#         encoding_mu = self.fc_encode_mu(x_f)
#         encoding_var = self.fc_encode_logvar(x_f)
#         indices_list = [i1,i2,i3,i4]
#         return encoding_mu, encoding_var, indices_list
    
#     def decode(self, confound, encoding):

#         encoding = torch.cat([encoding, confound.squeeze(1)], dim=0)
#         y = self.fc_decode(encoding)

#         y = y.view(162,-1,1)


#         #ico2
#         y = self.upconv1(y)
#         #ico3


#         y = self.activation_function(y)        
# #        y = self.conv7(y.squeeze(2),edge_index_3.to(self.device), pseudo_3.to(self.device)).unsqueeze(2)
# ##        y = self.bn7(y)
# #        y = self.activation_function(y)        

#         y = self.upconv2(y)
#         #ico4

        
        
#         y = self.activation_function(y)
        
# #        y = self.conv8(y.squeeze(2),edge_index_4.to(self.device), pseudo_4.to(self.device)).unsqueeze(2)
# ##        y = self.bn8(y)
# #
# #        y = self.activation_function(y)  
#         y = self.upconv3(y)

#         #ico5
                
#         y = self.activation_function(y)
# #        y = self.conv9(y.squeeze(2),edge_index_5.to(self.device), pseudo_5.to(self.device)).unsqueeze(2)
# ##        y = self.bn9(y)
# #
# #        y = self.activation_function(y)  
#         y = self.upconv4(y)

#         y = y.squeeze(2)
# #        y = y.permute(2,1,0)

#         y =self.final_conv(y,edge_index_6.to(self.device), pseudo_6.to(self.device))

        
#         y = self.out_ac(y)
        
#         y = y.view(40962,-1)
        
        
#         return y
    
#     def reparametrize(self, mean, var):
#         std = torch.exp(0.5 * var) 
#         q = torch.distributions.Normal(mean, std)
        
#         z = q.rsample()                      # reparameterization trick
#         return z

        
        
        
#     def forward(self, data):
#         encoding_mu, encoding_var, indices = self.encode(data)
#         encoding = self.reparametrize(encoding_mu, encoding_var)
#         confound = data.confound
#         reconstruction = self.decode(confound, encoding)
# #        print('reooding_var)
#         return reconstruction, encoding_mu, encoding_var   




# class graphnet_variational_upconv(nn.Module):

#     def __init__(self, num_features, conv_style=graphconv,activation_function=nn.Tanh(), in_channels = 4, latent_dim = 100, device='cuda'):
#         super(graphnet_variational_upconv, self).__init__()
#         self.conv_style = conv_style
#         self.device = device
#         self.in_channels = in_channels
#         self.fc_start = nn.Linear(self.in_channels, num_features[0])
#         self.fc_end = nn.Linear(num_features[0], self.in_channels)
        
#         self.conv1 = conv_style(num_features[0], num_features[0])
#         self.conv2 = conv_style(num_features[0], num_features[1])
#         self.conv3 = conv_style(num_features[1], num_features[2])
#         self.conv4 = conv_style(num_features[2], num_features[3])
#         self.pool1 = hex_pooling_2(0, self.device)
#         self.pool2 = hex_pooling_2(1, self.device)
#         self.pool3 = hex_pooling_2(2, self.device)
#         self.pool4 = hex_pooling_2(3, self.device)

# #        upconv_top_index_163842, upconv_down_index_163842, upconv_top_index_40962, upconv_down_index_40962, upconv_top_index_10242, upconv_down_index_10242,  upconv_top_index_2562, upconv_down_index_2562,  upconv_top_index_642, upconv_down_index_642, upconv_top_index_162, upconv_down_index_162 = Get_upconv_index() 
#         upconv_top_index_40962, upconv_down_index_40962, upconv_top_index_10242, upconv_down_index_10242,  upconv_top_index_2562, upconv_down_index_2562,  upconv_top_index_642, upconv_down_index_642, upconv_top_index_162, upconv_down_index_162, upconv_top_index_42, upconv_down_index_42, upconv_top_index_12, upconv_down_index_12 = my_mat_Get_2ring_upconv_index()
        
#         self.activation_function = activation_function
        
#         self.fc1 = nn.Linear(num_features[3] * 162, num_features[3])
#         self.fc_encode_mu = nn.Linear(num_features[3] , latent_dim)
#         self.fc_encode_logvar = nn.Linear(num_features[3] , latent_dim)

#         self.fc_decode = nn.Linear(latent_dim, num_features[3] * 162)
        

#         self.upconv1 = upconv_layer(num_features[3], num_features[2], upconv_top_index_642, upconv_down_index_642 )
#         self.upconv2 = upconv_layer(num_features[2], num_features[1], upconv_top_index_2562, upconv_down_index_2562 )
#         self.upconv3 = upconv_layer(num_features[1], num_features[0], upconv_top_index_10242, upconv_down_index_10242 )
#         self.upconv4 = upconv_layer(num_features[0], self.in_channels, upconv_top_index_40962, upconv_down_index_40962)


#         self.conv8 = conv_style(num_features[1], num_features[1])
#         self.conv7 = conv_style(num_features[2], num_features[2])
# #        self.conv6 = conv_style(num_features[3], num_features[2])
#         self.conv9 = conv_style(num_features[0], num_features[0])
        
# #        self.bn1 = nn.BatchNorm1d(num_features[0],momentum=0.15, track_running_stats=False)
# #        self.bn2 = nn.BatchNorm1d(num_features[1],momentum=0.15, track_running_stats=False)
# #        self.bn3 = nn.BatchNorm1d(num_features[2],momentum=0.15, track_running_stats=False)
# #        self.bn4 = nn.BatchNorm1d(num_features[3],momentum=0.15, track_running_stats=False)
        
        
#         self.out_ac = nn.Sigmoid()
        
    
#     def encode(self, data):
#         x = data.x

#         e = data.edge_index


#         batch=data.batch.to(self.device)
#         x = self.fc_start(x)
#         x = self.activation_function(x)

#         x = self.conv1(x,e) #ico6
#         x = self.activation_function(x)

#         x1, i1 = self.pool1(x)

# #        x = self.bn1(x)
#         #ico5

#         x = self.conv2(x1,edge_index_5.to(self.device))
#         x = self.activation_function(x)

#         x2, i2 = self.pool2(x)
# #        x = self.bn2(x)
        
#         #ico4
        
#         x = self.conv3(x2,edge_index_4.to(self.device))
#         x = self.activation_function(x)
#         x3, i3 = self.pool3(x)
        
# #        x = self.bn3(x)
#         #ico3
#         x = self.conv4(x3,edge_index_3.to(self.device))
#         x = self.activation_function(x)
#         x, i4 = self.pool4(x)
# #        x = self.bn4(x)

# #        x = self.fc1(x)

        
#         x = x.flatten()
#         x_f = self.fc1(x)
# #        x_out = self.bn0(x_out)

#         encoding_mu = self.fc_encode_mu(x_f)
#         encoding_var = self.fc_encode_logvar(x_f)
#         indices_list = [i1,i2,i3,i4]
#         return encoding_mu, encoding_var, indices_list
    
#     def decode(self, encoding):

#         y = self.fc_decode(encoding)

#         y = y.view(162,-1,1)


#         #ico2
#         y = self.upconv1(y)
#         #ico3


#         y = self.activation_function(y)        
#         y = self.conv7(y.squeeze(2),edge_index_3.to(self.device)).unsqueeze(2)

#         y = self.activation_function(y)        

#         y = self.upconv2(y)
#         #ico4

        
        
#         y = self.activation_function(y)
        
#         y = self.conv8(y.squeeze(2),edge_index_4.to(self.device)).unsqueeze(2)
#         y = self.activation_function(y)  
#         y = self.upconv3(y)

#         #ico5
                
#         y = self.activation_function(y)
#         y = self.conv9(y.squeeze(2),edge_index_5.to(self.device)).unsqueeze(2)
#         y = self.activation_function(y)  
#         y = self.upconv4(y)

#         y = y.flatten()
        
#         y = self.out_ac(y)
        
#         y = y.view(40962,-1)
        
        
#         return y
    
#     def reparametrize(self, mean, var):
#         epsilon = torch.randn_like(var)       # sampling epsilon        
#         z = mean + var*epsilon                          # reparameterization trick
#         return z
    
#     def forward(self, data):
#         encoding_mu, encoding_var, indices = self.encode(data)
#         encoding = self.reparametrize(encoding_mu, encoding_var)
        
#         reconstruction = self.decode(encoding)
# #        print('reooding_var)
#         return reconstruction, encoding_mu, encoding_var

# class chebnet_nopool_autoencoder(nn.Module):

#     def __init__(self, num_features, conv_style=chebconv,activation_function=nn.LeakyReLU(), in_channels = 4, latent_dim = 100, device='cuda'):
#         super(chebnet_nopool_autoencoder, self).__init__()
#         self.conv_style = conv_style
#         self.device = device
#         self.in_channels = in_channels
#         self.fc_start = nn.Linear(self.in_channels, num_features[0])
#         self.fc_end = nn.Linear(num_features[0], self.in_channels)
#         self.conv1 = conv_style(num_features[0], num_features[0])
#         self.conv2 = conv_style(num_features[0], num_features[1])
#         self.conv3 = conv_style(num_features[1], num_features[2])
#         self.conv4 = conv_style(num_features[2], num_features[3])
#         self.pool1 = hex_pooling_2(0, self.device)
#         self.pool2 = hex_pooling_2(1, self.device)
#         self.pool3 = hex_pooling_2(2, self.device)
#         self.pool4 = hex_pooling_2(3, self.device)

# #        upconv_top_index_163842, upconv_down_index_163842, upconv_top_index_40962, upconv_down_index_40962, upconv_top_index_10242, upconv_down_index_10242,  upconv_top_index_2562, upconv_down_index_2562,  upconv_top_index_642, upconv_down_index_642, upconv_top_index_162, upconv_down_index_162 = Get_upconv_index() 
# #        upconv_top_index_40962, upconv_down_index_40962, upconv_top_index_10242, upconv_down_index_10242,  upconv_top_index_2562, upconv_down_index_2562,  upconv_top_index_642, upconv_down_index_642, upconv_top_index_162, upconv_down_index_162, upconv_top_index_42, upconv_down_index_42, upconv_top_index_12, upconv_down_index_12 = my_mat_Get_2ring_upconv_index()
        
#         self.activation_function = activation_function
        
#         self.fc1 = nn.Linear(num_features[3], num_features[3])
#         self.fc_encode = nn.Linear(num_features[3] +1, latent_dim)


#         self.fc_decode = nn.Linear(latent_dim+1, num_features[3] * 162)
        

#         self.unpool1 = hex_unpooling(self.device)
#         self.unpool2 = hex_unpooling(self.device)
#         self.unpool3 = hex_unpooling(self.device)
#         self.unpool4 = hex_unpooling(self.device)

#         self.conv8 = conv_style(num_features[1], num_features[0])
#         self.conv7 = conv_style(num_features[2], num_features[1])
#         self.conv6 = conv_style(num_features[3], num_features[2])
#         self.conv9 = conv_style(num_features[0], num_features[0])
        
# #        self.bn1 = nn.BatchNorm1d(num_features[0],momentum=0.15, track_running_stats=False)
# #        self.bn2 = nn.BatchNorm1d(num_features[1],momentum=0.15, track_running_stats=False)
# #        self.bn3 = nn.BatchNorm1d(num_features[2],momentum=0.15, track_running_stats=False)
# #        self.bn4 = nn.BatchNorm1d(num_features[3],momentum=0.15, track_running_stats=False)
        
        
#         self.out_ac = nn.Sigmoid()
        
    
#     def encode(self, data):
#         x = data.x

#         e = data.edge_index


#         batch=data.batch.to(self.device)
#         x = self.conv1d_1(x, e)
#         x = self.activation_function(x)

#         x = self.conv1(x,e) #ico6
#         x = self.activation_function(x)

#         x1, i1 = self.pool1(x)

# #        x = self.bn1(x)
#         #ico5

#         x = self.conv2(x1,edge_index_5.to(self.device))
#         x = self.activation_function(x)

#         x2, i2 = self.pool2(x)
# #        x = self.bn2(x)
        
#         #ico4
        
#         x = self.conv3(x2,edge_index_4.to(self.device))
#         x = self.activation_function(x)
#         x3, i3 = self.pool3(x)
        
# #        x = self.bn3(x)
#         #ico3
#         x = self.conv4(x3,edge_index_3.to(self.device))
#         x = self.activation_function(x)
#         x, i4 = self.pool4(x)
# #        x = self.bn4(x)

# #        x = self.fc1(x)

# #        x_max = gnn.global_max_pool(x, batch[:162])
# #        x_mean = gnn.global_mean_pool(x, batch[:162])
# #        
# #        x_c = torch.cat([x_max, x_mean], dim = 1)
# #        x_out = self.bn0(x_out)
#         x_f = x.view(1, -1)
#         confound = data.metdata.to(self.device)
        
#         x_f = torch.cat([x_f, confound], dim=1)
#         encoding = self.fc_encode(x_f)

#         indices_list = [i1,i2,i3,i4]
#         return encoding, indices_list
    
#     def decode(self, confound, encoding, indices_list):
#         i1,i2,i3,i4 = indices_list
#         encoding = torch.cat([encoding, confound], dim=1)
#         y = self.fc_decode(encoding)
#         y = self.activation_function(y)
#         y = y.view(162,-1)

#         #ico2

#         y = self.unpool1(y, i4)

#         #ico3
#         y = self.conv6(y,edge_index_3.to(self.device))

#         y = self.activation_function(y)
#         y = self.unpool2(y, i3)

#         #ico4
#         y = self.conv7(y, edge_index_4.to(self.device))

        
#         y = self.activation_function(y)
#         y = self.unpool3(y, i2)

        
#         y = self.conv8(y, edge_index_5.to(self.device))
#         #ico5

                
#         y = self.activation_function(y)
#         y = self.unpool4(y, i1)

        
#         y = self.conv9(y, edge_index_6.to(self.device))
#         y = self.activation_function(y)
#         y = self.conv1d_2(y, edge_index_6.to(self.device))
#         y = y.flatten()
        
#         y = self.out_ac(y)
        
#         y = y.view(40962,-1)
        
#         return y
    
#     def forward(self, data):
#         encoding, indices = self.encode(data)
#         confound = data.metadata.to(self.device)
#         reconstruction = self.decode(confound,encoding, indices)
        
#         return reconstruction


    
# class hex_unpooling(nn.Module):
#     def __init__(self, device):
#         super(hex_unpooling, self).__init__()
#         self.device=device
        
#     def forward(self, x, indices):        

#         other_indices = torch.arange(indices.shape[1])
        
#         other_indices = other_indices.repeat(len(indices),1).to(self.device)
        
        
#         L = int( (len(x) * 4 ) - 6)

#         y = torch.zeros(L, x.shape[1]).to(self.device)
        
#         y[indices, other_indices] = x
        
#         return y    
    
# class do_downsample(nn.Module):

#     def __init__(self, inchans, outchans, conv_style, ico_level, device):
#         super(do_downsample, self).__init__()

#         self.conv_style = conv_style
#         self.inchans = inchans
#         self.outchans = outchans
#         self.ico_level = ico_level
#         self.device = device
#         self.conv1 = conv_style(self.inchans, self.outchans)
#         self.pooling_method = hex_pooling(self.ico_level, self.device)
        
        
#     def forward(self,x,e):

        
#         x = self.conv1(x,e)
        
#         x = self.pooling_method(x)
        
#         return x
        

# class upconv_layer(nn.Module):
#     """
#     The transposed convolution layer on icosahedron discretized sphere using 1-ring filter
    
#     Input: 
#         N x in_feats, tensor
#     Return:
#         ((Nx4)-6) x out_feats, tensor
    
#     """  

#     def __init__(self, in_feats, out_feats, upconv_top_index, upconv_down_index):
#         super(upconv_layer, self).__init__()

#         self.in_feats = in_feats
#         self.out_feats = out_feats
#         self.upconv_top_index = upconv_top_index
#         self.upconv_down_index = upconv_down_index
#         self.weight = nn.Linear(in_feats, 7 * out_feats)
        
#     def forward(self, x):

#         raw_nodes = x.size()[0]
#         new_nodes = int(raw_nodes*4 - 6)

#         x = x.reshape(x.size(2),x.size(0), self.in_feats)

#         x = self.weight(x).permute(1,2,0)

# #        x = self.weight(x)
#         x = x.view(len(x) * 7, self.out_feats,-1)

#         x1 = x[self.upconv_top_index]

#         assert x1.size()[:2] == torch.Size([raw_nodes, self.out_feats]) , print(self.out_feats, raw_nodes, x1.size())

#         x2 = x[self.upconv_down_index]
#         x2 = x2.reshape(-1, self.out_feats, 2, x.size(2))

#         x = torch.cat((x1,torch.mean(x2, 2)), 0)

#         assert(x.size()[:2] == torch.Size([new_nodes, self.out_feats]))

#         return x

      
        
        
        
        
        

# class down_block(nn.Module):
#     """
#     downsampling block in spherical unet
#     mean pooling => (conv => BN => ReLU) * 2
    
#     """
#     def __init__(self, conv_layer, in_ch, out_ch, neigh_orders, pool_neigh_orders, first = False):
#         super(down_block, self).__init__()


# #        Batch norm version
#         if first:
#             self.block = nn.Sequential(
#                 conv_layer(in_ch, out_ch, neigh_orders),
#                 nn.BatchNorm1d(out_ch, momentum=0.15, affine=True, track_running_stats=False),
#                 nn.LeakyReLU(0.2, inplace=True),
#                 conv_layer(out_ch, out_ch, neigh_orders),
#                 nn.BatchNorm1d(out_ch, momentum=0.15, affine=True, track_running_stats=False),
#                 nn.LeakyReLU(0.2, inplace=True)
#         )
            
#         else:
#             self.block = nn.Sequential(
#                 pool_layer(pool_neigh_orders, 'mean'),
#                 conv_layer(in_ch, out_ch, neigh_orders),
#                 nn.BatchNorm1d(out_ch, momentum=0.15, affine=True, track_running_stats=False),
#                 nn.LeakyReLU(0.2, inplace=True),
#                 conv_layer(out_ch, out_ch, neigh_orders),
#                 nn.BatchNorm1d(out_ch, momentum=0.15, affine=True, track_running_stats=False),
#                 nn.LeakyReLU(0.2, inplace=True),
#         )


#     def forward(self, x):
#         # batch norm version
#         x = self.block(x)
        
#         return x

# class up_block_no_skip(nn.Module):
#     """Define the upsamping block for a VAE
#     upconv => (conv => BN => ReLU) * 2
    
#     Parameters:
#             in_ch (int) - - input features/channels
#             out_ch (int) - - output features/channels    
#             neigh_orders (tensor, int)  - - conv layer's filters' neighborhood orders
            
#     """    
#     def __init__(self, conv_layer, in_ch, out_ch, neigh_orders, upconv_top_index, upconv_down_index):
#         super(up_block_no_skip, self).__init__()
        
#         self.up = upconv_layer(in_ch, out_ch, upconv_top_index, upconv_down_index)
        
#         # batch norm version
#         self.double_conv = nn.Sequential(
#              conv_layer(out_ch, out_ch, neigh_orders),
#              nn.BatchNorm1d(out_ch, momentum=0.15, affine=True, track_running_stats=False),
#              nn.LeakyReLU(0.2, inplace=True),
#              conv_layer(out_ch, out_ch, neigh_orders),
#              nn.BatchNorm1d(out_ch, momentum=0.15, affine=True, track_running_stats=False),
#              nn.LeakyReLU(0.2, inplace=True)
#         )

#     def forward(self, x1):
        
#         x1 = self.up(x1)

# #        x = torch.cat((x1, x2), 1) 
#         x = self.double_conv(x1)
    
#         return x
    
    
    
# class VAE_mat_1ring(nn.Module):
#     """Define the Spherical UNet structure

#     """    
#     def __init__(self, in_ch, size):
#         """ Initialize the Spherical UNet.

#         Parameters:
#             in_ch (int) - - input features/channels
#             out_ch (int) - - output features/channels
#         """
#         super(VAE_mat_1ring, self).__init__()

#         #neigh_indices_10242, neigh_indices_2562, neigh_indices_642, neigh_indices_162, neigh_indices_42 = Get_indices_order()
#         #neigh_orders_10242, neigh_orders_2562, neigh_orders_642, neigh_orders_162, neigh_orders_42, neigh_orders_12 = Get_neighs_order()
        
#         upconv_top_index_40962, upconv_down_index_40962, upconv_top_index_10242, upconv_down_index_10242,  upconv_top_index_2562, upconv_down_index_2562,  upconv_top_index_642, upconv_down_index_642, upconv_top_index_162, upconv_down_index_162, upconv_top_index_42, upconv_down_index_42, upconv_top_index_12, upconv_down_index_12 = my_mat_Get_2ring_upconv_index()
#         neigh_orders = my_Get_neighs_order()
        

#         chs = [in_ch, 64 , 64, 128, 256, 512, 1024]
        
#         conv_layer = onering_conv_layer
        
#         self.down1 = down_block(conv_layer, chs[0], chs[1], neigh_orders[1], neigh_orders[0], False)
#         self.down2 = down_block(conv_layer, chs[1], chs[2], neigh_orders[2], neigh_orders[1])
#         self.down3 = down_block(conv_layer, chs[2], chs[3], neigh_orders[3], neigh_orders[2])
#         self.down4 = down_block(conv_layer, chs[3], chs[4], neigh_orders[4], neigh_orders[3])
#         self.down5 = down_block(conv_layer, chs[4], chs[5], neigh_orders[5], neigh_orders[4])

#         self.fc1 = nn.Linear(chs[6] * 42, size) # logvar
        
#         self.fc2 = nn.Linear(chs[6] * 42, size) # mu

#         self.fc3 = nn.Linear( size, chs[6] * 42) # joins them back

#         self.up1 = up_block_no_skip(conv_layer, chs[5], chs[4], neigh_orders[4], upconv_top_index_162, upconv_down_index_162)
#         self.up2 = up_block_no_skip(conv_layer, chs[4], chs[3], neigh_orders[3], upconv_top_index_642, upconv_down_index_642)
#         self.up3 = up_block_no_skip(conv_layer, chs[3], chs[2], neigh_orders[2], upconv_top_index_2562, upconv_down_index_2562)
#         self.up4 = up_block_no_skip(conv_layer, chs[2], chs[1], neigh_orders[1], upconv_top_index_10242, upconv_down_index_10242)
#         self.up5 = down_block(conv_layer, chs[1],chs[1] , neigh_orders[1], neigh_orders[0], False)
        
# #        self.outc = nn.Sequential(
# #                nn.Linear(chs[1], out_ch)
# #                )
                
#     def encode(self, x):
#         x2 = self.down1(x)

#         x3 = self.down2(x2)
#         x4 = self.down3(x3)
#         x5 = self.down4(x4)
#         x6 = self.down5(x5)

# #        print(x6.shape)
#         x6=x6.unsqueeze(0)
# #        x6 = x6.permute(2,0,1)

#         x6 = x6.reshape(x6.shape[0], -1)
#         out1 = self.fc1(x6) 
        
#         out2 = self.fc2(x6)
        
        

#         return out1, out2
    
#     def reparameterize(self, mu, logvar):
        
#         std = torch.exp(0.5*logvar)
        
#         eps = torch.randn_like(std)
        
#         return eps.mul(std).add_(mu)
    
#     def decode(self, z):
#         z = self.fc3(z)

#         deconv_input = z.reshape(-1, 42, int(z.shape[1] / 42))
#         deconv_input = deconv_input.permute(1,2,0)
#         print(deconv_input.shape)
#         x = self.up1(deconv_input)

#         x = self.up2(x)
#         x = self.up3(x)
#         x = self.up4(x) 
#         x = self.up5(x) 
#         x = nn.Sigmoid()(x)
#         return x
    
#     def forward(self, x):
#         mu, logvar = self.encode(x)
        
#         z = self.reparameterize(mu, logvar)
        
#         return self.decode(z), mu, logvar
    
#     def representation(self,x):
#        encoding = self.encode(x)
#        return self.reparameterize(encoding[0], encoding[1])
        
        
        

        
# class VAE_mat_1ring(nn.Module):
#     """Define the Spherical UNet structure

#     """    
#     def __init__(self, in_ch, size):
#         """ Initialize the Spherical UNet.

#         Parameters:
#             in_ch (int) - - input features/channels
#             out_ch (int) - - output features/channels
#         """
#         super(VAE_mat_1ring, self).__init__()

#         #neigh_indices_10242, neigh_indices_2562, neigh_indices_642, neigh_indices_162, neigh_indices_42 = Get_indices_order()
#         #neigh_orders_10242, neigh_orders_2562, neigh_orders_642, neigh_orders_162, neigh_orders_42, neigh_orders_12 = Get_neighs_order()
        
#         neigh_orders = my_mat_Get_neighs_order()
#         upconv_top_index_40962, upconv_down_index_40962, upconv_top_index_10242, upconv_down_index_10242,  upconv_top_index_2562, upconv_down_index_2562, upconv_top_index_642, upconv_down_index_642, upconv_top_index_162, upconv_down_index_162, upconv_top_index_42, upconv_down_index_42, upconv_top_index_12, upconv_down_index_12 = my_mat_Get_2ring_upconv_index()

#         chs = [in_ch, 64 , 64, 128, 256, 512, 2048]
        
#         conv_layer = onering_conv_layer

#         self.down1 = down_block(conv_layer, chs[0], chs[1], neigh_orders[0], None, True)
#         self.down2 = down_block(conv_layer, chs[0], chs[1], neigh_orders[1], neigh_orders[0])
#         self.down3 = down_block(conv_layer, chs[1], chs[2], neigh_orders[2], neigh_orders[1])
#         self.down4 = down_block(conv_layer, chs[2], chs[3], neigh_orders[3], neigh_orders[2])
#         self.down5 = down_block(conv_layer, chs[3], chs[4], neigh_orders[4], neigh_orders[3])


#         self.fc1 = nn.Linear(chs[5] * 42, size) # logvar
        
#         self.fc2 = nn.Linear(chs[5] * 42, size) # mu

#         self.fc3 = nn.Linear( size, chs[5] * 42) # joins them back

#         self.up1 = up_block_no_skip(conv_layer, chs[5], chs[4], neigh_orders[4], upconv_top_index_162, upconv_down_index_162)
#         self.up2 = up_block_no_skip(conv_layer, chs[4], chs[3], neigh_orders[3], upconv_top_index_642, upconv_down_index_642)
#         self.up3 = up_block_no_skip(conv_layer, chs[3], chs[2], neigh_orders[2], upconv_top_index_2562, upconv_down_index_2562)
#         self.up4 = up_block_no_skip(conv_layer, chs[2], chs[1], neigh_orders[1], upconv_top_index_10242, upconv_down_index_10242)
#         self.up5 = up_block_no_skip(conv_layer, chs[1], chs[0], neigh_orders[0], upconv_top_index_40962, upconv_down_index_40962)
        
# #        self.outc = nn.Sequential(
# #                nn.Linear(chs[1], out_ch)
# #                )
                
#     def encode(self, x):
#         x2 = self.down1(x)

#         x3 = self.down2(x2)
#         x4 = self.down3(x3)
#         x5 = self.down4(x4)
#         x6 = self.down5(x5)
#         x6=x6.unsqueeze(0)
# #        x6 = x6.permute(2,0,1)

#         x6 = x6.reshape(x6.shape[0], -1)

#         out1 = self.fc1(x6) 
        
#         out2 = self.fc2(x6)
        
        

#         return out1, out2
    
#     def reparameterize(self, mu, logvar):
        
#         std = torch.exp(0.5*logvar)
        
#         eps = torch.randn_like(std)
        
#         return eps.mul(std).add_(mu)
    
#     def decode(self, z):
#         z = self.fc3(z)

#         deconv_input = z.reshape(42,int(z.shape[1] / 42),-1)


#         x = self.up1(deconv_input).unsqueeze(2)
#         x = self.up2(x).unsqueeze(2)
#         x = self.up3(x).unsqueeze(2)
#         x = self.up4(x).unsqueeze(2)
#         x = self.up5(x).unsqueeze(1)
#         x = nn.Sigmoid()(x)
#         return x
    
#     def forward(self, x):
#         mu, logvar = self.encode(x)
        
#         z = self.reparameterize(mu, logvar)
        
#         return self.decode(z), mu, logvar
    
#     def representation(self,x):
#        encoding = self.encode(x)
#        return self.reparameterize(encoding[0], encoding[1])
   
        
        
   
# class AE_mat_1ring(nn.Module):
#     """Define the Spherical UNet structure

#     """    
#     def __init__(self, in_ch, size):
#         """ Initialize the Spherical UNet.

#         Parameters:
#             in_ch (int) - - input features/channels
#             out_ch (int) - - output features/channels
#         """
#         super(AE_mat_1ring, self).__init__()

#         #neigh_indices_10242, neigh_indices_2562, neigh_indices_642, neigh_indices_162, neigh_indices_42 = Get_indices_order()
#         #neigh_orders_10242, neigh_orders_2562, neigh_orders_642, neigh_orders_162, neigh_orders_42, neigh_orders_12 = Get_neighs_order()
        
#         neigh_orders = my_mat_Get_neighs_order()
#         parent_nodes_40962, parent_nodes_10242, parent_nodes_2562, parent_nodes_642, parent_nodes_162, parent_nodes_42, parent_nodes_12 = my_mat_Get_Parents_Nodes()
#         chs = [in_ch, 128 , 64, 32, 16, 8, 2048]
        
#         conv_layer = onering_conv_layer
#         self.end = chs[5]
#         self.down1 = down_block(conv_layer, chs[0], chs[1], neigh_orders[0], None, True)
#         self.down2 = down_block(conv_layer, chs[1], chs[2], neigh_orders[1], neigh_orders[0])
#         self.down3 = down_block(conv_layer, chs[2], chs[3], neigh_orders[2], neigh_orders[1])
#         self.down4 = down_block(conv_layer, chs[3], chs[4], neigh_orders[3], neigh_orders[2])
#         self.down5 = down_block(conv_layer, chs[4], chs[5], neigh_orders[4], neigh_orders[3])

#         self.fc1 = nn.Linear(chs[5] * 162, size) # logvar
# #        
# #        self.fc2 = nn.Linear(chs[5] * 42, size) # mu
# #
#         self.fc3 = nn.Linear( size, chs[5] * 162) # joins them back

# #        self.up1 = up_block_no_skip(conv_layer, chs[5], chs[4], neigh_orders[4], upconv_top_index_162, upconv_down_index_162)
#         self.up1 = upsample_interpolation(parent_nodes_642)
#         self.up2 = upsample_interpolation(parent_nodes_2562)
#         self.up3 = upsample_interpolation(parent_nodes_10242)
#         self.up4 = upsample_interpolation(parent_nodes_40962)

# #        self.conv0 = up_block_no_skip(conv_layer, chs[5], chs[4], neigh_orders[3], upconv_top_index_642, upconv_down_index_642)
#         self.conv0 = down_block(conv_layer, chs[5], chs[4], neigh_orders[4], None, True)

#         self.conv1 = down_block(conv_layer, chs[4], chs[3], neigh_orders[3], None, True)
#         self.conv2 = down_block(conv_layer, chs[3], chs[2], neigh_orders[2], None, True)
#         self.conv3 = down_block(conv_layer, chs[2], chs[1], neigh_orders[1], None, True)
        
#         self.conv4 = down_block(conv_layer, chs[1], chs[0], neigh_orders[0], None, True)
        
# #        self.conv5 = down_block(conv_layer, chs[1], chs[0], neigh_orders[0], None, True)

# #        self.outc = nn.Sequential(
# #                nn.Linear(chs[1], out_ch)
# #                )
#         self.out_ac = nn.Tanh()
#         self.in_ac = nn.Tanh()
#     def encode(self, x):
#         x2 = self.down1(x)

#         x3 = self.down2(x2)
#         x4 = self.down3(x3)
#         x5 = self.down4(x4)
#         x6 = self.down5(x5).unsqueeze(2)
# #        
# #        x6 = x6.permute(2,0,1)
# #
#         x6 = x6.reshape(x6.shape[2],-1)
#         out = self.fc1(x6) 
# #        
# #        out2 = self.fc2(x6)
        
        
# #        out = x6
#         out = self.out_ac(out)
#         return out
    

    
#     def decode(self, z):

#         #connect all of them now#

#         z = self.fc3(z)
#         z = self.in_ac(z)
#         z = z.reshape(162,self.end)

#         x = self.conv0(z)
#         x = self.up1(x).unsqueeze(2)

#         x = self.conv1(x)
#         x = self.up2(x).unsqueeze(2)
#         x = self.conv2(x)

#         x = self.up3(x).unsqueeze(2)
#         x = self.conv3(x)

#         x = self.up4(x).unsqueeze(2)
#         x = self.conv4(x)

# #        x = self.up5(x).unsqueeze(2)
# #        x = self.conv5(x)

#         x = nn.Sigmoid()(x)
        
#         return x
    
#     def forward(self, x):
#         compressed = self.encode(x)
        
        
#         return self.decode(compressed)

    
    
# class AE_mat_1ring(nn.Module):
#     """Define the Spherical UNet structure

#     """    
#     def __init__(self, in_ch, size):
#         """ Initialize the Spherical UNet.

#         Parameters:
#             in_ch (int) - - input features/channels
#             out_ch (int) - - output features/channels
#         """
#         super(AE_mat_1ring, self).__init__()

#         #neigh_indices_10242, neigh_indices_2562, neigh_indices_642, neigh_indices_162, neigh_indices_42 = Get_indices_order()
#         #neigh_orders_10242, neigh_orders_2562, neigh_orders_642, neigh_orders_162, neigh_orders_42, neigh_orders_12 = Get_neighs_order()
        
#         neigh_orders = my_mat_Get_neighs_order()
#         parent_nodes_40962, parent_nodes_10242, parent_nodes_2562, parent_nodes_642, parent_nodes_162, parent_nodes_42, parent_nodes_12 = my_mat_Get_Parents_Nodes()
#         chs = [in_ch, 128 , 64, 32, 16, 8, 2048]
        
#         conv_layer = onering_conv_layer
#         self.end = chs[5]
#         self.down1 = down_block(conv_layer, chs[0], chs[1], neigh_orders[0], None, True)
#         self.down2 = down_block(conv_layer, chs[1], chs[2], neigh_orders[1], neigh_orders[0])
#         self.down3 = down_block(conv_layer, chs[2], chs[3], neigh_orders[2], neigh_orders[1])
#         self.down4 = down_block(conv_layer, chs[3], chs[4], neigh_orders[3], neigh_orders[2])
#         self.down5 = down_block(conv_layer, chs[4], chs[5], neigh_orders[4], neigh_orders[3])

#         self.fc1 = nn.Linear(chs[5] * 162, size) # logvar
# #        
# #        self.fc2 = nn.Linear(chs[5] * 42, size) # mu
# #
#         self.fc3 = nn.Linear( size, chs[5] * 162) # joins them back

# #        self.up1 = up_block_no_skip(conv_layer, chs[5], chs[4], neigh_orders[4], upconv_top_index_162, upconv_down_index_162)
#         self.up1 = upsample_interpolation(parent_nodes_642)
#         self.up2 = upsample_interpolation(parent_nodes_2562)
#         self.up3 = upsample_interpolation(parent_nodes_10242)
#         self.up4 = upsample_interpolation(parent_nodes_40962)

# #        self.conv0 = up_block_no_skip(conv_layer, chs[5], chs[4], neigh_orders[3], upconv_top_index_642, upconv_down_index_642)
#         self.conv0 = down_block(conv_layer, chs[5], chs[4], neigh_orders[4], None, True)

#         self.conv1 = down_block(conv_layer, chs[4]+chs[5], chs[3], neigh_orders[3], None, True)
#         self.conv2 = down_block(conv_layer, chs[3]+chs[4]+chs[5], chs[2], neigh_orders[2], None, True)
#         self.conv3 = down_block(conv_layer, chs[2]+chs[3]+chs[4]+chs[5], chs[1], neigh_orders[1], None, True)
        
#         self.conv4 = down_block(conv_layer, chs[1]+chs[2]+chs[3]+chs[4]+chs[5], chs[0], neigh_orders[0], None, True)
        
# #        self.conv5 = down_block(conv_layer, chs[1], chs[0], neigh_orders[0], None, True)

# #        self.outc = nn.Sequential(
# #                nn.Linear(chs[1], out_ch)
# #                )
#         self.out_ac = nn.LeakyReLU()
#         self.in_ac = nn.LeakyReLU()
#     def encode(self, x):
#         x2 = self.down1(x)

#         x3 = self.down2(x2)
#         x4 = self.down3(x3)
#         x5 = self.down4(x4)
#         x6 = self.down5(x5).unsqueeze(2)
# #        
# #        x6 = x6.permute(2,0,1)
# #
#         x6 = x6.reshape(x6.shape[2],-1)
#         out = self.fc1(x6) 
# #        
# #        out2 = self.fc2(x6)
        
        
# #        out = x6
#         out = self.out_ac(out)
#         return out
    

    
#     def decode(self, z):

#         #connect all of them now#

#         z = self.fc3(z)
#         z = self.in_ac(z)
#         z = z.reshape(162,self.end)

#         x = self.conv0(z)
#         x = self.up1(x).unsqueeze(2)
        

#         z2 = self.up1(z).unsqueeze(2)
#         z2 = torch.cat([x,z2],dim=1)




#         x = self.conv1(z2)
#         x = self.up2(x).unsqueeze(2)
        

#         z3 = self.up2(z2.squeeze(2)).unsqueeze(2)
#         z3 = torch.cat([x,z3],dim=1)

#         x = self.conv2(z3)
#         x = self.up3(x).unsqueeze(2)
        
#         z4 = self.up3(z3.squeeze(2)).unsqueeze(2)
#         z4 = torch.cat([x,z4],dim=1)
        
#         x = self.conv3(z4)
        

#         x = self.up4(x).unsqueeze(2)
#         z5 = self.up4(z4.squeeze(2)).unsqueeze(2)
#         z5 = torch.cat([x,z5],dim=1)        

#         x = self.conv4(z5)

# #        x = self.up5(x).unsqueeze(2)
# #        x = self.conv5(x)

#         x = nn.Sigmoid()(x)
        
#         return x
    
#     def forward(self, x):
#         compressed = self.encode(x)
        
        
#         return self.decode(compressed)

        
   
# class SUNET_generator(nn.Module):
#     """Define the Spherical UNet structure

#     """    
#     def __init__(self, latent_dim, inchans):
#         """ Initialize the Spherical UNet.

#         Parameters:
#             in_ch (int) - - input features/channels
#             out_ch (int) - - output features/channels
#         """
#         super(SUNET_generator, self).__init__()

#         #neigh_indices_10242, neigh_indices_2562, neigh_indices_642, neigh_indices_162, neigh_indices_42 = Get_indices_order()
#         #neigh_orders_10242, neigh_orders_2562, neigh_orders_642, neigh_orders_162, neigh_orders_42, neigh_orders_12 = Get_neighs_order()
        
#         neigh_orders = my_mat_Get_2ring_neighs_order()
#         upconv_top_index_40962, upconv_down_index_40962, upconv_top_index_10242, upconv_down_index_10242,  upconv_top_index_2562, upconv_down_index_2562, upconv_top_index_642, upconv_down_index_642, upconv_top_index_162, upconv_down_index_162, upconv_top_index_42, upconv_down_index_42, upconv_top_index_12, upconv_down_index_12 = my_mat_Get_2ring_upconv_index()

#         chs = [32 ,32, 64, 64, 128, 128, 2048]
        
#         self.inchans = inchans
#         self.latent_dim = latent_dim
#         conv_layer = tworing_conv_layer
#         self.l1 = nn.Linear(self.latent_dim, chs[4]*162)
  
# #        self.up1 = up_block_no_skip(conv_layer, chs[5], chs[4], neigh_orders[4], upconv_top_index_162, upconv_down_index_162)
#         self.up1 = up_block_no_skip(conv_layer, chs[4], chs[3], neigh_orders[3], upconv_top_index_642, upconv_down_index_642)
#         self.up2 = up_block_no_skip(conv_layer, chs[3], chs[2], neigh_orders[2], upconv_top_index_2562, upconv_down_index_2562)
#         self.up3 = up_block_no_skip(conv_layer, chs[2], chs[1], neigh_orders[1], upconv_top_index_10242, upconv_down_index_10242)
#         self.up4 = up_block_no_skip(conv_layer, chs[1], chs[0], neigh_orders[0], upconv_top_index_40962, upconv_down_index_40962)
#         self.up5 = down_block(conv_layer, chs[0], inchans, neigh_orders[0], None, True)

                

#         self.middle = chs[4]

    
#     def forward(self, x):

#         x = self.l1(x)


#         x = x.view(162,self.middle, -1)


#         x = self.up1(x)


#         x = self.up2(x)


#         x = self.up3(x)


#         x = self.up4(x)
#         x = self.up5(x)
#         x = nn.Sigmoid()(x)

        
#         return x
    
    

# class SUNET_discriminator(nn.Module):
#     """Define the Spherical UNet structure

#     """    
#     def __init__(self, inchans):
#         """ Initialize the Spherical UNet.

#         Parameters:
#             in_ch (int) - - input features/channels
#             out_ch (int) - - output features/channels
#         """
#         super(SUNET_discriminator, self).__init__()

#         #neigh_indices_10242, neigh_indices_2562, neigh_indices_642, neigh_indices_162, neigh_indices_42 = Get_indices_order()
#         #neigh_orders_10242, neigh_orders_2562, neigh_orders_642, neigh_orders_162, neigh_orders_42, neigh_orders_12 = Get_neighs_order()
        

#         neigh_orders = my_mat_Get_2ring_neighs_order()
#         upconv_top_index_40962, upconv_down_index_40962, upconv_top_index_10242, upconv_down_index_10242,  upconv_top_index_2562, upconv_down_index_2562, upconv_top_index_642, upconv_down_index_642, upconv_top_index_162, upconv_down_index_162, upconv_top_index_42, upconv_down_index_42, upconv_top_index_12, upconv_down_index_12 = my_mat_Get_2ring_upconv_index()

#         chs = [16,32 ,32, 64, 64, 128, 128, 2048]
        
#         conv_layer = tworing_conv_layer
#         self.inchans = inchans
#         self.down1 = down_block(conv_layer, self.inchans, chs[0], neigh_orders[0], None, True)
#         self.down2 = down_block(conv_layer, chs[0], chs[1], neigh_orders[1], neigh_orders[0])
#         self.down3 = down_block(conv_layer, chs[1], chs[2], neigh_orders[2], neigh_orders[1])
#         self.down4 = down_block(conv_layer, chs[2], chs[3], neigh_orders[3], neigh_orders[2])
#         self.down5 = down_block(conv_layer, chs[3], chs[4], neigh_orders[4], neigh_orders[3])

    
#         self.l2 = nn.Sequential( nn.Linear(162*chs[4], 1), nn.Sigmoid() )
#         self.dropout1 = nn.Dropout(0.25)
#         self.dropout2 = nn.Dropout(0.25)
#         self.dropout3 = nn.Dropout(0.25)
#         self.dropout4 = nn.Dropout(0.25)
#         self.dropout5 = nn.Dropout(0.25)

#     def forward(self, x):
#         x = self.down1(x)
#         x = self.dropout1(x)

#         x = self.down2(x)
#         x = self.dropout2(x)

#         x = self.down3(x)
#         x = self.dropout3(x)

#         x = self.down4(x)
#         x = self.dropout4(x)

#         x = self.down5(x)
#         x = self.dropout5(x)

#         x = x.permute(2,0,1)

#         x = x.reshape(x.shape[0], -1)
        
        
        
#         out = self.l2(x)
#         return out
    

   