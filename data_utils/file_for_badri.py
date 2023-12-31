#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 14:29:19 2021

@author: fa19
"""



import torch
import numpy as np


train_set = np.load('../data/scan_age/test.npy', allow_pickle=True)
"""

train_set example:
    
    
array([['CC00749XX24_2600', 44.0],
       ['CC00818XX18_4020', 42.57],
       ['CC00815XX15_4120', 42.57],
       ['CC00766XX16_4200', 42.0],
       ['CC00843XX19_4330', 40.71],
       ['CC00850XX09_4930', 40.86],
       ['CC00860XX11_7030', 40.71],
       ['CC00879XX22_7430', 43.71],
       ['CC00051XX02_7702', 40.0],
       ['CC00765XX15_8210', 42.14]], dtype=object)
    
"""



class My_dHCP_Data(torch.utils.data.Dataset):

    def __init__(self, input_arr, warped_files_directory, unwarped_files_directory,  rotations = False,
                 number_of_warps = 0, parity_choice = 'left', smoothing = False, normalisation = None, projected =False, sample_only = True, output_as_torch = True, *args):
        
        """
        
        A Full Dataset for the dHCP Data. Can include warps, rotations and parity flips.
        
        Fileanme style:
            
            in the array: only 'sub-X-ses-Y'
            but for the filenames themselves
                Left = 'sub-X_ses-Y_L'
                Right = 'sub-X_ses-Y_R'
                if warped:
                    'sub-X_ses-Y_L_W1'
        
        INPUT ARGS:
        
            1. input_arr:
                Numpy array size Nx2 
                FIRST index MUST be the filename (excluding directory AND L or R ) of MERGED nibabel files
                LAST index must be the (float) label 
                (OPTIONAL) Middle index if size 3 (optional) is any confounding metadata (Also float, e.g scan age for predicting birth age)
        
                        
            2 . rotations - boolean: to add rotations or not to add rotations              
            
            3. number of warps to include - INT
                NB WARPED AR INCLUDED AS FILENAME CHANGES. WARP NUMBER X IS WRITTEN AS filename_WX
                NUMBER OF WARPS CANNOT EXCEED NUMBER OF WARPES PRESENT IN FILES 
                
            4. Particy Choice (JMPORTANT!) - defines left and right-ness
            
                If: 'left'- will output ONLY LEFT 
                If: 'both' - will randomly choose L or R
                If 'combined' - will output a combined array (left first), will be eventually read as a file with twice the number of input channels. as they will be stacked together
                
            5. smoothing - boolean, will clip extremal values according to the smoothing_array 
            
            6. normalisation - str. Will normalise according to 'range', 'std' or 'None'
                Range is from -1 to 1
                Std is mean = 0, std = 1
                
            7. output_as_torch - boolean:
                outputs values as torch Tensors if you want (usually yes)
                
                
        """
        
        
        
        
        self.input_arr = input_arr
        
        self.image_files = input_arr[:,0]
        self.label = input_arr[:,-1]
        
            
        self.rotations = rotations
                
        self.projected = projected
    
        self.number_of_warps = number_of_warps
        
        self.parity = parity_choice

        self.smoothing = smoothing
        self.normalisation = normalisation
        self.sample_only = sample_only
        
        self.output_as_torch = output_as_torch
        if self.number_of_warps != 0 and self.number_of_warps != None:
            self.directory = warped_files_directory
        else:
            self.directory = unwarped_files_directory
            
    def __len__(self):
        
        L = len(self.input_arr)
        
        
        if self.number_of_warps !=0:
            if self.sample_only == False:
                L = L*self.number_of_warps
                
        if self.parity == 'both':
            L = 2* L
            
        return L
    
    
    def __test_input_params__(self):
        assert self.input_arr.shape[1] >=2, 'check your input array is a nunpy array of files and labels'
        assert type(self.number_of_warps) == int, "number of warps must be an in integer (can be 0)"
        assert self.parity in ['left', 'both', 'combined'], "parity choice must be either left, combined or both"
        if self.number_of_rotations != 0:
            assert self.rotation_arr != None,'Must specify a rotation file containing rotation vertex ids if rotations are non-zero'       
        assert self.rotations == bool, 'rotations must be boolean'
        assert self.normalisation in [None, 'none', 'std', 'range'], 'Normalisation must be either std or range'
        
    
    def __genfilename__(self,idx, right):
        
        """
        gets the appropriate file based on input parameters on PARITY and on WARPS
        
        """
        # grab raw filename
 
        raw_filename = self.image_files[idx]
        
        # add parity to it. IN THE FORM OF A LIST!  If requries both will output a list of length 2
        filename = []
        
        if self.parity != 'combined':
            if right == True:
                filename.append(raw_filename + '_R')
                
            else:
                
               filename.append(raw_filename + '_L')
           
           
#        if self.parity == 'left':
#            filename.append(raw_filename + '_L')
#            
#        elif self.parity == 'both':
#            coin_flip = random.randint(0,1)
#            if coin_flip == 0:
#                filename.append(raw_filename + '_L')
#            elif coin_flip == 1:
#                filename.append(raw_filename + '_R')
#                right = True
           
          
        if self.parity == 'combined':
            filename.append(raw_filename + '_L')
            filename.append(raw_filename+'_R')
            
        # filename is now a list of the correct filenames.
        
        # now add warps if required
        
        if self.number_of_warps != 0: 
            warp_choice = str(random.randint(0,self.number_of_warps))
            if warp_choice !='0':
                
                filename = [s + '_W'+warp_choice for s in filename ]


        return filename
                
        
            
            
    def __getitem__(self, idx):
        
        """
        First load the images and collect them as numpy arrays
        
        then collect the label
        
        then collect the metadata (though might be None)
        """
        
        if self.parity == 'both':
            T = self.__len__()//2

            idx, right  = idx % T, idx // T
            filename = self.__genfilename__(idx, right)
        else:
            right = False
            filename = self.__genfilename__(idx, right)        
        
        
        
        image_gifti = [nb.load(self.directory + '/'+individual_filename+'.shape.gii').darrays for individual_filename in filename]

        image = []
        if self.rotations == True:
            
            rotation_choice = random.randint(0, len(rotation_arr)-1)
            if rotation_choice !=0:
                for file in image_gifti:
                    image.extend(item.data[rotation_arr[rotation_choice]] for item in file) 
            else:
                for file in image_gifti:
                    image.extend(item.data for item in file)
        else:
            for file in image_gifti:
                image.extend(item.data for item in file)
        

        
        if right == True:
            image = [item[reversing_arr] for item in image]
        
        
        
        ### labels
#        if self.number_of_warps != 0:
#            
#            idx = idx%len(self.input_arr)
#        label = self.label[idx]

        
        ###### metadata grabbing if necessary
        
        label = self.label[idx]
        if self.input_arr.shape[1] > 2:
            
            self.metadata = self.input_arr[:,1:-1]

            metadata = self.metadata[idx]

        else:
            metadata = None
            
        
        if self.smoothing != False:
            for i in range(len(image)):
                image[i] = np.clip(image[i], lower_bound[i%len(lower_bound)].item(), upper_bound[i%len(upper_bound)].item())
                
            
        # torchify if required:
        
        
        if self.normalisation != None:
            if self.normalisation == 'std':
                for i in range(len(image)):
                    
                    image[i] = ( image[i] - means[i%len(means)].item( )) / stds[i%len(stds)].item()
            
            elif self.normalisation == 'range':
                for i in range(len(image)):
                    
                    image[i] = image[i] - minima[i%len(minima)].item()
                    image[i] = image[i] / (maxima[i%len(maxima)].item()- minima[i%len(minima)].item())
            
        if self.output_as_torch:
            image = torch.Tensor( image )

            label = torch.Tensor( [label] )

            if isinstance(metadata,np.ndarray):
                
                metadata = torch.Tensor( [metadata] ).squeeze(1)
         
                
        if self.projected == True:
            image = griddata(xy_points, image.T, grid, 'nearest')
            image = torch.Tensor(image.reshape(170,170,4)).permute(2,0,1)
            
                
        if hasattr(metadata,'shape'):

            sample = {'image': image, 'metadata' : metadata, 'label': label}
        
        else:
            sample = {'image': image,'label': label}

        return sample
