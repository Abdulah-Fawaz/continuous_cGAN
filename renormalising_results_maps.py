#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 10:03:16 2023

@author: fa19
"""


import nibabel as nb

import numpy as np
import torch
means = torch.Tensor([1.1267, 0.0345, 1.0176, 0.0556]).numpy()
stds = torch.Tensor([0.3522, 0.1906, 0.3844, 4.0476]).numpy()




sub = 'CC01006XX08'
# ['CC00597XX21
#        ['CC00594XX18 '
#        ['CC00622XX12

       
dir1 = '/home/fa19/Documents/neurips/full_3cycle_ba_only2/'
dir2 = '/home/fa19/Desktop/ba_maps/'

for num in [32, 36, 40]:
    try:
        file = f'{dir1}{sub}_new_ba={num}.0.metric.shape.gii'
    
        loaded = nb.load(file)
        myelin = loaded.darrays[0].data
        # print(np.mean(myelin))
        myelin *= stds[0]
        myelin  += means[0]
        # print(np.mean(myelin))
        
        sulc = loaded.darrays[1].data
        # print(np.mean(sulc))
        sulc *= stds[3]
        sulc  += means[3]
        # print(np.mean(sulc))
        # for m in range(0):
        #     loaded.darrays[m].data *= stds[m]
        #     loaded.darrays[m].data += means[m]
        # new_file_name = f'{dir2}{sub}_{num}.metric.shape.gii'
        # nb.save(loaded, new_file_name)
        
        loaded.darrays[0].data = myelin
        loaded.darrays[1].data = sulc
        new_file_name = f'{dir2}{sub}_{num}.metric.shape.gii'
        nb.save(loaded, new_file_name)
    except:
        continue
