#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  4 01:16:51 2022

@author: fa19
# """
# import os
# import nibabel as nb
# import numpy as np
# results_dir = '/home/fa19/Documents/neurips/cyclegan_results/'

# list_of_files = os.listdir(results_dir)



# for filename in list_of_files:
#     if 'metric' in filename:
        
        
#         file = nb.load(results_dir + filename)
#         F = file.darrays[0].data
#         F = (F - np.mean(F)) / np.std(F)
        
#         file.darrays[0].data = F

#         F = file.darrays[1].data
#         F = (F - np.mean(F)) / np.std(F)
        
#         file.darrays[1].data = F        
        
#         nb.save(file, results_dir + filename)
        
