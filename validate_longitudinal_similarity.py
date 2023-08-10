#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  2 13:14:24 2023

@author: fa19
"""

import numpy as np
import os 
import nibabel as nb
source_dir = '/home/fa19/Documents/neurips/3cycle_longitudinal_myelin_sulc/'
full_list = os.listdir(source_dir)

second_scan_list = [i for i in full_list if 'secondscan' in i]

generated_list = [ i for i in full_list if 'original' not in i]


def load_im(imgname):
    loaded = nb.load(source_dir + imgname)
    X = np.zeros([40962,2])
    X[:,0] = loaded.darrays[0].data
    X[:,1] = loaded.darrays[1].data
    return X



x = load_im(generated_list[0])

differences = []
for idx in second_scan_list:
    other_images_mse = []
    matching_image_mse = []
    second_scan_arr = load_im(idx)
    subj = 'CC'+ idx.split('CC')[1].split('.metric')[0]
    age = float(idx.split('secondscan_')[1].split('_')[0])
    age = str(int(age))
    
    subject_generated_list = [i for i in generated_list if subj in i]
    age_generated_list = [i for i in generated_list if age in i]
    
    for gen_img in age_generated_list:
        generated_arr = load_im(gen_img)
        mse_ = np.linalg.norm(generated_arr.flatten() - second_scan_arr.flatten())
        if subj in gen_img:
            matching_image_mse.append(mse_)
        else:
            other_images_mse.append(mse_)
    diff = min(other_images_mse) - min(matching_image_mse)
    differences.append(diff)
        
    