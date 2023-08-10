#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 29 11:51:34 2023

@author: fa19
"""
import os
import nibabel as nb
import numpy as np

dmri_means = np.array([0.10292342, 0.0010362, 0.08419238, 0.15764983])
dmri_stds = np.array([0.05, 0.0003837, 0.0377524, 0.09069089])


data_dir = '/data/dmri1_images/'
save_dir = '/data/dmri1_images_final/'

for file in os.listdir(data_dir):
    
    fullfile = data_dir + file
    
    im = nb.load(fullfile)
    
    for i in range(4):
        d = im.darrays[i].data
        
        d *= dmri_stds[i]
        d += dmri_means[i]
        im.darrays[i].data = d
    nb.save(im, save_dir + file)
        