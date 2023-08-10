#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 18:17:20 2021

@author: fa19
"""

import nibabel as nb
import numpy as np
random_image = nb.load('/home/fa19/Documents/dHCP_Data_merged/merged/CC00589XX21_184000_L.shape.gii')

def save_as_metric(arr, name):

    random_image = nb.load('/home/fa19/Documents/dHCP_Data_merged/merged/CC00589XX21_184000_L.shape.gii')

    for m in range(arr.shape[1]):
         
        I = np.array(arr[:,m]).astype(float) 
        random_image.darrays[m].data = I
        
    nb.save(random_image, name + '.metric.shape.gii')