#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 13:48:33 2022

@author: fa19
"""
import nibabel as nb
import numpy as np
from my_utils import save_as_metric

true_m = nb.load('/home/fa19/Documents/dHCP_Data_merged/merged/CC00714XX13_240900_L.shape.gii')
true_data = true_m.darrays[0].data
true_mean = true_m.darrays[0].data.mean()
true_std = true_m.darrays[0].data.std()
save_loc = '/home/fa19/Documents/neurips/3cycle/'

num = 40
im1 ='CC00714XX13_'+str(num)+'.metric.shape.gii'


image = nb.load(save_loc + im1)

im_data = image.darrays[0].data
# im_data /= true_std
# im_data = im_data + (true_mean - im_data.mean())
im_data = im_data - im_data.min()
im_data = im_data / (im_data.max())
im_data = im_data * true_data.max()

image.darrays[0].data = im_data

# save_as_metric(im_data[:,np.newaxis], '/home/fa19/realigned_myelination_'+str(num))


true_data_s = true_m.darrays[3].data
true_max_s = true_m.darrays[3].data.max()
true_std_s = true_m.darrays[3].data.std()




image = nb.load(save_loc + im1)

im_data2 = image.darrays[1].data

im_data2 = im_data2 * true_max_s / im_data2.max()



empty = np.zeros([40962,2])

empty[:,0] = im_data
empty[:,1] = im_data2


# # im_data /= true_std
# # im_data = im_data + (true_mean - im_data.mean())
# im_data = im_data - im_data.min()
# im_data = im_data / (im_data.max())
# im_data = im_data * true_data.max()

# image.darrays[1].data = im_data


save_as_metric(empty, '/home/fa19/realigned_full_'+str(num))
