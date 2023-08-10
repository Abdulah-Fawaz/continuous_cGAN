#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 10:03:32 2023

@author: fa19
"""

import numpy as np

import matplotlib.pyplot as plt


full_data = np.load('/home/fa19/Documents/Surface-ICAM/data/full/full.npy', allow_pickle=True)

test_data = np.load('/home/fa19/Documents/Surface-ICAM/data/scan_age/test.npy',allow_pickle=True)
orange_bit = full_data[np.logical_and((full_data[:,-1]>=37), full_data[:,1]>=37)]

blue_bit = full_data[np.logical_and((full_data[:,-1]<37), full_data[:,1]<37)]

brown_bit = full_data[np.logical_and((full_data[:,-1]<37), full_data[:,1]>=37)]



msize = 40
mshape='x'
plt.figure()

plt.scatter(blue_bit[:,-1].astype(float), blue_bit[:,1].astype(float), s=msize, marker='x', c='blue', label='preterm first scans')
plt.scatter(orange_bit[:,-1].astype(float), orange_bit[:,1].astype(float), s=msize, marker='x', c='orange', label = 'term scans')
plt.scatter(brown_bit[:,-1].astype(float), brown_bit[:,1].astype(float), s=msize, marker='x', c='brown', label ='preterm second scans')
plt.xlabel('Gestational Age at Birth (GA / weeks)')
plt.ylabel('PostMenstrual Age at Scan (PMA / weeks)')
#plt.title('A plot to Show the Distribution of Images in our Dataset')
plt.legend()
plt.show()


