#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 23 20:17:23 2022

@author: fa19
"""
import numpy as np

from math import log10, sqrt
def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = original.max()
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr

def ssim(x,y,C1=1, C2=1):
    """
    x = im1
    y = im2
    
    """
    
    mx = np.mean(x)
    my = np.mean(y)
    
    sx = np.std(x)
    sy = np.std(y)
    N = len(x)
    sxy = np.sum((x - mx) * (y - my))/ (N-1) 
    
    ssim = ((2*mx*my + C1) * (2*sxy + C2) ) / ((mx**2 + my**2 + C1) * (sx**2 + sy**2 + C2))
    
    return ssim


def weighted_ssim(x, y, w, alpha = 1, beta=1, gamma = 1,  C1=1, C2=1, C3=1/2, verbose=False):
    
    """
    x = im1
    y = im2
    
    """
    
    mx = np.sum(w*x)
    my = np.sum(w*y)
    mxy = mx * my

    
    sx = np.sqrt(np.sum(w*(x-mx)**2))
    sy = np.sqrt(np.sum(w*(y-my)**2))
    sxy = np.sum(w*(y-my)*(x-mx))
    # print(mx,my,mxy)
    # print(sx,sy,sxy)

    luminance = (2*mxy + C1) / (mx**2 +my**2 + C1) 

    luminance = luminance**alpha
    luminance = np.nan_to_num(luminance)
    structure = (sxy + C3) / (sx * sy + C3)


    structure = structure**beta
    structure = np.nan_to_num(structure)

    contrast = (2 * sx * sy + C2) / (sx**2 + sy**2 + C2)

    contrast = contrast**gamma
    contrast = np.nan_to_num(contrast)

    ssim = luminance * structure * contrast
    if verbose:
        print(f"Luminance is {luminance}, Structure is {structure}, and contrast is {contrast}.")
    return ssim

def weighted_structure(x, y, w, C3 = 0.02**2):
    
    """
    x = im1
    y = im2
    
    """
    
    mx = np.sum(w*x)
    my = np.sum(w*y)
    mxy = mx * my

    sx = np.sqrt(np.sum(w*(x-mx)**2))
    sy = np.sqrt(np.sum(w*(y-my)**2))
    sxy = np.sum(w*(y-my)*(x-mx))


    
    structure = (sxy + C3) / (sx * sy + C3)
    
    return structure

def weighted_contrast(x, y, w, C2 = 0.02**2):
    
    """
    x = im1
    y = im2
    
    """
    
    mx = np.sum(w*x)
    my = np.sum(w*y)
    mxy = mx * my
    # sx = np.sum( (w * x*x) )- mx**2 
    
    # sy = np.sum( (w * y*y)) - my**2
    
    # sxy = np.sum((w * x*y)) - mxy
    sx = np.sqrt(np.sum(w*(x-mx)**2))
    sy = np.sqrt(np.sum(w*(y-my)**2))



    
    contrast = (2 * sx * sy + C2) / (sx**2 + sy**2 + C2)
    
    return contrast
def weighted_luminance(x, y, w, C2 = 0.02**2):
    
    """
    x = im1
    y = im2
    
    """
    
    mx = np.sum(w*x)
    my = np.sum(w*y)
    mxy = mx * my
    # sx = np.sum( (w * x*x) )- mx**2 
    
    # sy = np.sum( (w * y*y)) - my**2
    
    # sxy = np.sum((w * x*y)) - mxy
    sx = np.sqrt(np.sum(w*(x-mx)**2))
    sy = np.sqrt(np.sum(w*(y-my)**2))



    luminance = (2*mxy + C2) / (mx**2 +my**2 + C2)

    
    return luminance


gaussian_points_arr = np.load('gaussian_arr_points_K5.npy', allow_pickle=True)
gaussian_filters = np.load('gaussian_arr_sigma2_K5.npy', allow_pickle=True)

points_113 = []
arr_113 = []
points_122 = []
arr_122 = []
points_128 = []
arr_128 = []
points_132 = []
arr_132 = []
points_134 = []
arr_134 = []
points_135 = []
arr_135 = []


def fix_numpy_arr(arr):
    L = len(arr[0])
    B = np.zeros([len(arr),L])
    
    for i, row in enumerate(arr):
        B[i]=row
    return B
    

for i, row in enumerate(gaussian_points_arr):
    L = len(row)
    
    if len(row) == 113:
        points_113.append(i)
        arr_113.append(row)
    elif len(row) == 122:
        points_122.append(i)
        arr_122.append(row)
    elif len(row) == 128:
        points_128.append(i)
        arr_128.append(row)
    elif len(row) == 132:
        points_132.append(i)
        arr_132.append(row)
    elif len(row) == 134:
        points_134.append(i)
        arr_134.append(row)
    elif len(row) == 135:
        points_135.append(i)
        arr_135.append(row)
    else:
        print('Error', i, row)

arr_113 = np.array(arr_113)
arr_122 = np.array(arr_122)
arr_128 = np.array(arr_128)
arr_132 = np.array(arr_132)
arr_134 = np.array(arr_134)
arr_135 = np.array(arr_135)



gaussian_113 = fix_numpy_arr(gaussian_filters[points_113])
gaussian_122 = fix_numpy_arr(gaussian_filters[points_122])
gaussian_128 = fix_numpy_arr(gaussian_filters[points_128])
gaussian_132 = fix_numpy_arr(gaussian_filters[points_132])
gaussian_134 = fix_numpy_arr(gaussian_filters[points_134])
gaussian_135 = fix_numpy_arr(gaussian_filters[points_135])


def weighted_ssim_batched(x, y, w, alpha = 1, beta=1, gamma = 1,  C1=0.01, C2=0.01, C3=0.01/2, verbose=False):
    
    """
    x = im1
    y = im2
    
    """
    
    mx = np.sum(np.multiply(w,x), axis=1)
    
    my = np.sum(np.multiply(w,y), axis=1)

    mxy = np.multiply(mx, my)
    # print(mx)
    # print(my)
    # print(mxy)

    
    
    sx = np.sqrt( np.sum(np.multiply(w, np.square(x-mx[:,np.newaxis])),axis=1))
    sy = np.sqrt( np.sum(np.multiply(w, np.square(y-my[:,np.newaxis])),axis=1))

    
    sxy = np.sum( np.multiply(w,np.multiply(y-my[:,np.newaxis], x-mx[:,np.newaxis] )), axis=1) 
    
    
    # sx = np.sqrt(np.sum(w*(x-mx)**2))
    # sy = np.sqrt(np.sum(w*(y-my)**2))
    # sxy = np.sum(w*(y-my)*(x-mx))
    # print(sx)
    # print(sy)
    # print(sxy)



    luminance =np.divide( (2*mxy + C1) ,(np.square(mx) + np.square(my) + C1) )

    luminance = luminance**alpha
    luminance = np.nan_to_num(luminance)
    structure = np.divide((sxy + C3) , (np.multiply(sx, sy) + C3))


    structure = structure**beta
    structure = np.nan_to_num(structure)

    contrast = (2 * np.multiply(sx , sy) + C2) / (np.square(sx) + np.square(sy) + C2)

    contrast = contrast**gamma
    contrast = np.nan_to_num(contrast)

    ssim = np.multiply(np.multiply(luminance , structure ), contrast)
    if verbose:
        print(f"Luminance is {luminance}, Structure is {structure}, and contrast is {contrast}.")
    return ssim

def weighted_ssim_batched(x, y, w,  C1=1, C2=1):
    
    """
    x = im1
    y = im2
    
    """
    
    mx = np.sum(np.multiply(w,x), axis=1)
    mx2 = np.square(mx)
    my = np.sum(np.multiply(w,y), axis=1)
    my2 = np.square(my)
    
    mxy = np.multiply(mx, my)
    # print(mx)
    # print(my)
    # print(mxy)

    
    # sx = np.sum( (w * x*x) )- mx**2 
    
    # sy = np.sum( (w * y*y)) - my**2
    
    # sxy = np.sum((w * x*y)) - mxy
    # sx = np.sqrt( np.sum(np.multiply(w, np.square(x-mx[:,np.newaxis])),axis=1))
    # sy = np.sqrt( np.sum(np.multiply(w, np.square(y-my[:,np.newaxis])),axis=1))

    sx = np.sum(np.multiply(w, np.square(x)),axis=1) -mx2
    sy = np.sum(np.multiply(w, np.square(y)),axis=1) -my2
    
    sxy = np.sum(np.multiply(w, np.multiply(x,y)),axis=1) -mxy
    
    
    # sx = np.sqrt(np.sum(w*(x-mx)**2))
    # sy = np.sqrt(np.sum(w*(y-my)**2))
    # sxy = np.sum(w*(y-my)*(x-mx))
    # print(sx)
    # print(sy)
    # print(sxy)


    ssim  = ((2 * mxy + C1) * (2 * sxy + C2)) / ((mx2 + my2 + C1) *
                                                            (sx + sy + C2))
    return ssim

def normalise_im_style(im, minimum, maximum):
    
    
    if len(im.shape)==1:
        
        # minimum = np.min(im)
        # maximum = np.max(im)
        diff = maximum - minimum
        
        im = im - minimum
        im = im / diff
        return im *255
    
    else:
        # minimum = np.min(im,axis=0)
        # maximum = np.max(im, axis=0)
        diff = maximum - minimum
        
        im = im - minimum
        im = np.divide(im, diff)
        return im*255


def normalise_full(x,y):
    if len(x.shape)==1:
        
        minimum1 = np.min(x)
        maximum1 = np.max(x)
        minimum2 = np.min(y)
        maximum2 = np.max(y)
        minimum = min(minimum1, minimum2)
        maximum = max(maximum1, maximum2)
        
       
    
    else:
        minimum1 = np.min(x,axis=0)
        maximum1 = np.max(x, axis=0)
        minimum2 = np.min(y,axis=0)
        maximum2 = np.max(y, axis=0)

        minimum = np.minimum(minimum1, minimum2)
        maximum = np.maximum(maximum1, maximum2)
    
    x = normalise_im_style(x, minimum, maximum)
    y = normalise_im_style(y, minimum, maximum)
    return x, y
def final_ssim_batched(x,y):
    s = 0   
    # x,y = normalise_full(x,y)
 
    
    if len(x.shape) == len(y.shape) == 1:
        

   
        xi = x[arr_135]
        yi = y[arr_135]
        wi = gaussian_135
        
        S = weighted_ssim_batched(xi, yi, wi)
        s += np.sum(S) # / len(points_135)
        xi = x[arr_113]
        yi = y[arr_113]
        wi = gaussian_113
        
        S = weighted_ssim_batched(xi, yi, wi)
        s += np.sum(S) #/ len(points_113)
        
        xi = x[arr_122]
        yi = y[arr_122]
        wi = gaussian_122
        
        S = weighted_ssim_batched(xi, yi, wi)
        s += np.sum(S) #/ len(points_122)
        
        xi = x[arr_128]
        yi = y[arr_128]
        wi = gaussian_128
        
        S = weighted_ssim_batched(xi, yi, wi)
        s += np.sum(S)# / len(points_128)
        
        xi = x[arr_132]
        yi = y[arr_132]
        wi = gaussian_132
        
        S = weighted_ssim_batched(xi, yi, wi)
        s += np.sum(S)#/len(points_132)
        
        xi = x[arr_134]
        yi = y[arr_134]
        wi = gaussian_134 
        
        S = weighted_ssim_batched(xi, yi, wi)
        s += np.sum(S)#/ len(points_134)
        
    
        
        return s / len(x)
    
    
    else:
        s = 0
        for m in range(x.shape[1]):
            
            xi = x[arr_135,m]
            yi = y[arr_135,m]
            wi = gaussian_135
            
            S = weighted_ssim_batched(xi, yi, wi)
            s += np.sum(S) # / len(points_135)
            xi = x[arr_113,m]
            yi = y[arr_113,m]
            wi = gaussian_113
            
            S = weighted_ssim_batched(xi, yi, wi)
            s += np.sum(S) #/ len(points_113)
            
            xi = x[arr_122,m]
            yi = y[arr_122,m]
            wi = gaussian_122
            
            S = weighted_ssim_batched(xi, yi, wi)
            s += np.sum(S) #/ len(points_122)
            
            xi = x[arr_128,m]
            yi = y[arr_128,m]
            wi = gaussian_128
            
            S = weighted_ssim_batched(xi, yi, wi)
            s += np.sum(S)# / len(points_128)
            
            xi = x[arr_132,m]
            yi = y[arr_132,m]
            wi = gaussian_132
            
            S = weighted_ssim_batched(xi, yi, wi)
            s += np.sum(S)#/len(points_132)
            
            xi = x[arr_134,m]
            yi = y[arr_134,m]
            wi = gaussian_134 
            
            S = weighted_ssim_batched(xi, yi, wi)
            s += np.sum(S)#/ len(points_134)
        
    
        
        
        return s / (x.shape[0]*x.shape[1])
    
    
    
def final_ssim(x,y, verbose=False):
    
    s = 0
    if len(x.shape) == len(y.shape) == 1:
        for i in range(40962):
            points = gaussian_points_arr[i]
            xi = x[points]
            yi = y[points]
            wi = gaussian_filters[i]
            
            s += weighted_ssim(xi, yi, wi, verbose=verbose)
        return s / len(x)

    else:
        for i in range(40962):
            for m in range(x.shape[0]):
                points = gaussian_points_arr[i]
                xi = x[m,points]
                yi = y[m,points]
                wi = gaussian_filters[i]
                
                s += weighted_ssim(xi, yi, wi, verbose=verbose)
        
        return s / (x.shape[0]*x.shape[1])



xy_points = np.load('data/equirectangular_ico_6_points.npy')
xy_points[:,0] = (xy_points[:,0] + 0.1)%1
grid = np.load('data/grid_170_square.npy')


grid_x, grid_y = np.meshgrid(np.linspace(0.02, 0.98, 170), np.linspace(0.02, 0.98, 170))
grid[:,0] = grid_x.flatten()
grid[:,1] = grid_y.flatten()

from scipy.interpolate import griddata
from skimage.metrics import structural_similarity as ssim

def ssim_project(x,y):
    
    # x,y = normalise_full(x, y)
    # x = np.round(x)
    # y = np.round(y)
    x = griddata(xy_points, x, grid, 'nearest')
    x = x.reshape(170,170,2)
    x = np.swapaxes(x, 0,2)
    y = griddata(xy_points, y, grid, 'nearest')
    y = y.reshape(170,170,2)
    y = np.swapaxes(y, 0, 2)

    return ssim(x[0],y[0])+ssim(x[1],y[1])

def crosscor(x,y):
    return float(np.correlate(x[:,0], y[:,0]) + np.correlate(x[:,1], y[:,1]))/len(x.flatten())

def structure_only(x,y):
    
    s = 0
    
    for i in range(40962):
        for m in range(x.shape[0]):
            points = gaussian_points_arr[i]
            xi = x[m,points]
            yi = y[m,points]
            wi = gaussian_filters[i]
            
            s += weighted_structure(xi, yi, wi)
    return s / (x.shape[0]*x.shape[1])

def contrast_only(x,y):
    
    s = 0
    
    for i in range(40962):
        for m in range(x.shape[0]):
            points = gaussian_points_arr[i]
            xi = x[m,points]
            yi = y[m,points]
            wi = gaussian_filters[i]
            
            s += weighted_contrast(xi, yi, wi)
    return s / (x.shape[0]*x.shape[1])

def luminance_only(x,y):
    
    s = 0
    
    for i in range(40962):
        for m in range(x.shape[0]):
            points = gaussian_points_arr[i]
            xi = x[m,points]
            yi = y[m,points]
            wi = gaussian_filters[i]
            
            s += weighted_luminance(xi, yi, wi)
    return s / (x.shape[0]*x.shape[1])

