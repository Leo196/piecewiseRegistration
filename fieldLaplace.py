# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 17:47:17 2020

@author: fangyan
"""
import numpy as np
import scipy.ndimage
import cv2
from skimage.color import rgb2gray
from skimage import feature
import skimage.morphology
import gc

def Laplace(fieldSingleDirection, thresh = 10):
    """need to double-check: how to set thresh"""
    
    # iteration maxium
    ITERMax = 300
    itera = 0
    
    # initial conditions
    p_curr = fieldSingleDirection
    eps_curr = 1000
        
    # kernal for averaging neighbors
    H = np.zeros(3 * 3).reshape(3, 3)
    H[2, 1] = 1/4
    H[0, 1] = 1/4
    H[1, 2] = 1/4
    H[1, 0] = 1/4

    stopflag = 0
    
    while stopflag == 0 and (itera < ITERMax):
        # update iterations
        itera = itera + 1;
        p_prev = p_curr
        eps_prev = eps_curr
        
        p_curr = np.swapaxes(scipy.ndimage.correlate(p_prev, H, mode='constant').transpose(),0,1) #  p_curr = imfilter(p_prev, H)
    
        if itera % 20 == 1:
            sobelx = cv2.Sobel(p_curr, cv2.CV_64F, 1, 0) # Find x and y gradients
            sobely = cv2.Sobel(p_curr, cv2.CV_64F, 0, 1)
            eps_curr = np.sum(np.sqrt(sobelx**2 + sobely**2))

            if abs(eps_prev - eps_curr) < thresh:
                stopflag = 1
    print(itera)
    return p_curr


def Laplace3D(fieldSingleDirection, thresh = 10): # fieldSingleDirection: 5*378*256
    """need to double-check: how to set thresh"""
    
    # iteration maxium
    ITERMax = 300
    itera = 0
    
    # initial conditions
    p_curr = fieldSingleDirection
    eps_curr = 1000
        
    # kernal for averaging neighbors
    H = np.zeros(3 * 3 *3).reshape(3, 3, 3)
    H[2, 1, 1] = 1/6
    H[0, 1, 1] = 1/6
    H[1, 2, 1] = 1/6
    H[1, 0, 1] = 1/6
    H[1, 1, 2] = 1/6
    H[1, 1, 0] = 1/6

    stopflag = 0
    
    while stopflag == 0 and (itera < ITERMax):
        # update iterations
        itera = itera + 1;
        p_prev = p_curr
        eps_prev = eps_curr
        
        p_curr = scipy.ndimage.correlate(p_prev, H, mode='constant') #  p_curr = imfilter(p_prev, H)
    
        if itera % 20 == 1:
            sobelx = scipy.ndimage.sobel(p_curr, 0)
            sobely = scipy.ndimage.sobel(p_curr, 1) # Find x and y gradients
            sobelz = scipy.ndimage.sobel(p_curr, 2)
            
            eps_curr = np.sum(np.sqrt(sobelx**2 + sobely**2 + sobelz**2))

            if abs(eps_prev - eps_curr) < thresh:
                stopflag = 1
    return p_curr



def fieldLaplace(field):
    
    fieldX = field[0,:,:]
    fieldLaplaceX = Laplace(fieldX)
    fieldY = field[1,:,:]
    fieldLaplaceY = Laplace(fieldY)
    
    field = np.stack((fieldLaplaceX, fieldLaplaceY))
    
    return field

def fieldLaplace3D(field):
    
    fieldZ = field[:,:,:,0]
    fieldLaplaceZ = Laplace3D(fieldZ)
    fieldX = field[:,:,:,1]
    fieldLaplaceX = Laplace3D(fieldX)
    fieldY = field[:,:,:,2]
    fieldLaplaceY = Laplace3D(fieldY)
    
    field = np.stack((fieldLaplaceZ, fieldLaplaceX, fieldLaplaceY), axis=0)
    
    # release memory
    del fieldZ, fieldX, fieldY, fieldLaplaceZ, fieldLaplaceX, fieldLaplaceY
    gc.collect()
    
    return field.transpose(1,2,3,0)


def LaplaceMask(fieldSingleDirection, mask_interior, thresh = 0):
    """Laplace function with mask"""
    interior = mask_interior > 0 # float => bool
    
    # iteration maxium
    ITERMax = 300
    itera = 0
    
    # initial conditions
    p_curr = fieldSingleDirection.copy()
    eps_curr = 1000
    mask_bnd = (1- mask_interior)>0
        
    # kernal for averaging neighbors
    H = np.zeros(3 * 3).reshape(3, 3)
    H[2, 1] = 1/4
    H[0, 1] = 1/4
    H[1, 2] = 1/4
    H[1, 0] = 1/4
    
    stopflag = 0
    
    while stopflag == 0 and (itera < ITERMax):
        # update iterations
        itera = itera + 1;
        p_prev = p_curr
        eps_prev = eps_curr
                
        p_curr = scipy.ndimage.correlate(p_prev, H, mode='constant') #  p_curr = imfilter(p_prev, H)
        p_curr[mask_bnd]  = fieldSingleDirection[mask_bnd] 
    
        if itera % 20 == 1:
            sobelx = cv2.Sobel(p_curr, cv2.CV_64F, 1, 0) # Find x and y gradients
            sobely = cv2.Sobel(p_curr, cv2.CV_64F, 0, 1)
    
#            eps_curr = np.sum(np.sqrt(sobelx[mask_interior]**2 + sobely[mask_interior]**2))
#            eps_curr = np.sum(np.sqrt(sobelx[(1-mask_interior)>0]**2 + sobely[(1-mask_interior)>0]**2))
            eps_curr = np.sum(np.sqrt(sobelx[interior]**2 + sobely[interior]**2))
            print(abs(eps_prev - eps_curr))

            if abs(eps_prev - eps_curr) < thresh:
                stopflag = 1
    
    return p_curr


def LaplaceMask3D(fieldSingleDirection, mask_interior, thresh = 0): # fieldSingleDirection: 5*378*256*3, mask_interior: 5*378*256
    """Laplace function with mask"""
    interior = mask_interior > 0 # float => bool
    
    # iteration maxium
    ITERMax = 300
    itera = 0
    
    # initial conditions
    p_curr = fieldSingleDirection.copy()
    eps_curr = 1000
    mask_bnd = (1- mask_interior)>0
        
    # kernal for averaging neighbors
    H = np.zeros(3 * 3 *3).reshape(3, 3, 3)
    H[2, 1, 1] = 1/6
    H[0, 1, 1] = 1/6
    H[1, 2, 1] = 1/6
    H[1, 0, 1] = 1/6
    H[1, 1, 2] = 1/6
    H[1, 1, 0] = 1/6

    stopflag = 0
    
    while stopflag == 0 and (itera < ITERMax):
        # update iterations
        itera = itera + 1;
        p_prev = p_curr
        eps_prev = eps_curr
                
        p_curr = scipy.ndimage.correlate(p_prev, H, mode='nearest') #  p_curr = imfilter(p_prev, H)
        p_curr[mask_bnd]  = fieldSingleDirection[mask_bnd] 
    
        if itera % 20 == 1:
            sobelx = scipy.ndimage.sobel(p_curr, 0)
            sobely = scipy.ndimage.sobel(p_curr, 1) # Find x and y gradients
            sobelz = scipy.ndimage.sobel(p_curr, 2)
    
#            eps_curr = np.sum(np.sqrt(sobelx[mask_interior]**2 + sobely[mask_interior]**2))
#            eps_curr = np.sum(np.sqrt(sobelx[(1-mask_interior)>0]**2 + sobely[(1-mask_interior)>0]**2))
            eps_curr = np.sum(np.sqrt(sobelx[interior]**2 + sobely[interior]**2 + sobelz[interior]**2))

            if abs(eps_prev - eps_curr) < thresh:
                stopflag = 1
    
    return p_curr


def fieldMaskLaplace(field, copy_labels_MR, dilationSigma = 10):
    
    img_gray = rgb2gray(copy_labels_MR) # to gray
    edges = feature.canny(img_gray, sigma=0.5, low_threshold=1, high_threshold=3)
    arr_edges = np.array(edges, dtype='int8')
    
#    kernel = skimage.morphology.disk(dilationSigma)
    kernel = skimage.morphology.square(dilationSigma)
    dilation_maskEdges = skimage.morphology.dilation(arr_edges, kernel)
        
    mask = 1-dilation_maskEdges   
    fieldMaskX = field[0,:,:] * mask.swapaxes(0, 1)
    fieldLaplacMaskX = LaplaceMask(fieldMaskX, dilation_maskEdges.swapaxes(0, 1), thresh = 0.1)
    fieldMaskY = field[1,:,:] * mask.swapaxes(0, 1)
    fieldLaplacMaskY = LaplaceMask(fieldMaskY, dilation_maskEdges.swapaxes(0, 1), thresh = 0.1)

    field = np.stack((fieldLaplacMaskX, fieldLaplacMaskY))
    
    return field, dilation_maskEdges


def fieldMaskLaplace3D(field, copy_labels_MR, dilationSigma = 10): # field: 5*378*256*3; copy_labels_MR: 5*378*256

    dilation_maskEdges = np.zeros_like(copy_labels_MR)
    for i in range(0, copy_labels_MR.shape[0]):
        img_gray = rgb2gray(copy_labels_MR[i,:,:]) # to gray
        edges = feature.canny(img_gray, sigma=0.5, low_threshold=1, high_threshold=3)
        arr_edges = np.array(edges, dtype='int8')
    
    #    kernel = skimage.morphology.disk(dilationSigma)
        kernel = skimage.morphology.square(dilationSigma)
        dilation_maskEdges[i,:,:] = skimage.morphology.dilation(arr_edges, kernel)
        
    masktemp = 1-dilation_maskEdges  
    fieldMaskZ = field[:,:,:,0] * masktemp
    fieldLaplacMaskZ = LaplaceMask3D(fieldMaskZ, dilation_maskEdges, thresh = 0.1)
    fieldMaskX = field[:,:,:,1] * masktemp
    fieldLaplacMaskX = LaplaceMask3D(fieldMaskX, dilation_maskEdges, thresh = 0.1)
    fieldMaskY = field[:,:,:,2] * masktemp
    fieldLaplacMaskY = LaplaceMask3D(fieldMaskY, dilation_maskEdges, thresh = 0.1)

    field = np.stack((fieldLaplacMaskZ, fieldLaplacMaskX, fieldLaplacMaskY), axis=0)
    
    # release memory
    del fieldMaskZ, fieldMaskX, fieldMaskY, fieldLaplacMaskZ, fieldLaplacMaskX, fieldLaplacMaskY
    gc.collect()
    
    return field.transpose(1,2,3,0), dilation_maskEdges