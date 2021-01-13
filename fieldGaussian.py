# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 17:55:54 2020

@author: fangyan
"""
import numpy as np
import SimpleITK as sitk
from skimage.color import rgb2gray
from skimage import feature
import skimage.morphology
import gc

def Gaussian(fieldMask, sigmaGaussian = 8):
    
    image = sitk.GetImageFromArray(fieldMask)
    pixelID = image.GetPixelID()
    gaussian = sitk.SmoothingRecursiveGaussianImageFilter()
    gaussian.SetSigma(sigmaGaussian)
    image = gaussian.Execute(image)
    caster = sitk.CastImageFilter()
    caster.SetOutputPixelType(pixelID)
    image = caster.Execute(image)
    fieldGaussianMask = sitk.GetArrayFromImage(image)
    
    # release memory
    del image, gaussian, caster
    gc.collect()
    
    return fieldGaussianMask


def fieldMaskGaussian(field, copy_labels_MR, dilationSigma = 10):
    
    img_gray = rgb2gray(copy_labels_MR) # to gray
    edges = feature.canny(img_gray, sigma=0.5, low_threshold=1, high_threshold=3)
    arr_edges = np.array(edges, dtype='int8')
    
#    kernel = skimage.morphology.disk(dilationSigma)
    kernel = skimage.morphology.square(dilationSigma)
    dilation_maskEdges = skimage.morphology.dilation(arr_edges, kernel)
    
    fieldMaskX = field[0,:,:] * dilation_maskEdges.swapaxes(0, 1)
    fieldGaussianMaskX = Gaussian(fieldMaskX)
    fieldMaskY = field[1,:,:] * dilation_maskEdges.swapaxes(0, 1)
    fieldGaussianMaskY = Gaussian(fieldMaskY)
    
    GaussianMask = (1 - dilation_maskEdges).copy()
    fiedlGaussianX = GaussianMask.swapaxes(0, 1) * field[0,:,:] + fieldGaussianMaskX
    fiedlGaussianY = GaussianMask.swapaxes(0, 1) * field[1,:,:] + fieldGaussianMaskY
    
    field = np.stack((fiedlGaussianX, fiedlGaussianY))
    
    return field, dilation_maskEdges


def fieldMaskGaussian3D(field, copy_labels_MR, dilationSigma = 10): #field: 5*378*256*3  copy_labels_MR: 5*378*256
    
    dilation_maskEdges = np.zeros_like(copy_labels_MR)
    for i in range(0, copy_labels_MR.shape[0]):
        img_gray = rgb2gray(copy_labels_MR[i,:,:]) # to gray
        edges = feature.canny(img_gray, sigma=0.5, low_threshold=1, high_threshold=3)
        arr_edges = np.array(edges, dtype='int8')
    
    #    kernel = skimage.morphology.disk(dilationSigma)
        kernel = skimage.morphology.square(dilationSigma)
        dilation_maskEdges[i,:,:] = skimage.morphology.dilation(arr_edges, kernel)
        
    fieldMaskZ = field[:,:,:,0] * dilation_maskEdges
    fieldGaussianMaskZ = Gaussian(field[:,:,:,0])
    fieldMaskX = field[:,:,:,1] * dilation_maskEdges
    fieldGaussianMaskX = Gaussian(field[:,:,:,1])
    fieldMaskY = field[:,:,:,2] * dilation_maskEdges
    fieldGaussianMaskY = Gaussian(field[:,:,:,2])
    
    GaussianMask = (1-dilation_maskEdges).copy()
    fiedlGaussianZ = GaussianMask * fieldGaussianMaskZ + fieldMaskZ
    fiedlGaussianX = GaussianMask * fieldGaussianMaskX + fieldMaskX
    fiedlGaussianY = GaussianMask * fieldGaussianMaskY + fieldMaskY
    
    field = np.stack((fiedlGaussianZ, fiedlGaussianX, fiedlGaussianY), axis = 0)
    
    # release memory
    del GaussianMask, fiedlGaussianZ, fiedlGaussianX, fiedlGaussianY, arr_edges
    gc.collect()
    
    return field.transpose(1,2,3,0), dilation_maskEdges
    

def fieldGaussian(field):
    
    fieldX = field[0,:,:]
    fieldGaussianMaskX = Gaussian(fieldX)
    fieldY = field[1,:,:]
    fieldGaussianMaskY = Gaussian(fieldY)
    
    field = np.stack((fieldGaussianMaskX, fieldGaussianMaskY))
        
    return field


def fieldGaussian3D(field):
    
    fieldZ = field[:,:,:,0]
    fieldGaussianMaskZ = Gaussian(fieldZ)
    fieldX = field[:,:,:,1]
    fieldGaussianMaskX = Gaussian(fieldX)
    fieldY = field[:,:,:,2]
    fieldGaussianMaskY = Gaussian(fieldY)
    
    field = np.stack((fieldGaussianMaskZ, fieldGaussianMaskX, fieldGaussianMaskY), axis=0)
    
    # release memory
    del fieldZ, fieldX, fieldY, fieldGaussianMaskZ, fieldGaussianMaskX, fieldGaussianMaskY
    gc.collect()
    
    return field.transpose(1,2,3,0)