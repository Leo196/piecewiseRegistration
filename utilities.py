# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 12:38:39 2021

@author: fangyan
"""
import numpy as np
import SimpleITK as sitk
from visiualization import niisave

class dataProperty:
    def __init__(self,path):
        self.path = path
        self.sitk = sitk.ReadImage(self.path)
        self.arry = np.swapaxes(sitk.GetArrayFromImage(self.sitk),0,2)
        self.origin = self.sitk.GetOrigin()
        self.spacing = self.sitk.GetSpacing()
        self.direction = self.sitk.GetDirection()
        self.size = self.sitk.GetSize()
        self.dtype = self.arry.dtype
    def axisConvert(self):
#        LPI TO LAS (intersubject data is LPI)
        orientation = 'LPI'
        if orientation == 'LPI':
            from nipype.interfaces.image import Reorient
            reorient = Reorient(orientation='RPI')
            reorient.inputs.in_file = self.path
            self.res = reorient.run()
        #    res.outputs.out_file


def norm(image):
    """MR-Lightsheet normalization"""
    minImg = np.sort(image.reshape(1,image.shape[0]*image.shape[1]*image.shape[2]))[0][0]
    maxImg = np.sort(image.reshape(1,image.shape[0]*image.shape[1]*image.shape[2]))[0][-100]
    image = (image - minImg) / (maxImg - minImg)
    
    return image

def norm255(image):
    """~->[0,255] for overlay visiualization"""
    
    ymax=255
    ymin=0
    xmax = np.max(image) #求得InImg中的最大值
    xmin = np.min(image) #求得InImg中的最小值
    image = np.round((ymax-ymin)*(image-xmin)/(xmax-xmin) + ymin)
    
    return image

def flip180(arr):
    new_arr = arr.reshape(arr.size)
    new_arr = new_arr[::-1]
    new_arr = new_arr.reshape(arr.shape)
    return new_arr

#def splitImage(img, label, savePath, spacing, origin, dire):
#    """Split images using random walk results"""
#    
#    subImg = np.multiply(img, label)
#    
#    niisave(subImg, origin, spacing, dire, path = savePath)
#    
#    return subImg
def splitImage(img, label, savePath, spacing, origin, dire):
    """Split images using random walk results"""
    
    subImg = np.multiply(img, label)
    
    niisave(subImg, origin, spacing, dire, path = savePath)
#    niisave(subImg, origin, spacing, path = savePath)
    
    return subImg