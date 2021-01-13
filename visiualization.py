# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 21:35:31 2020

@author: fangyan
"""
import SimpleITK as sitk
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np

def myshow(img, plane, filename, z =None, title=None, margin=0.05, dpi=80):
    nda = sitk.GetArrayFromImage(img)
    spacing = img.GetSpacing()
    slicer = False
    if nda.ndim == 3:
        # fastest dim, either component or x
        c = nda.shape[-1]
        # the the number of components is 3 or 4 consider it an RGB image
        if not c in (3,4):
            slicer = True
    elif nda.ndim == 4:
        c = nda.shape[-1]
        if not c in (3,4):
            raise Runtime("Unable to show 3D-vector Image")
        # take a z-slice
        slicer = True

    if (slicer):
        if plane == 'axial':
            ysize = nda.shape[1]
            xsize = nda.shape[2]
        elif plane == 'coronal':
            ysize = nda.shape[2]
            xsize = nda.shape[0]
        elif plane == 'sagittal':
            ysize = nda.shape[1]
            xsize = nda.shape[0]
    else:
        ysize = nda.shape[0]
        xsize = nda.shape[1]

    def callback(plane, filename='filename', z=None):
        if plane == 'axial':
            extent = (0, xsize*spacing[1], ysize*spacing[0], 0)
        elif plane == 'coronal':
            extent = (0, xsize*spacing[2], ysize*spacing[1], 0)
        elif plane == 'sagittal':
            extent = (0, ysize*spacing[0], xsize*spacing[1], 0)

        fig = plt.figure(dpi=dpi)
        # Make the axis the right size...
        ax = fig.add_axes([margin, margin, 1 - 2*margin, 1 - 2*margin])
        plt.set_cmap("gray")
        if z is None:
            ax.imshow(nda,extent=extent,interpolation=None)
        else:
            if plane == 'axial': 
                ax.imshow(cv2.flip(nda[z,...],1),extent=extent,interpolation=None)
            elif plane == 'coronal':
                ax.imshow(cv2.flip(nda[:,z,:],1),extent=extent,interpolation=None)
            elif plane == 'sagittal':
                ax.imshow(nda[:,:,z],extent=extent,interpolation=None)

        if title:
            plt.title(title)
        plt.axis('off')
        plt.savefig(filename+".png",bbox_inches='tight',pad_inches=0.0)

    callback(plane, filename, z)

def nii(ndarry, origin, spacing, dire): # ndarry 5*378*256
    ndarry = ndarry.transpose(2,1,0)
#    for i in range(1, ndarry.shape[0]):
#        ndarry[i,:,:] = np.fliplr(ndarry[i,:,:])
    
    sitksave = sitk.GetImageFromArray(ndarry)
    sitksave.SetOrigin(origin)
    sitksave.SetSpacing(spacing)
    sitksave.SetDirection(dire)
    return sitksave

def niisave(ndarry, origin, spacing, dire, path = '../result/save/save.nii.gz'): # ndarry 5*378*256
    ndarry = ndarry.transpose(2,1,0)
#    for i in range(1, ndarry.shape[0]):
#        ndarry[i,:,:] = np.fliplr(ndarry[i,:,:])
    
    sitksave = sitk.GetImageFromArray(ndarry)
    sitksave.SetOrigin(origin)
    sitksave.SetSpacing(spacing)
    sitksave.SetDirection(dire)
    sitk.WriteImage(sitksave, path)
#def niisave(ndarry, origin, spacing, path = '../result/save/save.nii.gz'): # ndarry 5*378*256
#    ndarry = ndarry.transpose(1,0)
##    for i in range(1, ndarry.shape[0]):
##        ndarry[i,:,:] = np.fliplr(ndarry[i,:,:])
#    
#    sitksave = sitk.GetImageFromArray(ndarry)
#    sitksave.SetOrigin(origin)
#    sitksave.SetSpacing(spacing)
##    sitksave.SetDirection(dire)
#    sitk.WriteImage(sitksave, path)

    