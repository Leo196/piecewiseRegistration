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
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl

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

def figureshow(arry_MR, arry_LS, SLIDE, label=False, cmap='gray', plane='sagittal'):
    if label == False:
        if plane == 'sagittal':
            mrshow = arry_MR[SLIDE,:,:].copy()
            lsshow = arry_LS[SLIDE,:,:].copy()
            mrshow = mrshow.transpose((1,0))
            lsshow = lsshow.transpose((1,0))
            plt.imshow(np.hstack((mrshow, lsshow)),cmap) 
            plt.axis('off')
            plt.show() 
        elif plane == 'axial':
            mrshow = arry_MR[:,:,SLIDE].copy()
            lsshow = arry_LS[:,:,SLIDE].copy()
            mrshow = mrshow.transpose((1,0))
            lsshow = lsshow.transpose((1,0))
            plt.imshow(np.hstack((mrshow, lsshow)),cmap) 
            plt.axis('off')
            plt.show() 
    elif label == True:
        if plane == 'sagittal':
            labelMRshow = arry_MR[SLIDE,:,:].copy()
            labelMRshow = labelMRshow.transpose(1,0)
            labelLSshow = arry_LS[SLIDE,:,:].copy()
            labelLSshow = labelLSshow.transpose(1,0)
            mat = plt.imshow(np.hstack((labelMRshow, labelLSshow)),cmap) 
            plt.axis('off') 
            ax = plt.gca()
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            unique_data = np.unique(arry_MR);
            plt.colorbar(mat, cax = cax, ticks=np.arange(np.min(unique_data),np.max(unique_data)+1), spacing='uniform')
        elif plane == 'axial':
            labelMRshow = arry_MR[:,:,SLIDE].copy()
            labelMRshow = labelMRshow.transpose(1,0)
            labelLSshow = arry_LS[:,:,SLIDE].copy()
            labelLSshow = labelLSshow.transpose(1,0)
            mat = plt.imshow(np.hstack((labelMRshow, labelLSshow)),cmap) 
            plt.axis('off') 
            ax = plt.gca()
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            unique_data = np.unique(arry_MR);
            plt.colorbar(mat, cax = cax, ticks=np.arange(np.min(unique_data),np.max(unique_data)+1), spacing='uniform')


def fieldshow(field, SLIDE, cmap='viridis', plane='sagittal',vmin=-50,vmax=20,WIDTH=256):
    if plane == 'sagittal':
        show_fieldz = field[SLIDE,:,:,0].copy()
        show_fieldz = np.fliplr(show_fieldz)
        show_fieldz = show_fieldz.transpose((1,0))
        show_fieldx = field[SLIDE,:,:,1].copy()
        show_fieldx = np.fliplr(show_fieldx)
        show_fieldx = show_fieldx.transpose((1,0))
        show_fieldy = field[SLIDE,:,:,2].copy()
        show_fieldy = np.fliplr(show_fieldy)
        show_fieldy = show_fieldy.transpose((1,0))
        plt.ylim(0, WIDTH-1)
        mat = plt.imshow(np.hstack((show_fieldz,show_fieldx,show_fieldy)),cmap=cmap,vmin=vmin,vmax=vmax) 
        plt.axis('off') 
        ax = plt.gca()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(mat,cax=cax,spacing='uniform')
    elif plane == 'axial':
        show_fieldz = field[:,:,SLIDE,0].copy()
        show_fieldz = show_fieldz.transpose((1,0))
        show_fieldy = field[:,:,SLIDE,1].copy()
        show_fieldy = show_fieldy.transpose((1,0))
        show_fieldx = field[:,:,SLIDE,2].copy()
        show_fieldx = show_fieldx.transpose((1,0))
        plt.ylim(0, WIDTH-1)
        mat = plt.imshow(np.hstack((show_fieldz,show_fieldy,show_fieldx)),vmin,vmax,cmap) 
        plt.axis('off') 
        ax = plt.gca()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(mat,cax=cax,spacing='uniform')


def maskshow(mask, dilation_maskEdges, SLIDE, cmap='viridis', plane='sagittal',WIDTH=256):
    if plane == 'sagittal':
        visualOriginal = mask[SLIDE,:,:].copy()
        visualOriginal = np.fliplr(visualOriginal)
        visualOriginal = visualOriginal.transpose((1,0))
        visualLaplace = dilation_maskEdges[SLIDE,:,:].copy()
        visualLaplace = np.fliplr(visualLaplace)
        visualLaplace = visualLaplace.transpose((1,0))
        plt.ylim(0, WIDTH-1)
        mat = plt.imshow(np.hstack((visualOriginal, visualLaplace+4)), cmap) 
        plt.axis('off') 
        ax = plt.gca()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        #unique_data = np.unique(visual);
        unique_data = np.array([1, 2, 3, 4, 5], dtype='int8')
        plt.colorbar(mat, cax = cax, ticks=np.arange(np.min(unique_data),np.max(unique_data)+1), spacing='uniform')
    elif plane == 'axial':
        visualOriginal = mask[:,:,SLIDE].copy()
        visualOriginal = visualOriginal.transpose((1,0))
        visualLaplace = dilation_maskEdges[:,:,SLIDE].copy()
        visualLaplace = visualLaplace.transpose((1,0))
        plt.ylim(0, WIDTH-1)
        mat = plt.imshow(np.hstack((visualOriginal, visualLaplace+4)), cmap) 
        plt.axis('off') 
        ax = plt.gca()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        #unique_data = np.unique(visual);
        unique_data = np.array([1, 2, 3, 4, 5], dtype='int8')
        plt.colorbar(mat, cax = cax, ticks=np.arange(np.min(unique_data),np.max(unique_data)+1), spacing='uniform')
   
    
def resultshow(warpImg,SLIDE,WIDTH=256,plane='sagittal'):
    if plane == 'sagittal':
        result = np.fliplr(warpImg[SLIDE,:,:].copy())
        result = result.transpose((1,0))
        plt.ylim(0, WIDTH-1)
        plt.axis('off')
        plt.imshow(result,cmap='gray') 
    elif plane == 'axial':
        result = warpImg[:,:,SLIDE].copy()
        result = result.transpose((1,0))
        plt.ylim(0, WIDTH-1)
        plt.axis('off')
        plt.imshow(result,cmap='gray') 
   