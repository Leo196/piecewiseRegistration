# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 17:02:31 2020

@author: fangyan

preprocessing

normalization


"""
import numpy as np
import SimpleITK as sitk
import cv2
from visiualization import niisave
import skimage
import scipy


def norm(image):
    """MR-Lightsheet normalization"""
    
    image = (image - np.min(image)) / (np.max(image) - np.min(image))
    
    return image

def norm255(image):
    """~->[0,255] for overlay visiualization"""
    
    ymax=255
    ymin=0
    xmax = np.max(image) #求得InImg中的最大值
    xmin = np.min(image) #求得InImg中的最小值
    image = np.round((ymax-ymin)*(image-xmin)/(xmax-xmin) + ymin)
    
    return image


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


def ant2mat2D(afftransform, center):
    """Compute offset"""
    
    offset = np.zeros((2,1))
    for i in range(0,2):
        offset[i] = afftransform[i,2] + center[i]
        for j in range(0,2):
            offset[i] = offset[i] - (afftransform[i,j] * center[j])
    
    afftransform[0,2] = offset[0]
    afftransform[1,2] = offset[1]
            
    return afftransform


def ant2mat3D(afftransform, center, origin, spacing, direction):
    """Compute offset"""
    
    offset = np.zeros((3,1))
    for i in range(0,3):
        offset[i] = afftransform[i,3] + center[i]
        for j in range(0,3):
            offset[i] = offset[i] - afftransform[i,j] * center[j]
    
    afftransform[0,3] = offset[0]
    afftransform[1,3] = offset[1]
    afftransform[2,3] = offset[2]
            
    return afftransform


def landmarkGenerate(shape,dis,label,coordinate,background):
    """Using the given coordinates"""
    
    landmarks = np.zeros_like(np.arange(shape[0]*shape[1]).reshape(shape[0],shape[1])) + background
    for i_label in range(len(coordinate)):
        for i_coordinate in range(len(coordinate[i_label])):
            x, y = coordinate[i_label][i_coordinate]
            landmarks[x:x+dis, y:y+dis] = label[i_label]
    
    return landmarks

def landmarkGenerate3D(shape,dis,label,coordinate,background):
    """Using the given coordinates"""
    
    landmarks = np.zeros_like(np.arange(shape[0]*shape[1]*shape[2]).reshape(shape[0],shape[1],shape[2])) + background
    for i_label in range(len(coordinate)):
        for i_coordinate in range(len(coordinate[i_label])):
            z, x, y = coordinate[i_label][i_coordinate]
            landmarks[z, x:x+dis, y:y+dis] = label[i_label]
    
    return landmarks


def denoiseLabels(img, ifmorphology='False'):
    """Post processing for random walk segmentation"""
    subLabel = cv2.convertScaleAbs(img) # int8 => uint8
    image, contours, hierarchy = cv2.findContours(subLabel, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) # 找到所有的轮廓
    area = []
    for k in range(len(contours)): # 找到最大的轮廓
    	area.append(cv2.contourArea(contours[k]))
        
    if area == []: # just background
        subMask = img
        maxContour = np.zeros_like(img)
    else:
        max_idx = np.argmax(np.array(area))
        maxsubLabel = cv2.drawContours(subLabel,contours,max_idx,2)
        maxContour = maxsubLabel - img;
        subMask = cv2.drawContours(maxContour.astype(np.float64), contours, max_idx, 1, cv2.FILLED) # 填充最大的轮廓
        
        if ifmorphology == 'True':
            kernel = skimage.morphology.disk(10)
            erosion_subLabel = skimage.morphology.erosion(subMask, kernel)
        #    kernel = skimage.morphology.disk(18)
#            kernel = skimage.morphology.disk(8)
            kernel = skimage.morphology.disk(6) # first postprocessing
#            kernel = skimage.morphology.disk(3) # second postprocessing
            dilation_subLabel = skimage.morphology.dilation(erosion_subLabel, kernel)
            subMask = dilation_subLabel
        elif ifmorphology == 'expand':
            kernel = skimage.morphology.disk(6)
            erosion_subLabel = skimage.morphology.dilation(subMask, kernel)
        #    kernel = skimage.morphology.disk(18)
            kernel = skimage.morphology.disk(6)
#            kernel = skimage.morphology.disk(6)
            dilation_subLabel = skimage.morphology.erosion(erosion_subLabel, kernel)
            subMask = dilation_subLabel
            
    return subMask, maxContour


#def denoiseLabels3D(img, ifmorphology='False'):
#    """Post processing for random walk segmentation"""
#    seg = img.copy()
#    kernel = skimage.morphology.ball(1) #cube
#    erosion_Label = skimage.morphology.erosion(seg, kernel)
#    maxContour = seg - erosion_Label # surface
#    subMask = scipy.ndimage.binary_fill_holes(maxContour.astype(np.float64)).astype('int8') # surface fill
#
#    if ifmorphology == 'True':
#        kernel = skimage.morphology.cube(9) #cube
#        erosion_subLabel = skimage.morphology.erosion(subMask, kernel)
#        kernel = skimage.morphology.cube(2) #cube
#        dilation_subLabel = skimage.morphology.dilation(erosion_subLabel, kernel)
#        subMask = dilation_subLabel
#    elif ifmorphology == 'expand':
#        kernel = skimage.morphology.cube(2) #cube
#        erosion_subLabel = skimage.morphology.dilation(subMask, kernel)
#        kernel = skimage.morphology.cube(2) #cube
#        dilation_subLabel = skimage.morphology.erosion(erosion_subLabel, kernel)
#        subMask = dilation_subLabel
#            
#    return subMask, maxContour


def denoiseLabels3D(img, ifmorphology='False'):
    
    imgSitk = sitk.GetImageFromArray(img)
    imgSitk = sitk.Cast(imgSitk, sitk.sitkInt8)
    
    subMaskSitk = sitk.ConnectedComponent(imgSitk, True)
    subMask = sitk.GetArrayFromImage(subMaskSitk)
    maxContourSitk = sitk.BinaryContour(subMaskSitk)
    maxContour = sitk.GetArrayFromImage(maxContourSitk)
    #boundary = sitk.BinaryGrindPeak( boundary )
    
    #contour = sitk.Cast(sitk.CannyEdgeDetection(test),sitk.sitkInt8)
    #contour=sitk.BinaryGrindPeak(contour)
#    filledImage = sitk.BinaryFillhole(boundary)
    
    if ifmorphology == 'True':
        kernel = skimage.morphology.cube(9) #cube
        erosion_subLabel = skimage.morphology.erosion(subMask, kernel)
        kernel = skimage.morphology.cube(2) #cube
        dilation_subLabel = skimage.morphology.dilation(erosion_subLabel, kernel)
        subMask = dilation_subLabel
    elif ifmorphology == 'expand':
        kernel = skimage.morphology.cube(2) #cube
        erosion_subLabel = skimage.morphology.dilation(subMask, kernel)
        kernel = skimage.morphology.cube(2) #cube
        dilation_subLabel = skimage.morphology.erosion(erosion_subLabel, kernel)
        subMask = dilation_subLabel
    
    return subMask, maxContour


#from scipy.spatial.distance import cdist
def maskExpand(mr, label):
    """neighbor nearest maps"""
    
    points = []
    for i in label:
        coor = np.where(mr==i)
        coorT = np.array([coor[0],coor[1]]).T
        points.append(np.hstack([coorT,np.ones([len(coor[0]),1])*i]).astype('int'))
    
    background = points[-1]    
    for i in range(len(background)):
        dist = np.zeros((len(label)-1))
        for j in range(len(label)-1):
            dist[j] = np.min(np.sqrt(np.sum(np.square(background[i][0:2] - np.array(points[j])[:,0:2]), axis=1))) # axis=0, 列
        
        background[i][2] = np.where(dist == np.min(dist))[0][0] + 1
        
    mask = mr.copy()
    for i in range(len(background)):
        y = background[i][0]
        x = background[i][1]
        mask[y,x] = background[i][2]
       
    return mask


def maskExpand3D(mr):
    """neighbor nearest maps"""
    mask = mr.copy()
    mask[mask==5]=0
    indices = np.zeros(((np.ndim(mask),) + mask.shape), dtype=np.int32)
    distanceMap = scipy.ndimage.morphology.distance_transform_edt(mask==0,return_indices=True,indices=indices) 
    for k in range(0,mr.shape[0]):
        for i in range(0,mr.shape[1]):
            for j in range(0,mr.shape[2]):
                z = indices[0,k,i,j]
                x = indices[1,k,i,j]
                y = indices[2,k,i,j]
                mask[k,i,j] = mr[z,x,y]
    
    return mask, distanceMap
    
#def maskExpand3Dprevious(mrMask, maxContourmr, label):    
#    mr = maxContourmr
#    if np.unique(mr).shape == (1,) or np.unique(mr).shape == (2,) or np.unique(mr).shape == (3,):
#        mask = np.zeros_like(mr) + 2
#    else:
#        points = []
#        for i in label[0:3]:
#            coor = np.where(mr==i)
#            coorT = np.array([coor[0],coor[1]]).T
#            points.append(np.hstack([coorT,np.ones([len(coor[0]),1])*i]).astype('int'))
#            
#        backgroundCoor = np.where(mrMask==4)
#        backgroundCoorT = np.array([np.array(backgroundCoor[0]),np.array(backgroundCoor[1])]).T
#        background = np.hstack([backgroundCoorT,np.ones([len(backgroundCoor[0]),1])*4]).astype('int')
#          
#        for i in range(len(background)):
#            dist = np.zeros((len(label)-1))
#            for j in range(len(label)-1):
#                dist[j] = np.min(np.sqrt(np.sum(np.square(background[i][0:2] - np.array(points[j])[:,0:2]), axis=1))) # axis=0, 列
#            
#            background[i][2] = np.where(dist == np.min(dist))[0][0] + 1
#            
#        mask = mrMask.copy()
#        for i in range(len(background)):
#            y = background[i][0]
#            x = background[i][1]
#            mask[y,x] = background[i][2]
       
#    return mask

#def maskExpand3D(mrMask, maxContourmr, label): #1,2,3
#    """neighbor nearest maps"""
#    points = []
#    for i in label[0:3]: # 1,2,3
#        coor = np.where(maxContourmr==i)
#        coorT = np.array([np.array(coor[0]),np.array(coor[1]),np.array(coor[2])]).T
#        points.append(np.hstack([coorT,np.ones([len(coor[0]),1])*i]).astype('int')) # (z,x,y,label),3
#    
#    backgroundCoor = np.where(mrMask==1)
#    backgroundCoorT = np.array([np.array(backgroundCoor[0]),np.array(backgroundCoor[1]),np.array(backgroundCoor[2])]).T
#    background = np.hstack([backgroundCoorT,np.ones([len(backgroundCoor[0]),1])*4]).astype('int')
#    
#    dist = np.zeros((len(label)-1))
#    for j in range(len(label)-1):
#        for i in range(len(points[j])):
#            pointExpand = np.tile(np.array(points[j])[i,0:3],len(background)).reshape(len(background),3)
#            temp = np.zeros_like(background[:,0:3]) + pointExpand
#
#        
#        dist[j] = np.min(np.sqrt(np.sum(np.square(background[i][0:3] - np.array(points[j])[:,0:3]), axis=1))) # axis=0, 列
#        
#        background[i][3] = np.where(dist == np.min(dist))[0][0] + 1
#    
#    mask = np.zeros_like(mrMask)
#    for i in range(len(background)):
#        z = background[i][0]
#        x = background[i][1]
#        y = background[i][2]
#        mask[z,x,y] = background[i][3]
#        
#    return mask
    

def waxLabelProcess(waxLabeltoMRarry, waxLabeltoLSarry, origin, spacing, direction):
    waxLabeltoMRarry[waxLabeltoMRarry==1057] = -1057
    waxLabeltoMRarry[waxLabeltoMRarry==1118] = -1118
    waxLabeltoMRarry[waxLabeltoMRarry==1125] = -1125
    waxLabeltoLSarry[waxLabeltoLSarry==1057] = -1057
    waxLabeltoLSarry[waxLabeltoLSarry==1118] = -1118
    waxLabeltoLSarry[waxLabeltoLSarry==1125] = -1125
    
    for i in range(1, waxLabeltoMRarry.shape[0]):
        mask = waxLabeltoMRarry[i,:,:]>= 1000
        tempA = np.zeros_like(waxLabeltoMRarry[i,:,:])
        tempA[mask] = waxLabeltoMRarry[i,:,:][mask]
        tempB = tempA - 1000
        tempB[tempB<0] = 0
        tempC = waxLabeltoMRarry[i,:,:]
        tempC[tempC>1000] = 0
        waxLabeltoMRarry[i,:,:] = tempB + tempC
    
    for i in range(1, waxLabeltoLSarry.shape[0]):
        mask = waxLabeltoLSarry[i,:,:]>= 1000
        tempA = np.zeros_like(waxLabeltoLSarry[i,:,:])
        tempA[mask] = waxLabeltoLSarry[i,:,:][mask]
        tempB = tempA - 1000
        tempB[tempB<0] = 0
        tempC = waxLabeltoLSarry[i,:,:]
        tempC[tempC>1000] = 0
        waxLabeltoLSarry[i,:,:] = tempB + tempC
        
    # classify waxholm label into 3 segments   
    waxLabeltoMRarryCopy = waxLabeltoMRarry.copy()
    waxLabeltoMRarryCopy[waxLabeltoMRarry==57] = -1
    waxLabeltoMRarryCopy[waxLabeltoMRarry==118] = -1
    waxLabeltoMRarryCopy[waxLabeltoMRarry==125] = -1
    waxLabeltoMRarryCopy[waxLabeltoMRarry==-1057] = -1-1
    waxLabeltoMRarryCopy[waxLabeltoMRarry==-1118] = -1-1
    waxLabeltoMRarryCopy[waxLabeltoMRarry==-1125] = -1-1
    
    waxLabeltoMRarryCopy[waxLabeltoMRarry==119] = -2-1
    waxLabeltoMRarryCopy[waxLabeltoMRarry==120] = -2-1
    waxLabeltoMRarryCopy[waxLabeltoMRarry==121] = -2-1
    waxLabeltoMRarryCopy[waxLabeltoMRarry==123] = -2-1 # test
    waxLabeltoMRarryCopy[waxLabeltoMRarry==124] = -2-1
    waxLabeltoMRarryCopy[waxLabeltoMRarry==127] = -2-1
    waxLabeltoMRarryCopy[waxLabeltoMRarry==133] = -2-1
    waxLabeltoMRarryCopy[waxLabeltoMRarry==142] = -2-1
    waxLabeltoMRarryCopy[waxLabeltoMRarry==148] = -2-1
    waxLabeltoMRarryCopy[waxLabeltoMRarry==149] = -2-1 # test
    waxLabeltoMRarryCopy[waxLabeltoMRarry==156] = -2-1
    waxLabeltoMRarryCopy[waxLabeltoMRarry==157] = -2-1
    waxLabeltoMRarryCopy[waxLabeltoMRarry==160] = -2-1
    for x in range(89, 162+1):
        waxLabeltoMRarryCopy[waxLabeltoMRarryCopy==x] = -3-1
    waxLabeltoMRarryCopy[waxLabeltoMRarryCopy>0] = -2-1
    waxLabeltoMRarryCopy = abs(waxLabeltoMRarryCopy)
    waxLabeltoMRarryCopy[0:77,:,:][waxLabeltoMRarryCopy[0:77,:,:]==1] = 2+1 # just to remove obvious noise
    waxLabeltoMRarryCopy[161:,:,:][waxLabeltoMRarryCopy[161:,:,:]==1] = 2+1 # just to remove obvious noise
    
    waxLabeltoMRarryCopy[:,160:224,:][waxLabeltoMRarryCopy[:,160:224,:]==2] = 2+1 # just to remove obvious noise 
# lS    
    waxLabeltoLSarryCopy = waxLabeltoLSarry.copy()
    waxLabeltoLSarryCopy[waxLabeltoLSarry==57] = -1
    waxLabeltoLSarryCopy[waxLabeltoLSarry==118] = -1
    waxLabeltoLSarryCopy[waxLabeltoLSarry==125] = -1
    waxLabeltoLSarryCopy[waxLabeltoLSarry==-1057] = -1-1
    waxLabeltoLSarryCopy[waxLabeltoLSarry==-1118] = -1-1
    waxLabeltoLSarryCopy[waxLabeltoLSarry==-1125] = -1-1
    
    waxLabeltoLSarryCopy[waxLabeltoLSarry==119] = -2-1
    waxLabeltoLSarryCopy[waxLabeltoLSarry==120] = -2-1
    waxLabeltoLSarryCopy[waxLabeltoLSarry==121] = -2-1
    waxLabeltoLSarryCopy[waxLabeltoLSarry==123] = -2-1 # test
    waxLabeltoLSarryCopy[waxLabeltoLSarry==124] = -2-1
    waxLabeltoLSarryCopy[waxLabeltoLSarry==127] = -2-1
    waxLabeltoLSarryCopy[waxLabeltoLSarry==133] = -2-1
    waxLabeltoLSarryCopy[waxLabeltoLSarry==142] = -2-1
    waxLabeltoLSarryCopy[waxLabeltoLSarry==148] = -2-1
    waxLabeltoLSarryCopy[waxLabeltoLSarry==149] = -2-1 # test
    waxLabeltoLSarryCopy[waxLabeltoLSarry==156] = -2-1
    waxLabeltoLSarryCopy[waxLabeltoLSarry==157] = -2-1
    waxLabeltoLSarryCopy[waxLabeltoLSarry==160] = -2-1
    for x in range(89, 162+1):
        waxLabeltoLSarryCopy[waxLabeltoLSarryCopy==x] = -3-1
    waxLabeltoLSarryCopy[waxLabeltoLSarryCopy>0] = -2-1
    waxLabeltoLSarryCopy = abs(waxLabeltoLSarryCopy)
    waxLabeltoLSarryCopy[0:67,:,:][waxLabeltoLSarryCopy[0:67,:,:]==1] = 2+1 # just to remove obvious noise
    waxLabeltoLSarryCopy[166:,:,:][waxLabeltoLSarryCopy[166:,:,:]==1] = 2+1 # just to remove obvious noise
    
    waxLabeltoLSarryCopy[:,160:224,:][waxLabeltoLSarryCopy[:,160:224,:]==2] = 2+1 # just to remove obvious noise
    
    niisave(waxLabeltoMRarryCopy, origin, spacing, direction, path = '../result/input/waxLabeltoMRarry.nii.gz')
    niisave(waxLabeltoLSarryCopy, origin, spacing, direction, path = '../result/input/waxLabeltoLSarry.nii.gz')

    return waxLabeltoMRarryCopy, waxLabeltoLSarryCopy