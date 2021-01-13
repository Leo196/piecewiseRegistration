# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 23:48:45 2020

@author: fangyan
"""

from skimage.segmentation import random_walker
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy.io as scio
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import matplotlib as mpl
import ants
import gc

from basicProcessing import norm
from basicProcessing import norm255
from basicProcessing import splitImage
from basicProcessing import ant2mat2D
from basicProcessing import ant2mat3D
from basicProcessing import landmarkGenerate
from basicProcessing import landmarkGenerate3D
from basicProcessing import denoiseLabels
from basicProcessing import maskExpand
from basicProcessing import maskExpand3D
from basicProcessing import waxLabelProcess

from labelProcessing import postProcessing
from labelProcessing import postProcessing3D

from affineWarp import affine
from affineWarp import warpAffine
from affineWarp import warp_field
from affineWarp import warp_field3D
from affineWarp import warp_fieldtoimage
from affineWarp import warp_fieldtoimage3D

from fieldLaplace import fieldLaplace
from fieldLaplace import fieldLaplace3D
from fieldLaplace import fieldMaskLaplace
from fieldLaplace import fieldMaskLaplace3D
from fieldGaussian import fieldGaussian
from fieldGaussian import fieldGaussian3D
from fieldGaussian import fieldMaskGaussian
from fieldGaussian import fieldMaskGaussian3D

from visiualization import myshow
from visiualization import niisave
from visiualization import nii

#import re
#for x in dir():
#    if not re.match('^__',x) and x!="re":
#        exec(" ".join(("del",x)))

#plt.subplots_adjust(top=0.97,bottom=0.03,left=0.03,right=1,hspace=0.1,wspace=0)

#plt.close()
WHITE = '#FFFFFF'
GREY = '#999999'
RED = '#E41A1C'
PURPLE = '#984EA3'
BROWN = '#A65628'
LIGHT_BROWN = '#C55A11'
LIGHT_PURPLE = '#C4A8A8'
SLIDE = 122
label = [1,2,3,4,5]
SEGMENTS = 4

#imread waxholm label, MR, LS
waxholmLabeltoMRPath = '../data/07waxholmLabel_to_FAMRI_elastixBspline.nii.gz'
waxholmLabeltoLSPath = '../data/waxholmLabel_to_LSFSL.nii.gz'
mrpath = '../data/02raw_FAMRI_brainBET.nii.gz'
lspath = '../data/01LS_to_FAMRI_fslAffine.nii.gz'

sitk_waxLabeltoMR = sitk.ReadImage(waxholmLabeltoMRPath)
sitk_waxLabeltoLS = sitk.ReadImage(waxholmLabeltoLSPath)
sitk_MR = sitk.ReadImage(mrpath)
sitk_LS = sitk.ReadImage(lspath)

mrOrigin = sitk_MR.GetOrigin()
mrSpacing = sitk_MR.GetSpacing()
mrDirection = np.array(sitk_MR.GetDirection())
lsOrigin = sitk_LS.GetOrigin()
lsSpacing = sitk_LS.GetSpacing()
lsDirection = np.array(sitk_LS.GetDirection())

waxLabeltoMRarry = sitk.GetArrayFromImage(sitk_waxLabeltoMR)
waxLabeltoLSarry = sitk.GetArrayFromImage(sitk_waxLabeltoLS)
mrarry = sitk.GetArrayFromImage(sitk_MR)
lsarry = sitk.GetArrayFromImage(sitk_LS)

mrarry = np.swapaxes(mrarry,0,2)
lsarry = np.swapaxes(lsarry,0,2)
waxLabeltoMRarry = np.swapaxes(waxLabeltoMRarry,0,2)
waxLabeltoLSarry = np.swapaxes(waxLabeltoLSarry,0,2)

arry_MR = mrarry
arry_LS = lsarry
arry_LS[arry_LS<30] = 0
arry_LS[:,:,180:256] = 0

arry_MR = norm(arry_MR)
arry_LS = norm(arry_LS)

niisave(arry_MR, sitk_MR.GetOrigin(), sitk_MR.GetSpacing(), sitk_LS.GetDirection(), path = '../result/input/mrisitk3D.nii.gz')
niisave(arry_LS, sitk_LS.GetOrigin(),  sitk_LS.GetSpacing(), sitk_LS.GetDirection(), path = '../result/input/lssitk3D.nii.gz')
niisave(waxLabeltoMRarry, sitk_MR.GetOrigin(), sitk_MR.GetSpacing(), sitk_LS.GetDirection(), path = '../result/input/waxLabeltoMRsitk3D.nii.gz')
niisave(waxLabeltoLSarry, sitk_LS.GetOrigin(),  sitk_LS.GetSpacing(), sitk_LS.GetDirection(), path = '../result/input/waxLabeltoLSsitk3D.nii.gz')

plt.figure(1,figsize=(6, 8))
plt.subplot(3,2,1)
plt.title('Image (MR, Light-sheet)')
mrshow = arry_MR[SLIDE,:,:].copy()
mrshow = mrshow.transpose(1,0)
lsshow = arry_LS[SLIDE,:,:].copy()
lsshow = lsshow.transpose(1,0)
plt.imshow(np.hstack((mrshow, lsshow)),cmap='gray') 
plt.axis('off')
plt.show()

plt.figure(10,figsize=(6, 8))
plt.subplot(3,2,1)
plt.title('Image (MR, Light-sheet)')
mrshow = arry_MR[:,:,SLIDE].copy()
mrshow = mrshow.transpose(1,0)
lsshow = arry_LS[:,:,SLIDE].copy()
lsshow = lsshow.transpose(1,0)
plt.imshow(np.hstack((mrshow, lsshow)),cmap='gray') 
plt.axis('off')
plt.show()

# label
landmarks_MR, landmarks_LS = waxLabelProcess(waxLabeltoMRarry, waxLabeltoLSarry, mrOrigin, mrSpacing, mrDirection)

'''check coordinates: remove incorrect landmarks'''
landmarks_MR[arry_MR==0] = 5
landmarks_LS[arry_LS<0.099] = 5

plt.figure(1,figsize=(8, 12))
plt.subplot(3,2,2)
plt.title('Waxholm Labels(MR, Light-sheet)')
labelMRshow = landmarks_MR[SLIDE,:,:].copy()
labelMRshow = labelMRshow.transpose(1,0)
labelLSshow = landmarks_LS[SLIDE,:,:].copy()
labelLSshow = labelLSshow.transpose(1,0)
cmap = mpl.colors.ListedColormap([WHITE,RED,LIGHT_BROWN,PURPLE,BROWN,GREY])
mat = plt.imshow(np.hstack((labelMRshow, labelLSshow)),cmap) 
plt.axis('off') 
ax = plt.gca()
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
unique_data = np.unique(landmarks_MR);
plt.colorbar(mat, cax = cax, ticks=np.arange(np.min(unique_data),np.max(unique_data)+1), spacing='uniform')

plt.figure(10,figsize=(6, 8))
plt.subplot(3,2,2)
plt.title('Waxholm Labels(MR, Light-sheet)')
labelMRshow = landmarks_MR[:,:,SLIDE].copy()
labelMRshow = labelMRshow.transpose(1,0)
labelLSshow = landmarks_LS[:,:,SLIDE].copy()
labelLSshow = labelLSshow.transpose(1,0)
cmap = mpl.colors.ListedColormap([WHITE,RED,LIGHT_BROWN,PURPLE,BROWN,GREY])
mat = plt.imshow(np.hstack((labelMRshow, labelLSshow)),cmap) 
plt.axis('off') 
ax = plt.gca()
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
unique_data = np.unique(landmarks_MR);
plt.colorbar(mat, cax = cax, ticks=np.arange(np.min(unique_data),np.max(unique_data)+1), spacing='uniform')

# add: post-processing after waxLabelProcess
mr0,maxContour_mr,ls0,maxContour_ls,dilation_mr1,dilation_mr2,dilation_mr3,dilation_mr4,dilation_ls1,dilation_ls2,dilation_ls3,dilation_ls4 = postProcessing(landmarks_MR, landmarks_LS, ifmorphology='True')

niisave(mr0, mrOrigin, mrSpacing, mrDirection, path = '../result/input/MRFirstPostProcessing.nii.gz')
niisave(ls0, lsOrigin, lsSpacing, lsDirection, path = '../result/input/LSFirstPostProcessing.nii.gz')

mr0[(landmarks_MR-mr0)==-5]=0
mr0[(landmarks_MR-mr0)==-4]=0
mr0[(landmarks_MR-mr0)==-3]=0
mr0[(landmarks_MR-mr0)==-2]=0
mr0[(landmarks_MR-mr0)==-1]=0
ls0[(landmarks_LS-ls0)==-5]=0
ls0[(landmarks_LS-ls0)==-4]=0
ls0[(landmarks_LS-ls0)==-3]=0
ls0[(landmarks_LS-ls0)==-2]=0
ls0[(landmarks_LS-ls0)==-1]=0 

plt.figure(1,figsize=(6, 8))
plt.subplot(3,2,3)
plt.title('First PostProcessing(MR, Light-sheet)')
mrshow = mr0[SLIDE,:,:].copy()
mrshow = mrshow.transpose(1,0)
lsshow = ls0[SLIDE,:,:].copy()
lsshow = lsshow.transpose(1,0)
#cmap = mpl.colors.ListedColormap([RED,PURPLE,BROWN,GREY,LIGHT_BROWN,LIGHT_PURPLE])
cmap = mpl.colors.ListedColormap([WHITE,RED,LIGHT_BROWN,PURPLE,BROWN,GREY])
mat = plt.imshow(np.hstack((mrshow, lsshow)),cmap) 
plt.axis('off') 
ax = plt.gca()
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
unique_data = np.unique(mr0);
plt.colorbar(mat, cax = cax, ticks=np.arange(np.min(unique_data),np.max(unique_data)+1), spacing='uniform')

plt.figure(10,figsize=(6, 8))
plt.subplot(3,2,3)
plt.title('First PostProcessing(MR, Light-sheet)')
mrshow = mr0[:,:,SLIDE].copy()
mrshow = mrshow.transpose(1,0)
lsshow = ls0[:,:,SLIDE].copy()
lsshow = lsshow.transpose(1,0)
#cmap = mpl.colors.ListedColormap([RED,PURPLE,BROWN,GREY,LIGHT_BROWN,LIGHT_PURPLE])
cmap = mpl.colors.ListedColormap([WHITE,RED,LIGHT_BROWN,PURPLE,BROWN,GREY])
mat = plt.imshow(np.hstack((mrshow, lsshow)),cmap) 
plt.axis('off') 
ax = plt.gca()
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
unique_data = np.unique(mr0);
plt.colorbar(mat, cax = cax, ticks=np.arange(np.min(unique_data),np.max(unique_data)+1), spacing='uniform')

'''random walk (first time)'''
labels_MR1 = random_walker(arry_MR.astype(np.float64),mr0,beta = 10,multichannel=False,spacing=sitk_MR.GetSpacing()).astype(np.int8)
labels_LS1 = random_walker(arry_LS,ls0,beta = 10,multichannel=False,spacing=sitk_LS.GetSpacing()).astype(np.int8)

niisave(labels_MR1, mrOrigin, mrSpacing, mrDirection, path = '../result/input/MRFirstRandomWalk.nii.gz')
niisave(labels_LS1, lsOrigin, lsSpacing, lsDirection, path = '../result/input/LSFirstRandomWalk.nii.gz')

plt.figure(1,figsize=(6, 8))
plt.subplot(3,2,4)
plt.title('RW(MR, Light-sheet)')
labelMRshow = labels_MR1[SLIDE,:,:].copy()
labelMRshow = labelMRshow.transpose(1,0)
labelLSshow = labels_LS1[SLIDE,:,:].copy()
labelLSshow = labelLSshow.transpose(1,0)
cmap = mpl.colors.ListedColormap([RED,LIGHT_BROWN,PURPLE,BROWN,GREY])
mat = plt.imshow(np.hstack((labelMRshow, labelLSshow)),cmap) 
plt.axis('off') 
ax = plt.gca()
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
unique_data = np.unique(landmarks_MR);
plt.colorbar(mat, cax = cax, ticks=np.arange(np.min(unique_data),np.max(unique_data)+1), spacing='uniform')

plt.figure(10,figsize=(6, 8))
plt.subplot(3,2,4)
plt.title('RW(MR, Light-sheet)')
labelMRshow = labels_MR1[:,:,SLIDE].copy()
labelMRshow = labelMRshow.transpose(1,0)
labelLSshow = labels_LS1[:,:,SLIDE].copy()
labelLSshow = labelLSshow.transpose(1,0)
cmap = mpl.colors.ListedColormap([RED,LIGHT_BROWN,PURPLE,BROWN,GREY])
mat = plt.imshow(np.hstack((labelMRshow, labelLSshow)),cmap) 
plt.axis('off') 
ax = plt.gca()
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
unique_data = np.unique(landmarks_MR);
plt.colorbar(mat, cax = cax, ticks=np.arange(np.min(unique_data),np.max(unique_data)+1), spacing='uniform')

"""PostProcessing-Second"""
mr1,maxContour_mr,ls1,maxContour_ls,dilation_mr1,dilation_mr2,dilation_mr3,dilation_mr4,dilation_ls1,dilation_ls2,dilation_ls3,dilation_ls4 = postProcessing(labels_MR1, labels_LS1, ifmorphology='True')

niisave(mr1, mrOrigin, mrSpacing, mrDirection, path = '../result/input/MRSecondPostProcessing.nii.gz')
niisave(ls1, lsOrigin, lsSpacing, lsDirection, path = '../result/input/LSSecondPostProcessing.nii.gz')

mr1[(labels_MR1-mr1)==-5]=0
mr1[(labels_MR1-mr1)==-4]=0
mr1[(labels_MR1-mr1)==-3]=0
mr1[(labels_MR1-mr1)==-2]=0
mr1[(labels_MR1-mr1)==-1]=0
ls1[(labels_LS1-ls1)==-5]=0
ls1[(labels_LS1-ls1)==-4]=0
ls1[(labels_LS1-ls1)==-3]=0
ls1[(labels_LS1-ls1)==-2]=0
ls1[(labels_LS1-ls1)==-1]=0 

plt.figure(1,figsize=(6, 8))
plt.subplot(3,2,5)
plt.title('Second PostProcessing(MR, Light-sheet)')
mrshow = mr1[SLIDE,:,:].copy()
mrshow = mrshow.transpose(1,0)
lsshow = ls1[SLIDE,:,:].copy()
lsshow = lsshow.transpose(1,0)
#cmap = mpl.colors.ListedColormap([RED,PURPLE,BROWN,GREY,LIGHT_BROWN,LIGHT_PURPLE])
cmap = mpl.colors.ListedColormap([WHITE,LIGHT_BROWN,PURPLE,BROWN,GREY])
mat = plt.imshow(np.hstack((mrshow, lsshow)),cmap) 
plt.axis('off') 
ax = plt.gca()
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
unique_data = np.unique(mr1);
plt.colorbar(mat, cax = cax, ticks=np.arange(np.min(unique_data),np.max(unique_data)+1), spacing='uniform')

plt.figure(10,figsize=(6, 8))
plt.subplot(3,2,5)
plt.title('Second PostProcessing(MR, Light-sheet)')
mrshow = mr1[:,:,SLIDE].copy()
mrshow = mrshow.transpose(1,0)
lsshow = ls1[:,:,SLIDE].copy()
lsshow = lsshow.transpose(1,0)
#cmap = mpl.colors.ListedColormap([RED,PURPLE,BROWN,GREY,LIGHT_BROWN,LIGHT_PURPLE])
cmap = mpl.colors.ListedColormap([WHITE,RED,LIGHT_BROWN,PURPLE,BROWN,GREY])
mat = plt.imshow(np.hstack((mrshow, lsshow)),cmap) 
plt.axis('off') 
ax = plt.gca()
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
unique_data = np.unique(mr1);
plt.colorbar(mat, cax = cax, ticks=np.arange(np.min(unique_data),np.max(unique_data)+1), spacing='uniform')

'''random walk (secone time)'''
labels_MR2 = random_walker(arry_MR.astype(np.float64),mr1,beta = 10,multichannel=False,spacing=sitk_MR.GetSpacing()).astype(np.int8)
labels_LS2 = random_walker(arry_LS,ls1,beta = 10,multichannel=False,spacing=sitk_LS.GetSpacing()).astype(np.int8)

niisave(labels_MR2, mrOrigin, mrSpacing, mrDirection, path = '../result/input/MRSecondRandomWalk.nii.gz')
niisave(labels_LS2, lsOrigin, lsSpacing, lsDirection, path = '../result/input/LSSecondRandomWalk.nii.gz')

plt.figure(1,figsize=(6, 8))
plt.subplot(3,2,6)
plt.title('RW(MR, Light-sheet)')
labelMRshow = labels_MR2[SLIDE,:,:].copy()
labelMRshow = labelMRshow.transpose(1,0)
labelLSshow = labels_LS2[SLIDE,:,:].copy()
labelLSshow = labelLSshow.transpose(1,0)
cmap = mpl.colors.ListedColormap([RED,LIGHT_BROWN,PURPLE,BROWN,GREY])
mat = plt.imshow(np.hstack((labelMRshow, labelLSshow)),cmap) 
plt.axis('off') 
ax = plt.gca()
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
unique_data = np.unique(landmarks_MR);
plt.colorbar(mat, cax = cax, ticks=np.arange(np.min(unique_data),np.max(unique_data)+1), spacing='uniform')

plt.figure(10,figsize=(6, 8))
plt.subplot(3,2,6)
plt.title('RW(MR, Light-sheet)')
labelMRshow = labels_MR2[:,:,SLIDE].copy()
labelMRshow = labelMRshow.transpose(1,0)
labelLSshow = labels_LS2[:,:,SLIDE].copy()
labelLSshow = labelLSshow.transpose(1,0)
cmap = mpl.colors.ListedColormap([RED,LIGHT_BROWN,PURPLE,BROWN,GREY])
mat = plt.imshow(np.hstack((labelMRshow, labelLSshow)),cmap) 
plt.axis('off') 
ax = plt.gca()
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
unique_data = np.unique(landmarks_MR);
plt.colorbar(mat, cax = cax, ticks=np.arange(np.min(unique_data),np.max(unique_data)+1), spacing='uniform')

mr = labels_MR2
mr1 = mr.copy()
mr1[mr1!=1] = 0
mr2 = mr.copy()
mr2[mr2!=2] = 0
mr3 = mr.copy()
mr3[mr3!=3] = 0
mr4 = mr.copy()
mr4[mr4!=4] = 0
ls = labels_LS2
ls1 = ls.copy()
ls1[ls1!=1] = 0 
ls2 = ls.copy()
ls2[ls2!=2] = 0 
ls3 = ls.copy()
ls3[ls3!=3] = 0 
ls4 = ls.copy()
ls4[ls4!=4] = 0 
'''
# segmentation overlay
plt.figure(1,figsize=(6, 8))
plt.subplot(3,2,6)
plt.title('Contour Overlay(MR, Light-sheet)')
mrshow = arry_MR[SLIDE,:,:].copy()
mrshow = mrshow.transpose(1,0)
lsshow = arry_LS[SLIDE,:,:].copy()
lsshow = lsshow.transpose(1,0)
mrContourshow = maxContour_mr[SLIDE,:,:].copy()
mrContourshow = mrContourshow.transpose(1,0)
lsContourshow = maxContour_ls[SLIDE,:,:].copy()
lsContourshow = lsContourshow.transpose(1,0)
mat = plt.imshow(np.hstack((mrshow*1.3+mrContourshow, lsshow*1.3+lsContourshow)),cmap = 'gray') 
plt.axis('off') 

plt.figure(10,figsize=(6, 8))
plt.subplot(3,2,6)
plt.title('Contour Overlay(MR, Light-sheet)')
mrshow = arry_MR[:,:,SLIDE].copy()
mrshow = mrshow.transpose(1,0)
lsshow = arry_LS[:,:,SLIDE].copy()
lsshow = lsshow.transpose(1,0)
mrContourshow = maxContour_mr[:,:,SLIDE].copy()
mrContourshow = mrContourshow.transpose(1,0)
lsContourshow = maxContour_ls[:,:,SLIDE].copy()
lsContourshow = lsContourshow.transpose(1,0)
mat = plt.imshow(np.hstack((mrshow*1.3+mrContourshow, lsshow*1.3+lsContourshow)),cmap = 'gray') 
plt.axis('off') 
'''

# boundary detection for 3D of seg1,seg2,seg3,seg4
subMaskmr1 = np.zeros_like(mr1)
maxContourmr1 = np.zeros_like(mr1)
subMaskmr2 = np.zeros_like(mr2)
maxContourmr2 = np.zeros_like(mr2)
subMaskmr3 = np.zeros_like(mr3)
maxContourmr3 = np.zeros_like(mr3)
subMaskmr4 = np.zeros_like(mr4)
maxContourmr4 = np.zeros_like(mr4)
subMaskls1 = np.zeros_like(ls1)
maxContourls1 = np.zeros_like(ls1)
subMaskls2 = np.zeros_like(ls2)
maxContourls2 = np.zeros_like(ls2)
subMaskls3 = np.zeros_like(ls3)
maxContourls3 = np.zeros_like(ls3)
subMaskls4 = np.zeros_like(ls4)
maxContourls4 = np.zeros_like(ls4)
for i in range(0, dilation_mr1.shape[0]):
    subMaskmr1[i,:,:], maxContourmr1[i,:,:] = denoiseLabels(norm(mr1[i,:,:]), ifmorphology='False')
    subMaskmr2[i,:,:], maxContourmr2[i,:,:] = denoiseLabels(norm(mr2[i,:,:]), ifmorphology='False')
    subMaskmr3[i,:,:], maxContourmr3[i,:,:] = denoiseLabels(norm(mr3[i,:,:]), ifmorphology='False')
    subMaskmr4[i,:,:], maxContourmr4[i,:,:] = denoiseLabels(norm(mr4[i,:,:]), ifmorphology='False')
for i in range(0, dilation_ls1.shape[0]):
    subMaskls1[i,:,:], maxContourls1[i,:,:] = denoiseLabels(norm(ls1[i,:,:]), ifmorphology='False')
    subMaskls2[i,:,:], maxContourls2[i,:,:] = denoiseLabels(norm(ls2[i,:,:]), ifmorphology='False')
    subMaskls3[i,:,:], maxContourls3[i,:,:] = denoiseLabels(norm(ls3[i,:,:]), ifmorphology='False')
    subMaskls4[i,:,:], maxContourls4[i,:,:] = denoiseLabels(norm(ls4[i,:,:]), ifmorphology='False')
maxContourmr1[maxContourmr1==1]=1
maxContourmr2[maxContourmr2==1]=2
maxContourmr3[maxContourmr3==1]=3
maxContourmr4[maxContourmr4==1]=4
maxContourmr = maxContourmr1 + maxContourmr2 + maxContourmr3 + maxContourmr4
maxContourls1[maxContourls1==1]=1
maxContourls2[maxContourls2==1]=2
maxContourls3[maxContourls3==1]=3
maxContourls4[maxContourls3==1]=4
maxContourls = maxContourls1 + maxContourls2 + maxContourls3 + maxContourls4
mrMask = np.zeros_like(mr) + 1
mrMask = mrMask - subMaskmr1 - subMaskmr2 - subMaskmr3 - subMaskmr4
mrMask[mrMask==-1] = 0 # overlap = 0
lsMask = np.zeros_like(ls) + 1
lsMask = lsMask - subMaskls1 - subMaskls2 - subMaskls3 - subMaskls4
lsMask[lsMask==-1] = 0 # overlap = 0
#######################################
#plt.figure(1,figsize=(6, 8))
#plt.subplot(4,1,2)
#plt.title('Labelmarks(MR, Light-sheet)')
#landContourMRshow = (landmarks_MR[SLIDE,:,:]+maxContour_mr[SLIDE,:,:]).copy()
#landContourMRshow = landContourMRshow.transpose(1,0)
#landContourLSshow = (landmarks_LS[SLIDE,:,:]+maxContour_ls[SLIDE,:,:]).copy()
#landContourLSshow = landContourLSshow.transpose(1,0)
#cmap = mpl.colors.ListedColormap([WHITE,RED,PURPLE,BROWN,GREY])
#mat = plt.imshow(np.hstack((landContourMRshow,landContourLSshow)),cmap)
#plt.axis('off') 
#ax = plt.gca()
#divider = make_axes_locatable(ax)
#cax = divider.append_axes("right", size="5%", pad=0.05)
#unique_data = np.unique(landmarks_MR);
#plt.colorbar(mat, cax = cax, ticks=np.arange(np.min(unique_data),np.max(unique_data)+1), spacing='uniform')
#
#niisave(landmarks_MR+maxContour_mr, mrOrigin, mrSpacing, mrDirection, path = '../result/input/landmarks_MR.nii.gz')
#niisave(landmarks_LS+maxContour_ls, lsOrigin, lsSpacing, lsDirection, path = '../result/input/landmarks_LS.nii.gz')

# label overlay visiualization
subimgmr_255 = sitk.GetImageFromArray(np.swapaxes(norm255(mrarry),0,2))
subimgmr_255 = sitk.Cast(subimgmr_255,sitk.sitkUInt8)
sublabelmr = sitk.GetImageFromArray(np.swapaxes(mr,0,2))
sublabelmr = sitk.Cast(sublabelmr,sitk.sitkUInt8)
subSitkmr = sitk.LabelOverlay(subimgmr_255, sublabelmr)
subSitkmr.SetSpacing(mrSpacing)
#plane = ['axial', 'sagittal', 'coronal']
plane = ['sagittal']
for i in range(0, len(plane)):
    myshow(subSitkmr,plane[i] ,'./regi_waxholmLabel to FAMRI',SLIDE,title='MR', dpi=80)
    
subimgls_255 = sitk.GetImageFromArray(np.swapaxes(norm255(lsarry),0,2))
subimgls_255 = sitk.Cast(subimgls_255,sitk.sitkUInt8)
sublabells = sitk.GetImageFromArray(np.swapaxes(ls,0,2))
sublabells = sitk.Cast(sublabells,sitk.sitkUInt8)
subSitkls = sitk.LabelOverlay(subimgls_255, sublabells)
subSitkls.SetSpacing(lsSpacing)
for i in range(0, len(plane)):
    myshow(subSitkls,plane[i] ,'./regi_waxholmLabel to FAMRI',SLIDE,title='LS', dpi=80)

"""Split image"""
savePath_mr1 = '../result/subImg_mr13D.nii.gz'
subImg_mr1 = splitImage(arry_MR, mr1, savePath_mr1, mrSpacing, mrOrigin, mrDirection)
savePath_mr2 = '../result/subImg_mr23D.nii.gz'
subImg_mr2 = splitImage(arry_MR, mr2, savePath_mr2, mrSpacing, mrOrigin, mrDirection)
savePath_mr3 = '../result/subImg_mr33D.nii.gz'
subImg_mr3 = splitImage(arry_MR, mr3, savePath_mr3, mrSpacing, mrOrigin, mrDirection)
savePath_mr4 = '../result/subImg_mr43D.nii.gz'
subImg_mr4 = splitImage(arry_MR, mr4, savePath_mr4, mrSpacing, mrOrigin, mrDirection)

savePath_ls1 = '../result/subImg_ls13D.nii.gz'
subImg_ls1 = splitImage(arry_LS, ls1, savePath_ls1, lsSpacing, lsOrigin, lsDirection)
savePath_ls2 = '../result/subImg_ls23D.nii.gz'
subImg_ls2 = splitImage(arry_LS, ls2, savePath_ls2, lsSpacing, lsOrigin, lsDirection)
savePath_ls3 = '../result/subImg_ls33D.nii.gz'
subImg_ls3 = splitImage(arry_LS, ls3, savePath_ls3, lsSpacing, lsOrigin, lsDirection)
savePath_ls4 = '../result/subImg_ls43D.nii.gz'
subImg_ls4 = splitImage(arry_LS, ls4, savePath_ls4, lsSpacing, lsOrigin, lsDirection)

"""label overlapping areas"""
overlap_mr12 = mr2 - mr1
overlap_mr12[overlap_mr12 != 1] = 0
overlap_mr23 = mr3 - mr2
overlap_mr23[overlap_mr23 != 1] = 0
overlap_mr13 = mr3 - mr1
overlap_mr13[overlap_mr13 != 2] = 0
overlap_mr34 = mr4 - mr3
overlap_mr34[overlap_mr34 != 1] = 0
overlap_mr = overlap_mr12 + overlap_mr23 + overlap_mr13 + overlap_mr34

overlap_ls12 = ls2 - ls1
overlap_ls12[overlap_ls12 != 1] = 0
overlap_ls23 = ls3 - ls2
overlap_ls23[overlap_ls23 != 1] = 0
overlap_ls13 = ls3 - ls1
overlap_ls13[overlap_ls13 != 2] = 0
overlap_ls34 = ls4 - ls3
overlap_ls34[overlap_ls34 != 1] = 0
overlap_ls = overlap_ls12 + overlap_ls23 + overlap_ls13 + overlap_ls34

#overlap_mr12 = dilation_mr2 - dilation_mr1
#overlap_mr12[overlap_mr12 != 1] = 0
#overlap_mr23 = dilation_mr3 - dilation_mr2
#overlap_mr23[overlap_mr23 != 1] = 0
#overlap_mr = overlap_mr12 + overlap_mr23
#
#overlap_ls12 = dilation_ls2 - dilation_ls1
#overlap_ls12[overlap_ls12 != 1] = 0
#overlap_ls23 = dilation_ls3 - dilation_ls2
#overlap_ls23[overlap_ls23 != 1] = 0
#overlap_ls = overlap_ls12 + overlap_ls23

############################################################
fix_img = ants.image_read('../result/input/mrisitk3D.nii.gz', 3)
fix = fix_img.numpy()
move_img = ants.image_read('../result/input/lssitk3D.nii.gz', 3)
dst = move_img.numpy()

"""be careful"""
#dst[dst<30]=0  
#dst[:,180:256]=0
mrcopy = mr.copy()
lscopy = ls.copy()

mrcopy[mrcopy!=5] = 1 # label 1234
mrcopy[mrcopy==5] = 0 # label 1234
fix = fix * mrcopy
lscopy[lscopy!=5] = 1 # label 1234
lscopy[lscopy==5] = 0 # label 1234
dst = dst * lscopy

fix_img1 = ants.image_read('C:/Users/fangyan/Desktop/PiecewiseRegistration/result/subImg_mr13D.nii.gz', 3)
move_img1 = ants.image_read('C:/Users/fangyan/Desktop/PiecewiseRegistration/result/subImg_ls13D.nii.gz', 3)
fix_img2 = ants.image_read('C:/Users/fangyan/Desktop/PiecewiseRegistration/result/subImg_mr23D.nii.gz', 3)
move_img2 = ants.image_read('C:/Users/fangyan/Desktop/PiecewiseRegistration/result/subImg_ls23D.nii.gz', 3)
fix_img3 = ants.image_read('C:/Users/fangyan/Desktop/PiecewiseRegistration/result/subImg_mr33D.nii.gz', 3)
move_img3 = ants.image_read('C:/Users/fangyan/Desktop/PiecewiseRegistration/result/subImg_ls33D.nii.gz', 3)
fix_img4 = ants.image_read('C:/Users/fangyan/Desktop/PiecewiseRegistration/result/subImg_mr43D.nii.gz', 3)
move_img4 = ants.image_read('C:/Users/fangyan/Desktop/PiecewiseRegistration/result/subImg_ls43D.nii.gz', 3)
img1 = fix_img1.numpy()
img2 = fix_img2.numpy()
img3 = fix_img3.numpy()
img4 = fix_img4.numpy()
dst1 = move_img1.numpy()
dst2 = move_img2.numpy()
dst3 = move_img3.numpy()
dst4 = move_img4.numpy()

THICK, LENGTH, WIDTH = img1.shape
tem_THICK, tem_WIDTH, tem_LENGTH = img1.shape

#mrImg1 = img1[SLIDE-30,:,:].copy()
#mrImg1 = np.fliplr(mrImg1)
#mrImg1 = mrImg1.transpose((1,0))
#lsImg1 = dst1[SLIDE-30,:,:].copy()
#lsImg1 = np.fliplr(lsImg1)
#lsImg1 = lsImg1.transpose((1,0))
#
#plt.figure(3,figsize=(6, 8))
#plt.subplot(5, 1, 2)
#plt.title('subImg1(MR, Light-sheet)')
#plt.ylim(0, WIDTH-1)
#plt.axis('off') 
#plt.imshow(np.hstack((mrImg1, lsImg1)),cmap = 'gray') 

mrImg2 = img2[SLIDE,:,:].copy()
mrImg2 = np.fliplr(mrImg2)
mrImg2 = mrImg2.transpose((1,0))
lsImg2 = dst2[SLIDE,:,:].copy()
lsImg2 = np.fliplr(lsImg2)
lsImg2 = lsImg2.transpose((1,0))

plt.figure(3,figsize=(6, 8))
plt.subplot(5, 1, 2)
plt.title('subImg2(MR, Light-sheet)')
plt.ylim(0, WIDTH-1)
plt.axis('off') 
plt.imshow(np.hstack((mrImg2, lsImg2)),cmap = 'gray') 

mrImg3 = img3[SLIDE,:,:].copy()
mrImg3 = np.fliplr(mrImg3)
mrImg3 = mrImg3.transpose((1,0))
lsImg3 = dst3[SLIDE,:,:].copy()
lsImg3 = np.fliplr(lsImg3)
lsImg3 = lsImg3.transpose((1,0))

plt.figure(3,figsize=(6, 8))
plt.subplot(5, 1, 3)
plt.title('subImg3(MR, Light-sheet)')
plt.ylim(0, WIDTH-1)
plt.axis('off') 
plt.imshow(np.hstack((mrImg3, lsImg3)),cmap = 'gray') 

mrImg4 = img4[SLIDE,:,:].copy() # be careful
mrImg4 = np.fliplr(mrImg4)
mrImg4 = mrImg4.transpose((1,0))
lsImg4 = dst4[SLIDE,:,:].copy()
lsImg4 = np.fliplr(lsImg4)
lsImg4 = lsImg4.transpose((1,0))

plt.figure(3,figsize=(6, 8))
plt.subplot(5, 1, 4)
plt.title('subImg4(MR, Light-sheet)')
plt.ylim(0, WIDTH-1)
plt.axis('off') 
plt.imshow(np.hstack((mrImg4, lsImg4)),cmap = 'gray') 

plt.figure(3,figsize=(6, 8))
plt.subplot(5, 1, 1)
plt.title('Image-PostProcessing(MR, Light-sheet)')
plt.ylim(0, WIDTH-1)
plt.axis('off') 
plt.imshow(np.hstack((mrImg2+mrImg3+mrImg4,lsImg2+lsImg3+lsImg4)),cmap = 'gray') 

"""Piecewise affine registration"""
regisMethod = 'Affine'
#regisMethod = 'Rigid'

outs1 = ants.registration(fix_img1, move_img1, type_of_transform = regisMethod)  
reg_img1 = outs1['warpedmovout'] 
arry1 = reg_img1.numpy()

outs2 = ants.registration(fix_img2, move_img2, type_of_transform = regisMethod)  
reg_img2 = outs2['warpedmovout'] 
arry2 = reg_img2.numpy()

outs3 = ants.registration(fix_img3, move_img3, type_of_transform = regisMethod)  
reg_img3 = outs3['warpedmovout'] 
arry3 = reg_img3.numpy()

outs4 = ants.registration(fix_img4, move_img4, type_of_transform = regisMethod)  
reg_img4 = outs4['warpedmovout'] 
arry4 = reg_img4.numpy()

plt.figure(4, figsize=(7, 6))
plt.subplot(2, 3, 1)
plt.title('ANTsResult')
plt.ylim(0, 255)

arry12_overlap = (arry1 * overlap_mr12) + (arry2 * overlap_mr12)
arry23_overlap = (arry2 * overlap_mr23) + (arry3 * overlap_mr23)
arry13_overlap = (arry1 * overlap_mr13) + (arry3 * overlap_mr13)
arry34_overlap = (arry3 * overlap_mr34) + (arry4 * overlap_mr34)
arry_overlap = arry12_overlap/2 + arry23_overlap/2 + arry13_overlap/2 + arry34_overlap/2

#arry1_overlap = arry1 * overlap_mr12
#arry12_overlap = arry2 * overlap_mr12
#arry23_overlap = arry2 * overlap_mr23
#arry3_overlap = arry3 * overlap_mr23
#arry_overlap = (arry1_overlap + arry12_overlap)/2 + (arry23_overlap + arry3_overlap)/2
 
mask = np.zeros_like(mr)
mask1 = np.zeros_like(mr)
mask2 = np.zeros_like(mr)
mask3 = np.zeros_like(mr)
mask4 = np.zeros_like(mr)

mask,distanceMap = maskExpand3D(mr)
mask1[mask!=1] = 0
mask1[mask==1] = 1
mask2[mask!=2] = 0
mask2[mask==2] = 1
mask3[mask!=3] = 0
mask3[mask==3] = 1
mask4[mask!=4] = 0
mask4[mask==4] = 1

#for i in range(0, mr.shape[0]):
#    copy_labels_MR = maskExpand3Dprevious(mr[i,:,:], maxContourmr[i,:,:], label) # mrMask: background=1,interior=0
#    mask1[i,:,:][copy_labels_MR==1] = 1
#    mask2[i,:,:][copy_labels_MR==2] = 1
#    mask3[i,:,:][copy_labels_MR==3] = 1
#    mask[i,:,:] = copy_labels_MR

arry_plus = norm(arry1) * mask1 + norm(arry2) * mask2 + norm(arry3) * mask3 + norm(arry4) * mask4
arry_mius = arry_plus * overlap_mr
arry1234 = arry_plus - arry_mius + arry_overlap

#show_arry = ((arry1+arry2+arry3+arry4)[SLIDE,:,:]).copy()
show_arry = ((arry1234)[SLIDE,:,:]).copy()
show_arry = np.fliplr(show_arry)
show_arry = show_arry.transpose((1,0))
plt.ylim(0, WIDTH-1)
plt.axis('off') 
plt.imshow(show_arry, cmap = 'gray')

"""Deformation field"""
# Displacement in region 1.
affine_trans1 = scio.loadmat(outs1['invtransforms'][0])['AffineTransform_float_3_3'][0:9].reshape(3, 3, order = 'F').T
affine_trans1 = np.column_stack((affine_trans1,scio.loadmat(outs1['invtransforms'][0])['AffineTransform_float_3_3'][9:12]))
add = np.array([0, 0, 0, 1])
affine1 = np.row_stack((affine_trans1, add)) 
center1 = scio.loadmat(outs1['invtransforms'][0])['fixed']
affine1 = ant2mat3D(affine1, center1, mrOrigin, mrSpacing, mrDirection) # mov * affine1 = target
field1 = warp_field3D(affine1, dst1.shape, mrOrigin, mrSpacing, mrDirection)

# Displacement in region 2.
affine_trans2 = scio.loadmat(outs2['invtransforms'][0])['AffineTransform_float_3_3'][0:9].reshape(3, 3, order = 'F').T
affine_trans2 = np.column_stack((affine_trans2,scio.loadmat(outs2['invtransforms'][0])['AffineTransform_float_3_3'][9:12]))
add = np.array([0, 0, 0, 1])
affine2 = np.row_stack((affine_trans2, add)) 
center2 = scio.loadmat(outs2['invtransforms'][0])['fixed']
affine2 = ant2mat3D(affine2, center2, mrOrigin, mrSpacing, mrDirection) # mov * affine2 = target
field2 = warp_field3D(affine2, dst2.shape, mrOrigin, mrSpacing, mrDirection)

# Displacement in region 3.
affine_trans3 = scio.loadmat(outs3['invtransforms'][0])['AffineTransform_float_3_3'][0:9].reshape(3, 3, order = 'F').T
affine_trans3 = np.column_stack((affine_trans3,scio.loadmat(outs3['invtransforms'][0])['AffineTransform_float_3_3'][9:12]))
add = np.array([0, 0, 0, 1])
affine3 = np.row_stack((affine_trans3, add)) 
center3 = scio.loadmat(outs3['invtransforms'][0])['fixed']
affine3 = ant2mat3D(affine3, center3, mrOrigin, mrSpacing, mrDirection) # mov * affine2 = target
field3 = warp_field3D(affine3, dst3.shape, mrOrigin, mrSpacing, mrDirection)

# Displacement in region 4.
affine_trans4 = scio.loadmat(outs4['invtransforms'][0])['AffineTransform_float_3_3'][0:9].reshape(3, 3, order = 'F').T
affine_trans4 = np.column_stack((affine_trans4,scio.loadmat(outs4['invtransforms'][0])['AffineTransform_float_3_3'][9:12]))
add = np.array([0, 0, 0, 1])
affine4 = np.row_stack((affine_trans4, add)) 
center4 = scio.loadmat(outs4['invtransforms'][0])['fixed']
affine4 = ant2mat3D(affine4, center4, mrOrigin, mrSpacing, mrDirection) # mov * affine2 = target
field4 = warp_field3D(affine4, dst4.shape, mrOrigin, mrSpacing, mrDirection)

## Displacement in region 1.
#field1 = np.swapaxes(sitk.GetArrayFromImage(sitk.ReadImage(outs1['invtransforms'][1])),0,2)
## Displacement in region 2.
#field2 = np.swapaxes(sitk.GetArrayFromImage(sitk.ReadImage(outs2['invtransforms'][1])),0,2)
## Displacement in region 3.
#field3 = np.swapaxes(sitk.GetArrayFromImage(sitk.ReadImage(outs3['invtransforms'][1])),0,2)
## Displacement in region 4.
#field4 = np.swapaxes(sitk.GetArrayFromImage(sitk.ReadImage(outs4['invtransforms'][1])),0,2)

"""blending displacement fields"""
# field: 
fieldXYZ = np.zeros_like(field1) # thick * 378 * 256 * 3 [z,x,y]
for i in range(0, 3):
    field12_overlap = (field1[:,:,:,i] * overlap_mr12) + (field2[:,:,:,i] * overlap_mr12)
    field23_overlap = (field2[:,:,:,i] * overlap_mr23) + (field3[:,:,:,i] * overlap_mr23)
    field13_overlap = (field1[:,:,:,i] * overlap_mr13) + (field3[:,:,:,i] * overlap_mr13)
    field34_overlap = (field3[:,:,:,i] * overlap_mr34) + (field4[:,:,:,i] * overlap_mr34)
    field_overlap = field12_overlap/2 + field23_overlap/2 + field13_overlap/2 + field34_overlap/2
    
    field1[:,:,:,i] = field1[:,:,:,i] * mask1
    field2[:,:,:,i] = field2[:,:,:,i] * mask2
    field3[:,:,:,i] = field3[:,:,:,i] * mask3
    field4[:,:,:,i] = field4[:,:,:,i] * mask4
    field_plus = field1[:,:,:,i] + field2[:,:,:,i] + field3[:,:,:,i] + field4[:,:,:,i]
    field_mius = field_plus * overlap_mr
    
    fieldXYZ[:,:,:,i] = field_plus - field_mius + field_overlap

#release memory
del field1,field2,field3,field4
gc.collect()

plt.figure(5,figsize=(6, 8))
plt.subplot(4,1,2)
plt.title('field (Z, Y, X) of 1')
show_fieldz = fieldXYZ[SLIDE,:,:,0].copy()
show_fieldz = np.fliplr(show_fieldz)
show_fieldz = show_fieldz.transpose((1,0))
show_fieldy = fieldXYZ[SLIDE,:,:,1].copy()
show_fieldy = np.fliplr(show_fieldy)
show_fieldy = show_fieldy.transpose((1,0))
show_fieldx = fieldXYZ[SLIDE,:,:,2].copy()
show_fieldx = np.fliplr(show_fieldx)
show_fieldx = show_fieldx.transpose((1,0))
plt.ylim(0, WIDTH-1)
mat = plt.imshow(np.hstack((show_fieldz, show_fieldy, show_fieldx)), vmin=-50, vmax=20,cmap='viridis') 
plt.axis('off') 
ax = plt.gca()
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(mat, cax = cax, spacing='uniform')

# just for visiualization
plt.figure(15,figsize=(6, 8))
plt.subplot(4,1,2)
plt.title('field (Z, Y, X) of 1')
show_fieldz = fieldXYZ[SLIDE,:,:,0].copy()
show_fieldz = np.fliplr(show_fieldz)
show_fieldz = show_fieldz.transpose((1,0))
show_fieldy = fieldXYZ[SLIDE,:,:,1].copy()
show_fieldy = np.fliplr(show_fieldy)
show_fieldy = show_fieldy.transpose((1,0))
show_fieldx = fieldXYZ[SLIDE,:,:,2].copy()
show_fieldx = np.fliplr(show_fieldx)
show_fieldx = show_fieldx.transpose((1,0))
plt.ylim(0, WIDTH-1)
mat = plt.imshow(np.hstack((show_fieldz, show_fieldy, show_fieldx)), vmin=-50, vmax=20,cmap='viridis') 
plt.axis('off') 
ax = plt.gca()
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(mat, cax = cax, spacing='uniform')

"""mask for displacement fields"""
#  fieldMaskGaussian
dilationSigma = 4
fieldmaskgaussian, dilation_maskEdges = fieldMaskGaussian3D(fieldXYZ, mask, dilationSigma)

plt.figure(5,figsize=(6, 8))
plt.subplot(4,1,3)
plt.title('field-Mask-Gaussian (Z, X, Y) of 1')
show_fieldz = fieldmaskgaussian[SLIDE,:,:,0].copy()
show_fieldz = np.fliplr(show_fieldz)
show_fieldz = show_fieldz.transpose((1,0))
show_fieldx = fieldmaskgaussian[SLIDE,:,:,1].copy()
show_fieldx = np.fliplr(show_fieldx)
show_fieldx = show_fieldx.transpose((1,0))
show_fieldy = fieldmaskgaussian[SLIDE,:,:,2].copy()
show_fieldy = np.fliplr(show_fieldy)
show_fieldy = show_fieldy.transpose((1,0))
plt.ylim(0, WIDTH-1)
mat = plt.imshow(np.hstack((show_fieldz, show_fieldx, show_fieldy)), vmin=-50, vmax=20,cmap='viridis') 
plt.axis('off') 
ax = plt.gca()
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(mat, cax = cax, spacing='uniform')

plt.figure(5, figsize=(6, 8))
plt.subplot(4,1,1)
plt.title('mask (original, Gaussian[WIDTH: %s]) of 1'%dilationSigma)
visualOriginal = mask[SLIDE,:,:].copy()
visualOriginal = np.fliplr(visualOriginal)
visualOriginal = visualOriginal.transpose((1,0))
visualLaplace = dilation_maskEdges[SLIDE,:,:].copy()
visualLaplace = np.fliplr(visualLaplace)
visualLaplace = visualLaplace.transpose((1,0))
plt.ylim(0, WIDTH-1)
cmap = mpl.colors.ListedColormap([RED,PURPLE,BROWN,LIGHT_PURPLE,LIGHT_BROWN])                                  
mat = plt.imshow(np.hstack((visualOriginal, visualLaplace+4)), cmap) 
plt.axis('off') 
ax = plt.gca()
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
#unique_data = np.unique(visual);
unique_data = np.array([1, 2, 3, 4, 5], dtype='int8')
plt.colorbar(mat, cax = cax, ticks=np.arange(np.min(unique_data),np.max(unique_data)+1), spacing='uniform')

#  fieldGaussian
fieldgaussian = fieldGaussian3D(fieldXYZ)

plt.figure(5,figsize=(6, 8))
plt.subplot(4,1,4)
plt.title('field-Gaussian (Z, X, Y) of 1')
show_fieldz = fieldgaussian[SLIDE,:,:,0].copy()
show_fieldz = np.fliplr(show_fieldz)
show_fieldz = show_fieldz.transpose((1,0))
show_fieldx = fieldgaussian[SLIDE,:,:,1].copy()
show_fieldx = np.fliplr(show_fieldx)
show_fieldx = show_fieldx.transpose((1,0))
show_fieldy = fieldgaussian[SLIDE,:,:,2].copy()
show_fieldy = np.fliplr(show_fieldy)
show_fieldy = show_fieldy.transpose((1,0))
plt.ylim(0, WIDTH-1)
mat = plt.imshow(np.hstack((show_fieldz, show_fieldx, show_fieldy)), vmin=-50, vmax=20,cmap='viridis') 
plt.axis('off') 
ax = plt.gca()
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(mat, cax = cax, spacing='uniform')

# fieldLaplace
fieldlaplace = fieldLaplace3D(fieldXYZ)

plt.figure(15,figsize=(6, 8))
plt.subplot(4,1,4)
plt.title('field-Laplace (Z, X, Y) of 1')
show_fieldz = fieldlaplace[SLIDE,:,:,0].copy()
show_fieldz = np.fliplr(show_fieldz)
show_fieldz = show_fieldz.transpose((1,0))
show_fieldx = fieldlaplace[SLIDE,:,:,1].copy()
show_fieldx = np.fliplr(show_fieldx)
show_fieldx = show_fieldx.transpose((1,0))
show_fieldy = fieldlaplace[SLIDE,:,:,2].copy()
show_fieldy = np.fliplr(show_fieldy)
show_fieldy = show_fieldy.transpose((1,0))
plt.ylim(0, WIDTH-1)
mat = plt.imshow(np.hstack((show_fieldz, show_fieldx, show_fieldy)), vmin=-50, vmax=20,cmap='viridis') 
plt.axis('off') 
ax = plt.gca()
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(mat, cax = cax, spacing='uniform')

#  fieldMaskLaplace
dilationSigma = 4
fieldmasklaplace, dilation_maskEdges = fieldMaskLaplace3D(fieldXYZ, mask, dilationSigma)

plt.figure(15,figsize=(6, 8))
plt.subplot(4,1,3)
plt.title('field-maskLaplace (Z,X, Y) of 1')
show_fieldz = fieldmasklaplace[SLIDE,:,:,0].copy()
show_fieldz = np.fliplr(show_fieldz)
show_fieldz = show_fieldz.transpose((1,0))
show_fieldx = fieldmasklaplace[SLIDE,:,:,1].copy()
show_fieldx = np.fliplr(show_fieldx)
show_fieldx = show_fieldx.transpose((1,0))
show_fieldy = fieldmasklaplace[SLIDE,:,:,2].copy()
show_fieldy = np.fliplr(show_fieldy)
show_fieldy = show_fieldy.transpose((1,0))
plt.ylim(0, WIDTH-1)
mat = plt.imshow(np.hstack((show_fieldz, show_fieldx, show_fieldy)), vmin=-50, vmax=20,cmap='viridis') 
plt.axis('off') 
ax = plt.gca()
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(mat, cax = cax, spacing='uniform')

plt.figure(15, figsize=(6, 8))
plt.subplot(4,1,1)
plt.title('mask (original, Laplace[WIDTH: %s]) of 1'%dilationSigma)
visualOriginal = mask[SLIDE,:,:].copy()
visualOriginal = np.fliplr(visualOriginal)
visualOriginal = visualOriginal.transpose((1,0))
visualLaplace = dilation_maskEdges[SLIDE,:,:].copy()
visualLaplace = np.fliplr(visualLaplace)
visualLaplace = visualLaplace.transpose((1,0))
plt.ylim(0, WIDTH-1)
cmap = mpl.colors.ListedColormap([RED,PURPLE,BROWN,LIGHT_PURPLE,LIGHT_BROWN])                                  
mat = plt.imshow(np.hstack((visualOriginal, visualLaplace+4)), cmap) 
plt.axis('off') 
ax = plt.gca()
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
#unique_data = np.unique(visual);
unique_data = np.array([1, 2, 3, 4, 5], dtype='int8')
plt.colorbar(mat, cax = cax, ticks=np.arange(np.min(unique_data),np.max(unique_data)+1), spacing='uniform')

"""warp(backward)"""
arryresult = warp_fieldtoimage3D(dst, fieldXYZ, [tem_THICK, tem_WIDTH,tem_LENGTH], mrOrigin, mrSpacing, mrDirection) # ANTs field
arryresult_maskGaussian = warp_fieldtoimage3D(dst, fieldmaskgaussian, [tem_THICK, tem_WIDTH,tem_LENGTH], mrOrigin, mrSpacing, mrDirection) # ANTs mask-Gussian-field
arryresult_Gaussian = warp_fieldtoimage3D(dst, fieldgaussian, [tem_THICK, tem_WIDTH,tem_LENGTH], mrOrigin, mrSpacing, mrDirection) # ANTs nonmask-Gussian-field
arryresult_Laplace = warp_fieldtoimage3D(dst, fieldlaplace, [tem_THICK, tem_WIDTH,tem_LENGTH], mrOrigin, mrSpacing, mrDirection) # ANTs field-Laplace
arryresult_maskLaplace = warp_fieldtoimage3D(dst, fieldmasklaplace, [tem_THICK, tem_WIDTH,tem_LENGTH], mrOrigin, mrSpacing, mrDirection) # ANTs field-Laplace

plt.figure(4, figsize=(7, 6))
plt.subplot(2, 3, 2)
plt.title('ourResult (overlap-average) of 1')
arryresultarryresult = np.fliplr(arryresult[SLIDE,:,:].copy())
arryresultarryresult = arryresultarryresult.transpose((1,0))
plt.ylim(0, WIDTH-1)
plt.axis('off')
plt.imshow(arryresultarryresult, cmap='gray')

plt.figure(4, figsize=(7, 6))
plt.subplot(2, 3, 3)
plt.title('ourResult (mask-Gussian-field) of 1')
arryresultarryresult = np.fliplr(arryresult_maskGaussian[SLIDE,:,:].copy())
arryresultarryresult = arryresultarryresult.transpose((1,0))
plt.ylim(0, WIDTH-1)
plt.axis('off')
plt.imshow(arryresultarryresult, cmap='gray')

plt.figure(4, figsize=(7, 6))
plt.subplot(2, 3, 4)
plt.title('ourResult (nonmask-Gussian-field) of 1')
arryresultarryresult = np.fliplr(arryresult_Gaussian[SLIDE,:,:].copy())
arryresultarryresult = arryresultarryresult.transpose((1,0))
plt.ylim(0, WIDTH-1)
plt.axis('off')
plt.imshow(arryresultarryresult, cmap='gray')

plt.figure(4, figsize=(7, 6))
plt.subplot(2, 3, 5)
plt.title('ourResult (nonmask-Laplace-field) of 1')
arryresultarryresult = np.fliplr(arryresult_Laplace[SLIDE,:,:].copy())
arryresultarryresult = arryresultarryresult.transpose((1,0))
plt.ylim(0, WIDTH-1)
plt.axis('off')
plt.imshow(arryresultarryresult, cmap='gray')

plt.figure(4, figsize=(7, 6))
plt.subplot(2, 3, 6)
plt.title('ourResult (mask-Laplace-field) of 1')
arryresultarryresult = np.fliplr(arryresult_maskLaplace[SLIDE,:,:].copy())
arryresultarryresult = arryresultarryresult.transpose((1,0))
plt.ylim(0, WIDTH-1)
plt.axis('off')
plt.imshow(arryresultarryresult, cmap='gray')

outs = ants.registration(ants.from_numpy(arry_MR), ants.from_numpy(arry_LS), type_of_transform = 'Affine')  # whole image using ants
reg_img = outs['warpedmovout'] 
arry = reg_img.numpy()
niisave(arry, origin = sitk_LS.GetOrigin(), spacing = sitk_LS.GetSpacing(), dire=sitk_LS.GetDirection(), path = '../result/save/wholeImageantAffineResult.nii.gz')

outs = ants.registration(ants.from_numpy(arry_MR), ants.from_numpy(arry_LS), type_of_transform = 'SyN')  # whole image using ants
reg_img = outs['warpedmovout'] 
arry = reg_img.numpy()
niisave(arry, origin = sitk_LS.GetOrigin(), spacing = sitk_LS.GetSpacing(), dire=sitk_LS.GetDirection(), path = '../result/save/wholeImageantSyNResult.nii.gz')


'''nifty save'''
#niisave(norm(arry1+arry2+arry3+arry4), origin = sitk_LS.GetOrigin(), spacing = sitk_LS.GetSpacing(), dire=sitk_LS.GetDirection(), path = '../result/save/antAffinePiecewiseResult.nii.gz')
niisave(arry1234, origin = sitk_LS.GetOrigin(), spacing = sitk_LS.GetSpacing(), dire=sitk_LS.GetDirection(), path = '../result/save/antAffinePiecewiseResult.nii.gz')
niisave(arryresult, origin = sitk_LS.GetOrigin(), spacing = sitk_LS.GetSpacing(), dire=sitk_LS.GetDirection(), path = '../result/save/overlapFieldPiecewiseResult.nii.gz')
niisave(arryresult_maskGaussian, origin = sitk_LS.GetOrigin(), spacing = sitk_LS.GetSpacing(), dire=sitk_LS.GetDirection(), path = '../result/save/maskGaussianPiecewiseResult.nii.gz')
niisave(arryresult_Gaussian, origin = sitk_LS.GetOrigin(), spacing = sitk_LS.GetSpacing(), dire=sitk_LS.GetDirection(), path = '../result/save/GaussianPiecewiseResult.nii.gz')
niisave(arryresult_Laplace, origin = sitk_LS.GetOrigin(), spacing = sitk_LS.GetSpacing(), dire=sitk_LS.GetDirection(), path = '../result/save/LaplacePiecewiseResult.nii.gz')
niisave(arryresult_maskLaplace, origin = sitk_LS.GetOrigin(), spacing = sitk_LS.GetSpacing(), dire=sitk_LS.GetDirection(), path = '../result/save/maskLaplacePiecewiseResult.nii.gz')

niisave(mask, origin = sitk_LS.GetOrigin(), spacing = sitk_LS.GetSpacing(), dire=sitk_LS.GetDirection(), path = '../result/save/mask.nii.gz')

niisave(fieldXYZ[:,:,:,0], origin = sitk_LS.GetOrigin(), spacing = sitk_LS.GetSpacing(), dire=sitk_LS.GetDirection(), path = '../result/save/Zfield.nii.gz')
niisave(fieldXYZ[:,:,:,1], origin = sitk_LS.GetOrigin(), spacing = sitk_LS.GetSpacing(), dire=sitk_LS.GetDirection(), path = '../result/save/Xfield.nii.gz')
niisave(fieldXYZ[:,:,:,2], origin = sitk_LS.GetOrigin(), spacing = sitk_LS.GetSpacing(), dire=sitk_LS.GetDirection(), path = '../result/save/Yfield.nii.gz')

niisave(arry_MR.transpose(2,1,0), origin = sitk_LS.GetOrigin(), spacing = sitk_LS.GetSpacing(), dire=sitk_LS.GetDirection(), path = '../result/save/mr.nii.gz')



"""visiualization"""
#x,y,z = np.meshgrid(np.arange(dst.shape[1]), np.arange(dst.shape[0]), np.arange(dst.shape[2]))
#dx = fieldmasklaplace[:,:,:,0]
#dy = fieldmasklaplace[:,:,:,1]
#dz = fieldmasklaplace[:,:,:,2]
#plt.figure(20)
#plt.quiver(x[::20,::20,122], y[::20,::20,122], -dx[::20,::20,122], -dy[::20,::20,122],color='red')
#plt.imshow(dst[:,:,122])
#
#x,y,z = np.meshgrid(np.arange(dst.shape[1]), np.arange(dst.shape[0]), np.arange(dst.shape[2]))
#dx = fieldmasklaplace[:,:,:,0]
#dy = fieldmasklaplace[:,:,:,1]
#dz = fieldmasklaplace[:,:,:,2]
#plt.figure(20)
#fig, ax = plt.subplots(figsize=(13,7))
#ax.quiver(x[::10,::10,105], y[::10,::10,105], dy[::10,::10,105], dz[::10,::10,105],color='red')
#
##ax.xaxis.set_ticks([])
##ax.yaxis.set_ticks([])
#ax.set_aspect('equal')
#plt.figure(21)
#plt.imshow(dz[:,:,122])
#


import sys
sys.path.insert(0, './PiecewiseRegistration')
import gui

outs = ants.registration(ants.from_numpy(fix), ants.from_numpy(dst), type_of_transform = 'Affine')  
reg_img = outs['warpedmovout'] 
arry = reg_img.numpy()

fix_sitk = sitk.GetImageFromArray(fix.transpose(0,2,1))
dst_sitk = sitk.GetImageFromArray(dst.transpose(0,2,1))
ants_wholeresult_sitk = sitk.GetImageFromArray(arry.transpose(0,2,1))
ants_result_sitk = sitk.GetImageFromArray(arry1234.transpose(0,2,1))
result_sitk = sitk.GetImageFromArray(arryresult_maskLaplace.transpose(0,2,1))

fix_255 = sitk.Cast(sitk.IntensityWindowing(fix_sitk, windowMinimum=0.0, windowMaximum=0.89, 
                                             outputMinimum=0.0, outputMaximum=255.0), sitk.sitkUInt8)
dst_255 = sitk.Cast(sitk.IntensityWindowing(dst_sitk, windowMinimum=0.0, windowMaximum=437.92, 
                                             outputMinimum=0.0, outputMaximum=255.0), sitk.sitkUInt8)
ants_wholeresult_255 = sitk.Cast(sitk.IntensityWindowing(ants_wholeresult_sitk, windowMinimum=0.0, windowMaximum=424.4, 
                                             outputMinimum=0.0, outputMaximum=255.0), sitk.sitkUInt8)
ants_result_255 = sitk.Cast(sitk.IntensityWindowing(ants_result_sitk, windowMinimum=0.0, windowMaximum=2.58,
                                             outputMinimum=0.0, outputMaximum=255.0), sitk.sitkUInt8)
result_255 = sitk.Cast(sitk.IntensityWindowing(result_sitk, windowMinimum=0.0, windowMaximum=437.92,
                                             outputMinimum=0.0, outputMaximum=255.0), sitk.sitkUInt8)

gui.MultiImageDisplay(image_list = [sitk.Cast(sitk.Compose(dst_255, fix_255, dst_255), sitk.sitkVectorUInt8), # magenta-green
                                    sitk.Cast(sitk.Compose(ants_wholeresult_255, fix_255, ants_wholeresult_255), sitk.sitkVectorUInt8), # magenta-green
                                    sitk.Cast(sitk.Compose(ants_result_255, fix_255, ants_result_255), sitk.sitkVectorUInt8), # magenta-green  
                                    sitk.Cast(sitk.Compose(result_255, fix_255, result_255), sitk.sitkVectorUInt8)], # magenta-green              
                      title_list= ['Original Overlay (Green:MR-Magenta:LS)', 
                                   'ants_wholeResult Overlay',
                                   'ants_Result Overlay',
                                   'Result Overlay'], 
                      figure_size=(22,5))