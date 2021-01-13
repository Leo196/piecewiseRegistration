# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 02:28:20 2020

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

from basicProcessing import norm
from basicProcessing import splitImage
from basicProcessing import ant2mat2D
from basicProcessing import landmarkGenerate
from basicProcessing import denoiseLabels
from basicProcessing import maskExpand

from affineWarp import affine
from affineWarp import warpAffine
from affineWarp import warp_field
from affineWarp import warp_fieldtoimage

from fieldLaplace import fieldLaplace
from fieldLaplace import fieldMaskLaplace
from fieldGaussian import fieldGaussian
from fieldGaussian import fieldMaskGaussian

#import re
#for x in dir():
#    if not re.match('^__',x) and x!="re":
#        exec(" ".join(("del",x)))


#plt.close()
WHITE = '#FFFFFF'
GREY = '#999999'
RED = '#E41A1C'
PURPLE = '#984EA3'
BROWN = '#A65628'
LIGHT_BROWN = '#C55A11'
LIGHT_PURPLE = '#C4A8A8'

"""extract image"""
lspath = '../data/01LS_to_FAMRI_fslAffine.nii.gz'
mripath = '../data/02raw_FAMRI_brainBET.nii.gz'
#
ls = sitk.ReadImage(lspath)
mri = sitk.ReadImage(mripath)

lsarry = sitk.GetArrayFromImage(ls)
mriarry = sitk.GetArrayFromImage(mri)

lsselect = lsarry[:,:,97]
mriselect = mriarry[:,:,102]

sitk_LS = sitk.GetImageFromArray(lsselect)
sitk_MR = sitk.GetImageFromArray(mriselect)
sitk_LS.SetSpacing(ls.GetSpacing())
sitk_LS.SetOrigin(ls.GetOrigin())
sitk_MR.SetSpacing(mri.GetSpacing())
sitk_MR.SetOrigin(mri.GetOrigin())
direction_mr = mri.GetDirection()
direction_ls = ls.GetDirection()

sitk.WriteImage(sitk_LS,'../lssitk.nii.gz')
sitk.WriteImage(sitk_MR,'../mrisitk.nii.gz')

"""read image"""
#MR = 'C:/Users/fangyan/Desktop/piecewiseregistration/mrisitk.nii.gz'
#LS = 'C:/Users/fangyan/Desktop/piecewiseregistration/lssitk.nii.gz'

#sitk_MR = sitk.ReadImage(MR)
#sitk_LS = sitk.ReadImage(LS)
arry_MR = sitk.GetArrayFromImage(sitk_MR)
arry_LS = sitk.GetArrayFromImage(sitk_LS)

arry_LS[arry_LS<30]=0
arry_LS[180:256,:]=0
arry_MR = norm(arry_MR)
arry_LS = norm(arry_LS)

plt.figure(1,figsize=(6, 8))
plt.subplot(4,1,1)
plt.title('Image (MR, Light-sheet)')
plt.imshow(np.hstack((arry_MR, arry_LS)),cmap='gray') 
plt.axis('off')
plt.show()

"""set landmarks"""
dis = 4
label = [1,2,3,4]
coordinate_mr = []
coordinate1 = [(119,107),(130,85),(123,85),(139,102)] # part 1
coordinate2 = [(105,199),(105,128),(60,181),(87,187),(134,155)] # part 2
coordinate3 = [(109,318),(147,282),(156,287),(73,294),(109,280),(97,282)] # part 3
coordinate4 = [(235,215),(35,86),(209,352),(28,352),(28,192),(39,330),(50,96)] # background
coordinate_mr.append(coordinate1)
coordinate_mr.append(coordinate2)
coordinate_mr.append(coordinate3)
coordinate_mr.append(coordinate4)
coordinate_ls = []
coordinate1 = [(119,107),(130,85),(123,85),(139,102)] # part 1
coordinate2 = [(105,199),(105,128),(60,181),(87,187),(134,155), (86,94), (89,103),(87,97),(86,95),(88,98),(89,100)] # part 2
coordinate3 = [(109,318),(147,282),(156,287),(73,294),(109,280),(97,282)] # part 3
coordinate4 = [(235,215),(35,86),(209,352),(28,352),(28,192),(39,330),(50,96),(100,89)] # background
coordinate_ls.append(coordinate1)
coordinate_ls.append(coordinate2)
coordinate_ls.append(coordinate3)
coordinate_ls.append(coordinate4)

shape = arry_MR.shape
shape = arry_LS.shape

landmarks_MR = landmarkGenerate(shape,dis,label,coordinate_mr,background=0) # just for visualization
landmarks_LS = landmarkGenerate(shape,dis,label,coordinate_ls,background=0) # just for visualization
landmarks_MR = landmarks_MR.astype(np.int8)
landmarks_LS = landmarks_LS.astype(np.int8)

plt.figure(1,figsize=(6, 8))
plt.subplot(4,1,2)
plt.title('Labelmarks(MR, Light-sheet)')
cmap = mpl.colors.ListedColormap([WHITE,RED,PURPLE,BROWN,GREY])
mat = plt.imshow(np.hstack((landmarks_MR,landmarks_LS)),cmap)
plt.axis('off') 
ax = plt.gca()
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
unique_data = np.unique(landmarks_MR);
plt.colorbar(mat, cax = cax, ticks=np.arange(np.min(unique_data),np.max(unique_data)+1), spacing='uniform')

landmarks_MR = landmarkGenerate(shape,dis,label,coordinate_mr,background=0) 
landmarks_LS = landmarkGenerate(shape,dis,label,coordinate_ls,background=0)
landmarks_MR = landmarks_MR.astype(np.int8)
landmarks_LS = landmarks_LS.astype(np.int8)

labels_MR = random_walker(arry_MR,landmarks_MR,beta = 3800,spacing=None).astype(np.int8)
labels_LS = random_walker(arry_LS,landmarks_LS,beta = 2000,spacing=None).astype(np.int8)

plt.figure(1,figsize=(6, 8))
plt.subplot(4,1,3)
plt.title('Labels(MR, Light-sheet)')
cmap = mpl.colors.ListedColormap([RED,PURPLE,BROWN,GREY])
mat = plt.imshow(np.hstack((labels_MR,labels_LS)),cmap) 
plt.axis('off') 
ax = plt.gca()
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
unique_data = np.unique(labels_MR);
plt.colorbar(mat, cax = cax, ticks=np.arange(np.min(unique_data),np.max(unique_data)+1), spacing='uniform')

"""PostProcessing"""
# MR
mr1 = labels_MR.copy()
mr1[mr1!=1] = 0
mr1[mr1==1] = 1
dilation_mr1, maxContour_mr1 = denoiseLabels(mr1)

mr2 = labels_MR.copy()
mr2[mr2!=2] = 0
mr2[mr2==2] = 1
dilation_mr2, maxContour_mr2 = denoiseLabels(mr2)

mr3 = labels_MR.copy()
mr3[mr3!=3] = 0
mr3[mr3==3] = 1
dilation_mr3, maxContour_mr3 = denoiseLabels(mr3)

mr = np.zeros_like(arry_MR) 
dilation_mr1[dilation_mr1==1] = 1 + 1
dilation_mr2[dilation_mr2==1] = 2 + 1
dilation_mr3[dilation_mr3==1] = 3 + 1
mr = mr + (dilation_mr1 + dilation_mr2 + dilation_mr3 - 1)
mr[mr==4] = 5
mr[mr==-1] = 4

maxContour_mr = maxContour_mr1 + maxContour_mr2 + maxContour_mr3;

# LS
ls1 = labels_LS.copy()
ls1[ls1!=1] = 0
ls1[ls1==1] = 1
dilation_ls1, maxContour_ls1 = denoiseLabels(ls1);

ls2 = labels_LS.copy()
ls2[ls2!=2] = 0
ls2[ls2==2] = 1
dilation_ls2, maxContour_ls2 = denoiseLabels(ls2);

ls3 = labels_LS.copy()
ls3[ls3!=3] = 0
ls3[ls3==3] = 1
dilation_ls3, maxContour_ls3 = denoiseLabels(ls3);

ls = np.zeros_like(arry_LS) 
dilation_ls1[dilation_ls1==1] = 1 + 1
dilation_ls2[dilation_ls2==1] = 2 + 1
dilation_ls3[dilation_ls3==1] = 3 + 1
ls = ls + (dilation_ls1 + dilation_ls2 + dilation_ls3 - 1)
ls[ls==4] = 5
ls[ls==-1] = 4

maxContour_ls = maxContour_ls1 + maxContour_ls2 + maxContour_ls3;

mr = mr.astype('int8');
ls = ls.astype('int8');

plt.figure(1,figsize=(6, 8))
plt.subplot(4,1,4)
plt.title('PostProcessing(MR, Light-sheet)')
#cmap = mpl.colors.ListedColormap([RED,PURPLE,BROWN,GREY,LIGHT_BROWN,LIGHT_PURPLE])
cmap = mpl.colors.ListedColormap([RED,PURPLE,BROWN,GREY])
mat = plt.imshow(np.hstack((mr,ls)),cmap) 
plt.axis('off') 
ax = plt.gca()
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
unique_data = np.unique(mr);
plt.colorbar(mat, cax = cax, ticks=np.arange(np.min(unique_data),np.max(unique_data)+1), spacing='uniform')

#plt.figure(3,figsize=(6, 8))
#plt.subplot(4,1,4)
#plt.title('Contour(MR, Light-sheet)')
#cmap = mpl.colors.ListedColormap([GREY,RED])
#mat = plt.imshow(np.hstack((maxContour_mr,maxContour_ls)),cmap) 
#plt.axis('off') 

plt.figure(1,figsize=(6, 8))
plt.subplot(4,1,2)
plt.title('Labelmarks(MR, Light-sheet)')
cmap = mpl.colors.ListedColormap([WHITE,RED,PURPLE,BROWN,GREY])
mat = plt.imshow(np.hstack((landmarks_MR+maxContour_mr,landmarks_LS+maxContour_ls)),cmap)
plt.axis('off') 
ax = plt.gca()
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
unique_data = np.unique(landmarks_MR);
plt.colorbar(mat, cax = cax, ticks=np.arange(np.min(unique_data),np.max(unique_data)+1), spacing='uniform')

"""Split image"""
spacing_mr = sitk_MR.GetSpacing()
origin_mr = sitk_MR.GetOrigin()
savePath_mr1 = 'C:/Users/fangyan/Desktop/PiecewiseRegistration/result/subImg_mr1.nii.gz'
subImg_mr1 = splitImage(arry_MR, dilation_mr1, savePath_mr1, spacing_mr, origin_mr)
savePath_mr2 = 'C:/Users/fangyan/Desktop/PiecewiseRegistration/result/subImg_mr2.nii.gz'
subImg_mr2 = splitImage(arry_MR, dilation_mr2, savePath_mr2, spacing_mr, origin_mr)
savePath_mr3 = 'C:/Users/fangyan/Desktop/PiecewiseRegistration/result/subImg_mr3.nii.gz'
subImg_mr3 = splitImage(arry_MR, dilation_mr3, savePath_mr3, spacing_mr, origin_mr)
#savePath_mr1 = 'C:/Users/fangyan/Desktop/PiecewiseRegistration/result/subImg_mr1.nii.gz'
#subImg_mr1 = splitImage(arry_MR, mr1, savePath_mr1, spacing_mr, origin_mr)
#savePath_mr2 = 'C:/Users/fangyan/Desktop/PiecewiseRegistration/result/subImg_mr2.nii.gz'
#subImg_mr2 = splitImage(arry_MR, mr2, savePath_mr2, spacing_mr, origin_mr)
#savePath_mr3 = 'C:/Users/fangyan/Desktop/PiecewiseRegistration/result/subImg_mr3.nii.gz'
#subImg_mr3 = splitImage(arry_MR, mr3, savePath_mr3, spacing_mr, origin_mr)

spacing_ls = sitk_LS.GetSpacing()
origin_ls = sitk_LS.GetOrigin()
savePath_ls1 = 'C:/Users/fangyan/Desktop/PiecewiseRegistration/result/subImg_ls1.nii.gz'
subImg_ls1 = splitImage(arry_LS, dilation_ls1, savePath_ls1, spacing_ls, origin_ls)
savePath_ls2 = 'C:/Users/fangyan/Desktop/PiecewiseRegistration/result/subImg_ls2.nii.gz'
subImg_ls2 = splitImage(arry_LS, dilation_ls2, savePath_ls2, spacing_ls, origin_ls)
savePath_ls3 = 'C:/Users/fangyan/Desktop/PiecewiseRegistration/result/subImg_ls3.nii.gz'
subImg_ls3 = splitImage(arry_LS, dilation_ls3, savePath_ls3, spacing_ls, origin_ls)
#savePath_ls1 = 'C:/Users/fangyan/Desktop/PiecewiseRegistration/result/subImg_ls1.nii.gz'
#subImg_ls1 = splitImage(arry_LS, ls1, savePath_ls1, spacing_ls, origin_ls)
#savePath_ls2 = 'C:/Users/fangyan/Desktop/PiecewiseRegistration/result/subImg_ls2.nii.gz'
#subImg_ls2 = splitImage(arry_LS, ls2, savePath_ls2, spacing_ls, origin_ls)
#savePath_ls3 = 'C:/Users/fangyan/Desktop/PiecewiseRegistration/result/subImg_ls3.nii.gz'
#subImg_ls3 = splitImage(arry_LS, ls3, savePath_ls3, spacing_ls, origin_ls)

#plt.imshow(subImg_mr1+subImg_mr2+subImg_mr3, cmap = 'gray')
#plt.imshow(subImg_ls1+subImg_ls2+subImg_ls3, cmap = 'gray')

"""label overlapping areas"""
overlap_mr12 = dilation_mr2 - dilation_mr1
overlap_mr12[overlap_mr12 != 1] = 0
overlap_mr23 = dilation_mr3 - dilation_mr2
overlap_mr23[overlap_mr23 != 1] = 0
overlap_mr = overlap_mr12 + overlap_mr23

overlap_ls12 = dilation_ls2 - dilation_ls1
overlap_ls12[overlap_ls12 != 1] = 0
overlap_ls23 = dilation_ls3 - dilation_ls2
overlap_ls23[overlap_ls23 != 1] = 0
overlap_ls = overlap_ls12 + overlap_ls23

############################################################

WIDTH = 256
LENGTH = 378
tem_WIDTH = 378
tem_LENGTH = 256

LIM_X = 205
LIM_Y = 200

fix_img = ants.image_read('C:/Users/fangyan/Desktop/piecewiseregistration/mrisitk.nii.gz', 2)
fix = fix_img.numpy()
move_img = ants.image_read('C:/Users/fangyan/Desktop/piecewiseregistration/lssitk.nii.gz', 2)
dst = move_img.numpy()

"""be careful"""
#dst[dst<30]=0  
#dst[:,180:256]=0
mrcopy = mr.copy()
lscopy = ls.copy()

mrcopy[mrcopy!=4] = 1
mrcopy[mrcopy==4] = 0
fix = fix * mrcopy.transpose((1,0))
lscopy[lscopy!=4] = 1
lscopy[lscopy==4] = 0
dst = dst * lscopy.transpose((1,0))

fix_img1 = ants.image_read('C:/Users/fangyan/Desktop/PiecewiseRegistration/result/subImg_mr1.nii.gz', 2)
move_img1 = ants.image_read('C:/Users/fangyan/Desktop/PiecewiseRegistration/result/subImg_ls1.nii.gz', 2)
fix_img2 = ants.image_read('C:/Users/fangyan/Desktop/PiecewiseRegistration/result/subImg_mr2.nii.gz', 2)
move_img2 = ants.image_read('C:/Users/fangyan/Desktop/PiecewiseRegistration/result/subImg_ls2.nii.gz', 2)
fix_img3 = ants.image_read('C:/Users/fangyan/Desktop/PiecewiseRegistration/result/subImg_mr3.nii.gz', 2)
move_img3 = ants.image_read('C:/Users/fangyan/Desktop/PiecewiseRegistration/result/subImg_ls3.nii.gz', 2)
img1 = fix_img1.numpy()
img2 = fix_img2.numpy()
img3 = fix_img3.numpy()
dst1 = move_img1.numpy()
dst2 = move_img2.numpy()
dst3 = move_img3.numpy()

LENGTH, WIDTH = img1.shape
tem_WIDTH, tem_LENGTH = img1.shape

mrImg1 = img1.copy()
mrImg1 = np.fliplr(mrImg1)
mrImg1 = mrImg1.transpose((1,0))
lsImg1 = dst1.copy()
lsImg1 = np.fliplr(lsImg1)
lsImg1 = lsImg1.transpose((1,0))

plt.figure(3,figsize=(6, 8))
plt.subplot(4, 1, 2)
plt.title('subImg1(MR, Light-sheet)')
plt.ylim(0, WIDTH-1)
plt.axis('off') 
plt.imshow(np.hstack((mrImg1, lsImg1)),cmap = 'gray') 

mrImg2 = img2.copy()
mrImg2 = np.fliplr(mrImg2)
mrImg2 = mrImg2.transpose((1,0))
lsImg2 = dst2.copy()
lsImg2 = np.fliplr(lsImg2)
lsImg2 = lsImg2.transpose((1,0))

plt.figure(3,figsize=(6, 8))
plt.subplot(4, 1, 3)
plt.title('subImg2(MR, Light-sheet)')
plt.ylim(0, WIDTH-1)
plt.axis('off') 
plt.imshow(np.hstack((mrImg2, lsImg2)),cmap = 'gray') 

mrImg3 = img3.copy()
mrImg3 = np.fliplr(mrImg3)
mrImg3 = mrImg3.transpose((1,0))
lsImg3 = dst3.copy()
lsImg3 = np.fliplr(lsImg3)
lsImg3 = lsImg3.transpose((1,0))

plt.figure(3,figsize=(6, 8))
plt.subplot(4, 1, 4)
plt.title('subImg3(MR, Light-sheet)')
plt.ylim(0, WIDTH-1)
plt.axis('off') 
plt.imshow(np.hstack((mrImg3, lsImg3)),cmap = 'gray') 

plt.figure(3,figsize=(6, 8))
plt.subplot(4, 1, 1)
plt.title('Image-PostProcessing(MR, Light-sheet)')
plt.ylim(0, WIDTH-1)
plt.axis('off') 
plt.imshow(np.hstack((mrImg1+mrImg2+mrImg3,lsImg1+lsImg2+lsImg3)),cmap = 'gray') 

"""Piecewise affine registration"""
fix_img1 = ants.from_numpy(img1)
move_img1 = ants.from_numpy(dst1)
outs1 = ants.registration(fix_img1, move_img1, type_of_transform = 'SyN')  
reg_img1 = outs1['warpedmovout'] 
arry1 = reg_img1.numpy()

fix_img2 = ants.from_numpy(img2)
move_img2 = ants.from_numpy(dst2)
outs2 = ants.registration(fix_img2, move_img2, type_of_transform = 'SyN')  
reg_img2 = outs2['warpedmovout'] 
arry2 = reg_img2.numpy()

fix_img3 = ants.from_numpy(img3)
move_img3 = ants.from_numpy(dst3)
outs3 = ants.registration(fix_img3, move_img3, type_of_transform = 'SyN')  
reg_img3 = outs3['warpedmovout'] 
arry3 = reg_img3.numpy()

plt.figure(4,figsize=(6, 8))
plt.subplot(2, 2, 1)
plt.title('ANTsResult')
plt.ylim(0, 255)

arry1_overlap = arry1 * overlap_mr12
arry12_overlap = arry2 * overlap_mr12
arry23_overlap = arry2 * overlap_mr23
arry3_overlap = arry3 * overlap_mr23
arry_overlap = (arry1_overlap + arry12_overlap)/2 + (arry23_overlap + arry3_overlap)/2

copy_labels_MR = maskExpand(mr, label)
mask1 = np.zeros_like(copy_labels_MR)
mask1[copy_labels_MR==1] = 1
copy_labels_MR = maskExpand(mr, label)
mask2 = np.zeros_like(copy_labels_MR)
mask2[copy_labels_MR==2] = 1
mask3 = np.zeros_like(copy_labels_MR)
mask3[copy_labels_MR==3] = 1
arry_plus = arry1 * mask1 + arry2 * mask2 + arry3 * mask3
arry_mius = arry_plus * overlap_mr

arry123 = arry_plus - arry_mius + arry_overlap

show_arry = (arry123).copy()
show_arry = np.flipud(show_arry)
show_arry = show_arry
plt.ylim(0, WIDTH-1)
plt.axis('off') 
plt.imshow(show_arry, cmap = 'gray')

"""Deformation field"""
# Displacement in region 1.
affine_trans1 = scio.loadmat(outs1['invtransforms'][0])['AffineTransform_float_2_2'].reshape(2, 3, order = 'F')
add = np.array([0, 0, 1])
affine1 = np.row_stack((affine_trans1, add)) 
temp = affine1[1, 0]
affine1[1,0] = affine1[0, 1]
affine1[0,1] = temp
center1 = scio.loadmat(outs1['invtransforms'][0])['fixed']
affine1 = ant2mat2D(affine1, center1) # mov * affine1 = target
field1 = warp_field(affine1, dst1.shape)

# Displacement in region 2.
affine_trans2 = scio.loadmat(outs2['invtransforms'][0])['AffineTransform_float_2_2'].reshape(2, 3, order = 'F')
add = np.array([0, 0, 1])
affine2 = np.row_stack((affine_trans2, add)) # affine_trans2
temp = affine2[1,0]
affine2[1,0] = affine2[0,1]
affine2[0,1] = temp
center2 = scio.loadmat(outs2['invtransforms'][0])['fixed']
affine2 = ant2mat2D(affine2, center2)
field2 = warp_field(affine2, dst2.shape)

# Displacement in region 3.
affine_trans3 = scio.loadmat(outs3['invtransforms'][0])['AffineTransform_float_2_2'].reshape(2, 3, order = 'F')
add = np.array([0, 0, 1])
affine3 = np.row_stack((affine_trans3, add)) # affine_trans2
temp = affine3[1,0]
affine3[1,0] = affine3[0,1]
affine3[0,1] = temp
center3 = scio.loadmat(outs3['invtransforms'][0])['fixed']
affine3 = ant2mat2D(affine3, center3)
field3 = warp_field(affine3, dst3.shape)

"""mask for displacement fields"""
copy_labels_MR = maskExpand(mr, label);

#plt.figure(5,figsize=(6, 8))
#plt.subplot(4,1,1)
#plt.title('mask')
#visual = np.flipud(copy_labels_MR)
#plt.ylim(0, WIDTH-1)
#cmap = mpl.colors.ListedColormap([RED,PURPLE,BROWN])                                  
#mat = plt.imshow(visual, cmap) 
#plt.axis('off') 
#ax = plt.gca()
#divider = make_axes_locatable(ax)
#cax = divider.append_axes("right", size="5%", pad=0.05)
#unique_data = np.unique(visual);
#plt.colorbar(mat, cax = cax, ticks=np.arange(np.min(unique_data),np.max(unique_data)+1), spacing='uniform')

mask1 = np.zeros_like(copy_labels_MR)
mask1[copy_labels_MR==1] = 1
mask1 = np.stack((mask1, mask1))
mask1 = mask1.swapaxes(1,2)

mask2 = np.zeros_like(copy_labels_MR)
mask2[copy_labels_MR==2] = 1
mask2 = np.stack((mask2, mask2))
mask2 = mask2.swapaxes(1,2)

mask3 = np.zeros_like(copy_labels_MR)
mask3[copy_labels_MR==3] = 1
mask3 = np.stack((mask3, mask3))
mask3 = mask3.swapaxes(1,2)

# field: 
field1_overlap = field1 * np.stack((overlap_mr12, overlap_mr12), axis = 0)
field12_overlap = field2 * np.stack((overlap_mr12, overlap_mr12), axis = 0)
field23_overlap = field2 * np.stack((overlap_mr23, overlap_mr23), axis = 0)
field3_overlap = field3 * np.stack((overlap_mr23, overlap_mr23), axis = 0)
field_overlap = (field1_overlap + field12_overlap)/2 + (field23_overlap + field3_overlap)/2

field1 = field1.transpose(0,2,1) * mask1 
field2 = field2.transpose(0,2,1) * mask2
field3 = field3.transpose(0,2,1) * mask3
field_plus = field1 + field2 + field3
field_mius = field_plus * np.stack((overlap_mr.transpose(1,0), overlap_mr.transpose(1,0)), axis = 0)

field = field_plus - field_mius + field_overlap.transpose(0,2,1)

plt.figure(5,figsize=(6, 8))
plt.subplot(4,1,2)
plt.title('field (X, Y)')
show_fieldx = field[0,:,:].copy()
show_fieldx = np.fliplr(show_fieldx)
show_fieldx = show_fieldx.transpose((1,0))
show_fieldy = field[1,:,:].copy()
show_fieldy = np.fliplr(show_fieldy)
show_fieldy = show_fieldy.transpose((1,0))
plt.ylim(0, WIDTH-1)
mat = plt.imshow(np.hstack((show_fieldx, show_fieldy)), vmin=-50, vmax=20,cmap='viridis') 
plt.axis('off') 
ax = plt.gca()
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(mat, cax = cax, spacing='uniform')

plt.figure(15,figsize=(6, 8))
plt.subplot(4,1,2)
plt.title('field (X, Y)')
show_fieldx = field[0,:,:].copy()
show_fieldx = np.fliplr(show_fieldx)
show_fieldx = show_fieldx.transpose((1,0))
show_fieldy = field[1,:,:].copy()
show_fieldy = np.fliplr(show_fieldy)
show_fieldy = show_fieldy.transpose((1,0))
plt.ylim(0, WIDTH-1)
mat = plt.imshow(np.hstack((show_fieldx, show_fieldy)), vmin=-50, vmax=20,cmap='viridis') 
plt.axis('off') 
ax = plt.gca()
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(mat, cax = cax, spacing='uniform')

"""blending displacement fields"""
#  fieldMaskGaussian
dilationSigma = 6
fieldmaskgaussian, dilation_maskEdges = fieldMaskGaussian(field, copy_labels_MR, dilationSigma)

plt.figure(5,figsize=(6, 8))
plt.subplot(4,1,3)
plt.title('field-Mask-Gaussian (X, Y)')
show_fieldx = fieldmaskgaussian[0,:,:].copy()
show_fieldx = np.fliplr(show_fieldx)
show_fieldx = show_fieldx.transpose((1,0))
show_fieldy = fieldmaskgaussian[1,:,:].copy()
show_fieldy = np.fliplr(show_fieldy)
show_fieldy = show_fieldy.transpose((1,0))
plt.ylim(0, WIDTH-1)
mat = plt.imshow(np.hstack((show_fieldx, show_fieldy)), vmin=-50, vmax=20,cmap='viridis') 
plt.axis('off') 
ax = plt.gca()
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(mat, cax = cax, spacing='uniform')

plt.figure(5, figsize=(6, 8))
plt.subplot(4,1,1)
plt.title('mask (original, Gaussian[WIDTH: %s])'%dilationSigma)
visualOriginal = np.flipud(copy_labels_MR)
visualGaussian = np.flipud(dilation_maskEdges)
plt.ylim(0, WIDTH-1)
cmap = mpl.colors.ListedColormap([RED,PURPLE,BROWN,LIGHT_PURPLE,LIGHT_BROWN])                                  
mat = plt.imshow(np.hstack((visualOriginal, visualGaussian+4)), cmap) 
plt.axis('off') 
ax = plt.gca()
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
#unique_data = np.unique(visual);
unique_data = np.array([1, 2, 3, 4, 5], dtype='int8')
plt.colorbar(mat, cax = cax, ticks=np.arange(np.min(unique_data),np.max(unique_data)+1), spacing='uniform')

#  fieldGaussian
fieldgaussian = fieldGaussian(field)

plt.figure(5,figsize=(6, 8))
plt.subplot(4,1,4)
plt.title('field-Gaussian (X, Y)')
show_fieldx = fieldgaussian[0,:,:].copy()
show_fieldx = np.fliplr(show_fieldx)
show_fieldx = show_fieldx.transpose((1,0))
show_fieldy = fieldgaussian[1,:,:].copy()
show_fieldy = np.fliplr(show_fieldy)
show_fieldy = show_fieldy.transpose((1,0))
plt.ylim(0, WIDTH-1)
mat = plt.imshow(np.hstack((show_fieldx, show_fieldy)), vmin=-50, vmax=20,cmap='viridis') 
plt.axis('off') 
ax = plt.gca()
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(mat, cax = cax, spacing='uniform')

# fieldLaplace
fieldlaplace = fieldLaplace(field)

plt.figure(15,figsize=(6, 8))
plt.subplot(4,1,4)
plt.title('field-Laplace (X, Y)')
show_fieldx = fieldlaplace[0,:,:].copy()
show_fieldx = np.fliplr(show_fieldx)
show_fieldx = show_fieldx.transpose((1,0))
show_fieldy = fieldlaplace[1,:,:].copy()
show_fieldy = np.fliplr(show_fieldy)
show_fieldy = show_fieldy.transpose((1,0))
plt.ylim(0, WIDTH-1)
mat = plt.imshow(np.hstack((show_fieldx, show_fieldy)), vmin=-50, vmax=20,cmap='viridis') 
plt.axis('off') 
ax = plt.gca()
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(mat, cax = cax, spacing='uniform')

#  fieldMaskLaplace
dilationSigma = 6
fieldmasklaplace, dilation_maskEdges = fieldMaskLaplace(field, copy_labels_MR, dilationSigma)

plt.figure(15,figsize=(6, 8))
plt.subplot(4,1,3)
plt.title('field-maskLaplace (X, Y)')
show_fieldx = fieldmasklaplace[0,:,:].copy()
show_fieldx = np.fliplr(show_fieldx)
show_fieldx = show_fieldx.transpose((1,0))
show_fieldy = fieldmasklaplace[1,:,:].copy()
show_fieldy = np.fliplr(show_fieldy)
show_fieldy = show_fieldy.transpose((1,0))
plt.ylim(0, WIDTH-1)
mat = plt.imshow(np.hstack((show_fieldx, show_fieldy)), vmin=-50, vmax=20,cmap='viridis') 
plt.axis('off') 
ax = plt.gca()
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(mat, cax = cax, spacing='uniform')

plt.figure(15, figsize=(6, 8))
plt.subplot(4,1,1)
plt.title('mask (original, Laplace[WIDTH: %s])'%dilationSigma)
visualOriginal = np.flipud(copy_labels_MR)
visualGaussian = np.flipud(dilation_maskEdges)
plt.ylim(0, WIDTH-1)
cmap = mpl.colors.ListedColormap([RED,PURPLE,BROWN,LIGHT_PURPLE,LIGHT_BROWN])                                  
mat = plt.imshow(np.hstack((visualOriginal, visualGaussian+4)), cmap) 
plt.axis('off') 
ax = plt.gca()
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
#unique_data = np.unique(visual);
unique_data = np.array([1, 2, 3, 4, 5], dtype='int8')
plt.colorbar(mat, cax = cax, ticks=np.arange(np.min(unique_data),np.max(unique_data)+1), spacing='uniform')

"""warp(backward)"""
arryresult = warp_fieldtoimage(np.fliplr(dst), field, [WIDTH,LENGTH]) # ANTs field
arryresult_maskGaussian = warp_fieldtoimage(dst, fieldmaskgaussian, [WIDTH,LENGTH]) # ANTs mask-Gussian-field
arryresult_Gaussian = warp_fieldtoimage(dst, fieldgaussian, [WIDTH,LENGTH]) # ANTs nonmask-Gussian-field
arryresult_Laplace = warp_fieldtoimage(dst, fieldlaplace, [WIDTH,LENGTH]) # ANTs field-Laplace
arryresult_maskLaplace = warp_fieldtoimage(dst, fieldmasklaplace, [WIDTH,LENGTH]) # ANTs field-Laplace

plt.figure(4, figsize=(7, 6))
plt.subplot(2, 2, 2)
plt.title('ourResult (overlap-average)')
arryresultarryresult = np.fliplr(arryresult)
arryresultarryresult = arryresultarryresult.transpose((1,0))
plt.ylim(0, WIDTH-1)
plt.axis('off')
plt.imshow(arryresultarryresult, cmap='gray')

plt.figure(4, figsize=(7, 6))
plt.subplot(2, 2, 3)
plt.title('ourResult (mask-Gussian-field)')
arryresultarryresult = np.fliplr(arryresult_maskGaussian)
arryresultarryresult = arryresultarryresult.transpose((1,0))
plt.ylim(0, WIDTH-1)
plt.axis('off')
plt.imshow(arryresultarryresult, cmap='gray')

plt.figure(4, figsize=(7, 6))
plt.subplot(2, 2, 4)
plt.title('ourResult (nonmask-Gussian-field)')
arryresultarryresult = np.fliplr(arryresult_Gaussian)
arryresultarryresult = arryresultarryresult.transpose((1,0))
plt.ylim(0, WIDTH-1)
plt.axis('off')
plt.imshow(arryresultarryresult, cmap='gray')

plt.figure(16)
plt.title('ourResult (nonmask-Laplace-field)')
arryresultarryresult = np.fliplr(arryresult_Laplace)
arryresultarryresult = arryresultarryresult.transpose((1,0))
plt.ylim(0, WIDTH-1)
plt.axis('off')
plt.imshow(arryresultarryresult, cmap='gray')

plt.figure(17)
plt.title('ourResult (mask-Laplace-field)')
arryresultarryresult = np.fliplr(arryresult_maskLaplace)
arryresultarryresult = arryresultarryresult.transpose((1,0))
plt.ylim(0, WIDTH-1)
plt.axis('off')
plt.imshow(arryresultarryresult, cmap='gray')

#
## ants.apply_transforms
#fix_img1 = ants.from_numpy(img1)
#move_img1 = ants.from_numpy(dst1)
#mywarpedimage1 = ants.apply_transforms(fixed=fix_img1, moving=move_img1,
#                                               transformlist=outs1['fwdtransforms'])
#fix_img2 = ants.from_numpy(img2)
#move_img2 = ants.from_numpy(dst2)
#mywarpedimage2 = ants.apply_transforms(fixed=fix_img2, moving=move_img2,
#                                               transformlist=outs2['fwdtransforms'])
#fix_img3 = ants.from_numpy(img3)
#move_img3 = ants.from_numpy(dst3)
#mywarpedimage3 = ants.apply_transforms(fixed=fix_img3, moving=move_img3,
#                                               transformlist=outs3['fwdtransforms'])
#arr1 = mywarpedimage1.numpy()
#arr2 = mywarpedimage2.numpy()
#arr3 = mywarpedimage3.numpy()
#
#plt.figure(4,figsize=(6, 8))
#plt.subplot(4, 1, 3)
#plt.title('ANTsResult (transform & warp)')
#arr1 = np.fliplr(arr1)
#arrr1 = arr1.transpose((1,0))
#arr2 = np.fliplr(arr2)
#arrr2 = arr2.transpose((1,0))
#arr3 = np.fliplr(arr3)
#arrr3 = arr3.transpose((1,0))
#plt.ylim(0, WIDTH-1)
#
#arrr1_overlap = arrr1 * overlap_mr
#arrr2_overlap = arrr2 * overlap_mr
#arrr3_overlap = arrr3 * overlap_mr
#arrr_overlap = (arrr1_overlap + arrr2_overlap)/2 + (arrr2_overlap + arrr3_overlap)/2
#
#copy_labels_MR = maskExpand(mr, label)
#mask1 = np.zeros_like(copy_labels_MR)
#mask1[copy_labels_MR==1] = 1
#copy_labels_MR = maskExpand(mr, label)
#mask2 = np.zeros_like(copy_labels_MR)
#mask2[copy_labels_MR==2] = 1
#mask3 = np.zeros_like(copy_labels_MR)
#mask3[copy_labels_MR==3] = 1
#arrr_plus = arrr1 * mask1 + arrr2 * mask2 + arrr3 * mask3
#arrr_mius = arrr_plus * overlap_mr
#
#arrr123 = arrr_plus - arrr_mius + arrr_overlap
#
#plt.axis('off')
#plt.imshow(arrr123, cmap="gray")

"""visiualization"""
import sys
sys.path.insert(0, 'C:/Users/fangyan/Desktop/PiecewiseRegistration')
import gui

outs = ants.registration(ants.from_numpy(fix), ants.from_numpy(dst), type_of_transform = 'Affine')  
reg_img = outs['warpedmovout'] 
arry = reg_img.numpy()

fix_sitk = sitk.GetImageFromArray(fix.transpose((1,0))[np.newaxis,:,:])
dst_sitk = sitk.GetImageFromArray(dst.transpose((1,0))[np.newaxis,:,:])
ants_wholeresult_sitk = sitk.GetImageFromArray(arry.transpose((1,0))[np.newaxis,:,:])
ants_result_sitk = sitk.GetImageFromArray(arry123.transpose((1,0))[np.newaxis,:,:])
result_sitk = sitk.GetImageFromArray(arryresult_maskLaplace.transpose((1,0))[np.newaxis,:,:])

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



