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

from parameter import get_parser

from utilities import dataProperty 
from utilities import norm
from utilities import norm255
from utilities import flip180
from utilities import splitImage

from basicProcessing import ant2mat2D
from basicProcessing import ant2mat3D
from basicProcessing import landmarkGenerate
from basicProcessing import landmarkGenerate3D
from basicProcessing import denoiseLabels
from basicProcessing import denoiseLabels3D
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
from visiualization import figureshow
from visiualization import fieldshow
from visiualization import maskshow
from visiualization import resultshow

#import re
#for x in dir():
#    if not re.match('^__',x) and x!="re":
#        exec(" ".join(("del",x)))

#plt.subplots_adjust(top=0.97,bottom=0.03,left=0.03,right=1,hspace=0.1,wspace=0)
#plt.close()
"""
segmentation module
"""
parser = get_parser()                        
args = parser.parse_args()
WHITE = args.WHITE
GREY = args.GREY
RED = args.RED
PURPLE = args.PURPLE
BROWN = args.BROWN
LIGHT_BROWN = args.LIGHT_BROWN
LIGHT_PURPLE = args.LIGHT_PURPLE
label = args.label
SEGMENTS = args.SEGMENTS
SUBJECT = args.SUBJECT
SLIDEAXIAL = args.SLIDEAXIAL
SLIDE = args.SLIDE
waxholmLabeltoMRPath = args.waxholmLabeltoMRPath
waxholmLabeltoLSPath = args.waxholmLabeltoLSPath
mrpath = args.mrpath
lspath = args.lspath
regisMethod = args.regisMethod

# dataProperty: path,sitk,arry,origin,spacing,direction,size,dtype
# object: waxholmLabelMR,waxholmLabelLS,MR,LS 
waxholmLabelMR = dataProperty(waxholmLabeltoMRPath)
waxholmLabelLS = dataProperty(waxholmLabeltoLSPath)
MR = dataProperty(mrpath)
LS = dataProperty(lspath)

# note!
#waxholmLabelMR.bake()
#waxholmLabelLS.bake()
#MR.bake()
#LS.bake()

#arry_LS = ants.n4_bias_field_correction(ants.from_numpy(lsarry)).numpy()
# remove carrier in LS
if SUBJECT == 'intra':
    LS.arry[LS.arry<30] = 0
    LS.arry[:,:,180:256] = 0
MR.arry = norm(MR.arry)
LS.arry = norm(LS.arry)

niisave(MR.arry,MR.origin,MR.spacing,MR.direction, path = '../result/input/mrisitk3D.nii.gz')
niisave(LS.arry,LS.origin,LS.spacing,LS.direction, path = '../result/input/lssitk3D.nii.gz')
niisave(waxholmLabelMR.arry,waxholmLabelMR.origin,waxholmLabelMR.spacing,waxholmLabelMR.direction, path = '../result/input/waxLabeltoMRsitk3D.nii.gz')
niisave(waxholmLabelLS.arry,waxholmLabelLS.origin,waxholmLabelLS.spacing,waxholmLabelLS.direction, path = '../result/input/waxLabeltoLSsitk3D.nii.gz')

plt.figure(1,figsize=(6, 8))
plt.subplot(3,2,1)
plt.title('Image (MR, Light-sheet)')
figureshow(MR.arry,LS.arry,SLIDE,label=False,cmap='gray',plane='sagittal')

plt.figure(10,figsize=(6, 8))
plt.subplot(3,2,1)
plt.title('Image (MR, Light-sheet)')
figureshow(MR.arry,LS.arry,SLIDEAXIAL,label=False,cmap='gray',plane='axial')

# label
landmarks_MR,landmarks_LS = waxLabelProcess(waxholmLabelMR.arry,waxholmLabelLS.arry,\
                                             waxholmLabelMR.origin,waxholmLabelMR.spacing,waxholmLabelMR.direction)
# check coordinates: remove incorrect landmarks
landmarks_MR[MR.arry==0] = 5
landmarks_LS[LS.arry<0.099] = 5

plt.figure(1,figsize=(6, 8))
plt.subplot(3,2,2)
plt.title('Waxholm Labels(MR, Light-sheet)')
cmap = mpl.colors.ListedColormap([WHITE,RED,LIGHT_BROWN,PURPLE,BROWN,GREY])
figureshow(landmarks_MR,landmarks_LS,SLIDE,True,cmap,'sagittal')

plt.figure(10,figsize=(6, 8))
plt.subplot(3,2,2)
plt.title('Waxholm Labels(MR, Light-sheet)')
cmap = mpl.colors.ListedColormap([WHITE,RED,LIGHT_BROWN,PURPLE,BROWN,GREY])
figureshow(landmarks_MR,landmarks_LS,SLIDEAXIAL,True,cmap,'axial')

'''Post processing  [first time]'''
# add: post-processing after waxLabelProcess
# mr0,maxContour_mr,ls0,maxContour_ls,dilation_mr1,dilation_mr2,dilation_mr3,dilation_mr4,dilation_ls1,dilation_ls2,dilation_ls3,dilation_ls4 = postProcessing(landmarks_MR, landmarks_LS, ifmorphology='True')
mr0,maxContour_mr,ls0,maxContour_ls,\
dilation_mr1,dilation_mr2,dilation_mr3,dilation_mr4,\
dilation_ls1,dilation_ls2,dilation_ls3,dilation_ls4 \
= postProcessing3D(landmarks_MR, landmarks_LS, ifmorphology='True')

niisave(mr0,MR.origin,MR.spacing,MR.direction,path = '../result/input/MRFirstPostProcessing.nii.gz')
niisave(ls0,LS.origin,LS.spacing,LS.direction,path = '../result/input/LSFirstPostProcessing.nii.gz')

# compare labels of MR and LS before/after post-processing
for i in range(1,len(label)+1):
    mr0[(landmarks_MR-mr0)==-i] = 0 # -5,-4,-3,-2,-1
    ls0[(landmarks_LS-ls0)==-i] = 0 # -5,-4,-3,-2,-1

plt.figure(1,figsize=(6, 8))
plt.subplot(3,2,3)
plt.title('First PostProcessing(MR, Light-sheet)')
#cmap = mpl.colors.ListedColormap([RED,PURPLE,BROWN,GREY,LIGHT_BROWN,LIGHT_PURPLE])
cmap = mpl.colors.ListedColormap([WHITE,RED,LIGHT_BROWN,PURPLE,BROWN,GREY])
figureshow(mr0,ls0,SLIDE,True,cmap,'sagittal')

plt.figure(10,figsize=(6, 8))
plt.subplot(3,2,3)
plt.title('First PostProcessing(MR, Light-sheet)')
#cmap = mpl.colors.ListedColormap([RED,PURPLE,BROWN,GREY,LIGHT_BROWN,LIGHT_PURPLE])
cmap = mpl.colors.ListedColormap([WHITE,RED,LIGHT_BROWN,PURPLE,BROWN,GREY])
figureshow(mr0,ls0,SLIDEAXIAL,True,cmap,'axial')

'''random walk [first time]'''
mr0[mr0==5] = -1 # MR background pixels 
ls0[ls0==5] = -1 # LS background pixels 
labels_MR1 = random_walker(MR.arry,mr0,beta=130,mode='cg_mg',multichannel=False)
labels_LS1 = random_walker(LS.arry,ls0,beta=130,mode='cg_mg',multichannel=False)
labels_MR1 = labels_MR1.astype(np.int8)
labels_LS1 = labels_LS1.astype(np.int8)
labels_MR1[labels_MR1==-1] = 5
labels_LS1[labels_LS1==-1] = 5

niisave(labels_MR1,MR.origin,MR.spacing,MR.direction,path = '../result/input/MRFirstRandomWalk.nii.gz')
niisave(labels_LS1,LS.origin,LS.spacing,LS.direction,path = '../result/input/LSFirstRandomWalk.nii.gz')

plt.figure(1,figsize=(6, 8))
plt.subplot(3,2,4)
plt.title('RW(MR, Light-sheet)')
cmap = mpl.colors.ListedColormap([RED,LIGHT_BROWN,PURPLE,BROWN,GREY])
figureshow(labels_MR1,labels_LS1,SLIDE,True,cmap,'sagittal')

plt.figure(10,figsize=(6, 8))
plt.subplot(3,2,4)
plt.title('RW(MR, Light-sheet)')
cmap = mpl.colors.ListedColormap([RED,LIGHT_BROWN,PURPLE,BROWN,GREY])
figureshow(labels_MR1,labels_LS1,SLIDEAXIAL,True,cmap,'axial')

"""PostProcessing [second time]"""
#mr1,maxContour_mr,ls1,maxContour_ls,dilation_mr1,dilation_mr2,dilation_mr3,dilation_mr4,dilation_ls1,dilation_ls2,dilation_ls3,dilation_ls4 = postProcessing(labels_MR1, labels_LS1, ifmorphology='True')
mr1,maxContour_mr,ls1,maxContour_ls,\
dilation_mr1,dilation_mr2,dilation_mr3,dilation_mr4,\
dilation_ls1,dilation_ls2,dilation_ls3,dilation_ls4 \
= postProcessing3D(labels_MR1, labels_LS1, ifmorphology='False')

niisave(mr1,MR.origin,MR.spacing,MR.direction,path = '../result/input/MRSecondPostProcessing.nii.gz')
niisave(ls1,LS.origin,LS.spacing,LS.direction,path = '../result/input/LSSecondPostProcessing.nii.gz')

# compare labels of MR and LS before/after post-processing
for i in range(1,len(label)+1):
    mr1[(labels_MR1-mr1)==-i] = 0 # -5,-4,-3,-2,-1
    ls1[(labels_LS1-ls1)==-i] = 0 # -5,-4,-3,-2,-1

plt.figure(1,figsize=(6, 8))
plt.subplot(3,2,5)
plt.title('Second PostProcessing(MR, Light-sheet)')
cmap = mpl.colors.ListedColormap([RED,LIGHT_BROWN,PURPLE,BROWN,GREY])
figureshow(mr1,ls1,SLIDE,True,cmap,'sagittal')

plt.figure(10,figsize=(6, 8))
plt.subplot(3,2,5)
plt.title('Second PostProcessing(MR, Light-sheet)')
#cmap = mpl.colors.ListedColormap([RED,PURPLE,BROWN,GREY,LIGHT_BROWN,LIGHT_PURPLE])
#cmap = mpl.colors.ListedColormap([WHITE,RED,LIGHT_BROWN,PURPLE,BROWN,GREY])
cmap = mpl.colors.ListedColormap([RED,LIGHT_BROWN,PURPLE,BROWN,GREY])
figureshow(mr1,ls1,SLIDEAXIAL,True,cmap,'axial')

'''random walk [second time]'''
labels_MR2 = random_walker(MR.arry,mr1,beta=130,mode='cg_mg',multichannel=False)
labels_LS2 = random_walker(LS.arry,ls1,beta=130,mode='cg_mg',multichannel=False)
labels_MR2 = labels_MR2.astype(np.int8)
labels_LS2 = labels_LS2.astype(np.int8)

niisave(labels_MR2,MR.origin,MR.spacing,MR.direction,path = '../result/input/MRSecondRandomWalk.nii.gz')
niisave(labels_LS2,LS.origin,LS.spacing,LS.direction,path = '../result/input/LSSecondRandomWalk.nii.gz')

plt.figure(1,figsize=(6, 8))
plt.subplot(3,2,6)
plt.title('RW(MR, Light-sheet)')
cmap = mpl.colors.ListedColormap([RED,LIGHT_BROWN,PURPLE,BROWN,GREY])
figureshow(labels_MR2,labels_LS2,SLIDE,True,cmap,'sagittal')

plt.figure(10,figsize=(6, 8))
plt.subplot(3,2,6)
plt.title('RW(MR, Light-sheet)')
cmap = mpl.colors.ListedColormap([RED,LIGHT_BROWN,PURPLE,BROWN,GREY])
figureshow(labels_MR2,labels_LS2,SLIDEAXIAL,True,cmap,'axial')

mr = labels_MR2.copy()
mrlist = [labels_MR2.copy(),labels_MR2.copy(),labels_MR2.copy(),labels_MR2.copy()]
for i in range(0,SEGMENTS):
    mrlist[i][mrlist[i]!=(i+1)] = 0

ls = labels_LS2.copy()
lslist = [labels_LS2.copy(),labels_LS2.copy(),labels_LS2.copy(),labels_LS2.copy()]
for i in range(0,SEGMENTS):
    lslist[i][lslist[i]!=(i+1)] = 0

## segmentation overlay
#plt.figure(1,figsize=(6, 8))
#plt.subplot(3,2,6)
#plt.title('Contour Overlay(MR, Light-sheet)')
#mrshow = arry_MR[SLIDE,:,:].copy()
#mrshow = mrshow.transpose(1,0)
#lsshow = arry_LS[SLIDE,:,:].copy()
#lsshow = lsshow.transpose(1,0)
#mrContourshow = maxContour_mr[SLIDE,:,:].copy()
#mrContourshow = mrContourshow.transpose(1,0)
#lsContourshow = maxContour_ls[SLIDE,:,:].copy()
#lsContourshow = lsContourshow.transpose(1,0)
#mat = plt.imshow(np.hstack((mrshow*1.3+mrContourshow, lsshow*1.3+lsContourshow)),cmap = 'gray') 
#plt.axis('off') 
#
#plt.figure(10,figsize=(6, 8))
#plt.subplot(3,2,6)
#plt.title('Contour Overlay(MR, Light-sheet)')
#mrshow = arry_MR[:,:,SLIDE].copy()
#mrshow = mrshow.transpose(1,0)
#lsshow = arry_LS[:,:,SLIDE].copy()
#lsshow = lsshow.transpose(1,0)
#mrContourshow = maxContour_mr[:,:,SLIDE].copy()
#mrContourshow = mrContourshow.transpose(1,0)
#lsContourshow = maxContour_ls[:,:,SLIDE].copy()
#lsContourshow = lsContourshow.transpose(1,0)
#mat = plt.imshow(np.hstack((mrshow*1.3+mrContourshow, lsshow*1.3+lsContourshow)),cmap = 'gray') 
#plt.axis('off') 

# boundary detection for 3D of seg1,seg2,seg3,seg4
#for i in range(0, dilation_mr1.shape[0]):
#    subMaskmr1[i,:,:], maxContourmr1[i,:,:] = denoiseLabels(norm(mr1[i,:,:]), ifmorphology='False')
#    subMaskmr2[i,:,:], maxContourmr2[i,:,:] = denoiseLabels(norm(mr2[i,:,:]), ifmorphology='False')
#    subMaskmr3[i,:,:], maxContourmr3[i,:,:] = denoiseLabels(norm(mr3[i,:,:]), ifmorphology='False')
#    subMaskmr4[i,:,:], maxContourmr4[i,:,:] = denoiseLabels(norm(mr4[i,:,:]), ifmorphology='False')
#for i in range(0, dilation_ls1.shape[0]):
#    subMaskls1[i,:,:], maxContourls1[i,:,:] = denoiseLabels(norm(ls1[i,:,:]), ifmorphology='False')
#    subMaskls2[i,:,:], maxContourls2[i,:,:] = denoiseLabels(norm(ls2[i,:,:]), ifmorphology='False')
#    subMaskls3[i,:,:], maxContourls3[i,:,:] = denoiseLabels(norm(ls3[i,:,:]), ifmorphology='False')
#    subMaskls4[i,:,:], maxContourls4[i,:,:] = denoiseLabels(norm(ls4[i,:,:]), ifmorphology='False')
subMaskmrlist = [[],[],[],[]]
maxContourmrlist = [[],[],[],[]]
maxContourmr = np.zeros_like(mrlist[0])
for i in range(0,SEGMENTS):
    subMaskmrlist[i], maxContourmrlist[i] = denoiseLabels3D(norm(mrlist[i]),ifmorphology='False')
    maxContourmrlist[i][maxContourmrlist[i]==1] = i+1
    maxContourmr = maxContourmr + maxContourmrlist[i]
mrMask = np.zeros_like(mr) + 1
mrMask = mrMask - subMaskmrlist[0] - subMaskmrlist[1] - subMaskmrlist[2] - subMaskmrlist[3]
mrMask[mrMask==-1] = 0 # overlap = 0

subMasklslist = [[],[],[],[]]
maxContourlslist = [[],[],[],[]]
maxContourls = np.zeros_like(lslist[0])
for i in range(0,SEGMENTS):
    subMasklslist[i], maxContourlslist[i] = denoiseLabels3D(norm(lslist[i]),ifmorphology='False')
    maxContourlslist[i][maxContourlslist[i]==1] = i+1
    maxContourls = maxContourls + maxContourlslist[i]
lsMask = np.zeros_like(ls) + 1
lsMask = lsMask - subMasklslist[0] - subMasklslist[1] - subMasklslist[2] - subMasklslist[3]
lsMask[lsMask==-1] = 0 # overlap = 0

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
subimgmr_255 = sitk.GetImageFromArray(np.swapaxes(norm255(MR.arry),0,2))
subimgmr_255 = sitk.Cast(subimgmr_255,sitk.sitkUInt8)
sublabelmr = sitk.GetImageFromArray(np.swapaxes(mr,0,2))
sublabelmr = sitk.Cast(sublabelmr,sitk.sitkUInt8)
subSitkmr = sitk.LabelOverlay(subimgmr_255, sublabelmr)
subSitkmr.SetSpacing(MR.spacing)
#plane = ['axial', 'sagittal', 'coronal']
plane = ['sagittal']
for i in range(0, len(plane)):
    myshow(subSitkmr,plane[i] ,'./regi_waxholmLabel to FAMRI',SLIDE,title='MR', dpi=80)
    
subimgls_255 = sitk.GetImageFromArray(np.swapaxes(norm255(LS.arry),0,2))
subimgls_255 = sitk.Cast(subimgls_255,sitk.sitkUInt8)
sublabells = sitk.GetImageFromArray(np.swapaxes(ls,0,2))
sublabells = sitk.Cast(sublabells,sitk.sitkUInt8)
subSitkls = sitk.LabelOverlay(subimgls_255, sublabells)
subSitkls.SetSpacing(LS.spacing)
for i in range(0, len(plane)):
    myshow(subSitkls,plane[i] ,'./regi_waxholmLabel to FAMRI',SLIDE,title='LS', dpi=80)

# Split image
subImg_mr = [[],[],[],[]]    
for i in range(0,SEGMENTS):
    savePath_mr = '../result/subImg_mr%d3D.nii.gz'%(i+1)
    subImg_mr[i] = splitImage(MR.arry,mrlist[i],savePath_mr,MR.spacing,MR.origin,MR.direction)
subImg_ls = [[],[],[],[]]    
for i in range(0,SEGMENTS):
    savePath_ls = '../result/subImg_ls%d3D.nii.gz'%(i+1)
    subImg_ls[i] = splitImage(LS.arry,lslist[i],savePath_ls,LS.spacing,LS.origin,LS.direction)

# label overlapping areas
overlap_mr12 = mrlist[1] - mrlist[0]
overlap_mr12[overlap_mr12!=1] = 0
overlap_mr23 = mrlist[2] - mrlist[1]
overlap_mr23[overlap_mr23!=1] = 0
overlap_mr13 = mrlist[2] - mrlist[0]
overlap_mr13[overlap_mr13!=2] = 0
overlap_mr34 = mrlist[3] - mrlist[2]
overlap_mr34[overlap_mr34!=1] = 0
overlap_mr = overlap_mr12 + overlap_mr23 + overlap_mr13 + overlap_mr34
overlap_ls12 = lslist[1] - lslist[0]
overlap_ls12[overlap_ls12!=1] = 0
overlap_ls23 = lslist[2] - lslist[1]
overlap_ls23[overlap_ls23!=1] = 0
overlap_ls13 = lslist[2] - lslist[0]
overlap_ls13[overlap_ls13!=2] = 0
overlap_ls34 = lslist[3] - lslist[2]
overlap_ls34[overlap_ls34!=1] = 0
overlap_ls = overlap_ls12 + overlap_ls23 + overlap_ls13 + overlap_ls34

"""
registration module
"""
THICK, LENGTH, WIDTH = MR.size
tem_THICK, tem_WIDTH, tem_LENGTH = MR.size

# convert
fix = MR.arry
dst = LS.arry
fix_img = ants.from_numpy(MR.arry,list(MR.origin),list(MR.spacing))
move_img = ants.from_numpy(LS.arry,list(LS.origin),list(LS.spacing))

# remove background
#dst[dst<30]=0  
#dst[:,180:256]=0
mrcopy = mr.copy()
lscopy = ls.copy()
mrcopy[mrcopy!=5] = 1 # brain
mrcopy[mrcopy==5] = 0 # background
fix = fix * mrcopy
lscopy[lscopy!=5] = 1 # brain
lscopy[lscopy==5] = 0 # background
dst = dst * lscopy

# convert
fix_imglist = [[],[],[],[]]
imglist = [[],[],[],[]]
for i in range(0,SEGMENTS):
    imglist[i] = subImg_mr[i]
    fix_imglist[i] = ants.from_numpy(subImg_mr[i],list(MR.origin),list(MR.spacing))
move_imglist = [[],[],[],[]]
dstlist = [[],[],[],[]]
for i in range(0,SEGMENTS):
    dstlist[i] = subImg_ls[i]
    move_imglist[i] = ants.from_numpy(subImg_ls[i],list(LS.origin),list(LS.spacing))

plt.figure(3,figsize=(6, 8))
plt.subplot(5,1,1)
plt.title('Image(MR, Light-sheet)')
figureshow(fix,dst,SLIDE,label=False,cmap='gray',plane='sagittal')
for i in range(0,SEGMENTS):
    plt.subplot(5,1,i+2)
    plt.title('subImg%d(MR, Light-sheet)'%(i+1))
    figureshow(imglist[i],dstlist[i],SLIDE,label=False,cmap='gray',plane='sagittal')

plt.figure(30,figsize=(6,8))
plt.subplot(5,1,1)
plt.title('Image(MR, Light-sheet)')
figureshow(fix,dst,SLIDEAXIAL,label=False,cmap='gray',plane='axial')
for i in range(0,SEGMENTS):
    plt.subplot(5,1,i+2)
    plt.title('subImg%d(MR, Light-sheet)'%(i+1))
    figureshow(imglist[i],dstlist[i],SLIDEAXIAL,label=False,cmap='gray',plane='axial')

"""affine registration"""
outslist = [[],[],[],[]]
reg_imglist = [[],[],[],[]]
arrylist = [[],[],[],[]]
for i in range(0,SEGMENTS):
    outslist[i] = ants.registration(fix_imglist[i],move_imglist[i],type_of_transform =regisMethod)  
    reg_imglist[i] = outslist[i]['warpedmovout'] 
    arrylist[i] = reg_imglist[i].numpy()

# compute the average of  overlapping regions
plt.figure(4,figsize=(7,6))
plt.subplot(2,3,1)
plt.title('ANTsResult')
arry12_overlap = (arrylist[0] * overlap_mr12) + (arrylist[1] * overlap_mr12)
arry23_overlap = (arrylist[1] * overlap_mr23) + (arrylist[2] * overlap_mr23)
arry13_overlap = (arrylist[0] * overlap_mr13) + (arrylist[2] * overlap_mr13)
arry34_overlap = (arrylist[2] * overlap_mr34) + (arrylist[3] * overlap_mr34)
arry_overlap = arry12_overlap/2 + arry23_overlap/2 + arry13_overlap/2 + arry34_overlap/2

mask = np.zeros_like(mr)
mask,distanceMap = maskExpand3D(mr)
masklist = [[],[],[],[]]
for i in range(0,SEGMENTS):
    masklist[i] = np.zeros_like(mr)
    masklist[i][mask!=(i+1)] = 0
    masklist[i][mask==(i+1)] = 1
arry_plus = np.zeros_like(mr)
for i in range(0,SEGMENTS):
    arry_plus = arry_plus + norm(arrylist[i]) * masklist[i]
arry_mius = arry_plus * overlap_mr
arry1234 = arry_plus - arry_mius + arry_overlap

#show_arry = ((arry1+arry2+arry3+arry4)[SLIDE,:,:]).copy()
show_arry = ((arry1234)[SLIDE,:,:]).copy()
show_arry = np.fliplr(show_arry)
show_arry = show_arry.transpose((1,0))
plt.ylim(0, WIDTH-1)
plt.axis('off') 
plt.imshow(show_arry,cmap ='gray')

"""Deformation field"""
add = np.array([0,0,0,1])
affine_translist = [[],[],[],[]]
affinelist = [[],[],[],[]]
centerlist = [[],[],[],[]]
fieldlist = [[],[],[],[]]
for i in range(0,SEGMENTS):
    affine_translist[i] = scio.loadmat(outslist[i]['invtransforms'][0])['AffineTransform_float_3_3'][0:9].reshape(3,3,order='F').T
    affine_translist[i] = np.column_stack((affine_translist[i],scio.loadmat(outslist[i]['invtransforms'][0])['AffineTransform_float_3_3'][9:12]))
    affinelist[i] = np.row_stack((affine_translist[i],add)) 
    centerlist[i] = scio.loadmat(outslist[i]['invtransforms'][0])['fixed']
    affinelist[i] = ant2mat3D(affinelist[i],centerlist[i],MR.origin,MR.spacing,MR.direction) # mov * affine[i] = target
    fieldlist[i] = warp_field3D(affinelist[i],MR.size,MR.origin,MR.spacing,MR.direction)

## Displacement in region 1.
#field1 = np.swapaxes(sitk.GetArrayFromImage(sitk.ReadImage(outs1['invtransforms'][1])),0,2)
## Displacement in region 2.
#field2 = np.swapaxes(sitk.GetArrayFromImage(sitk.ReadImage(outs2['invtransforms'][1])),0,2)
## Displacement in region 3.
#field3 = np.swapaxes(sitk.GetArrayFromImage(sitk.ReadImage(outs3['invtransforms'][1])),0,2)
## Displacement in region 4.
#field4 = np.swapaxes(sitk.GetArrayFromImage(sitk.ReadImage(outs4['invtransforms'][1])),0,2)

"""
Fields blending module
"""
# field: 
fieldXYZ = np.zeros_like(fieldlist[0]) # thick * 378 * 256 * 3 [z,x,y]
for i in range(0,3):
    field12_overlap = (fieldlist[0][:,:,:,i] * overlap_mr12) + (fieldlist[1][:,:,:,i] * overlap_mr12)
    field23_overlap = (fieldlist[1][:,:,:,i] * overlap_mr23) + (fieldlist[2][:,:,:,i] * overlap_mr23)
    field13_overlap = (fieldlist[0][:,:,:,i] * overlap_mr13) + (fieldlist[2][:,:,:,i] * overlap_mr13)
    field34_overlap = (fieldlist[2][:,:,:,i] * overlap_mr34) + (fieldlist[3][:,:,:,i] * overlap_mr34)
    field_overlap = field12_overlap/2 + field23_overlap/2 + field13_overlap/2 + field34_overlap/2
    
    field_plus = np.zeros_like(MR.arry)
    for j in range(0,SEGMENTS):
        fieldlist[j][:,:,:,i] = fieldlist[j][:,:,:,i] * masklist[j]
        field_plus = field_plus + fieldlist[j][:,:,:,i]
    field_mius = field_plus * overlap_mr
    fieldXYZ[:,:,:,i] = field_plus - field_mius + field_overlap

plt.figure(5,figsize=(6,8))
plt.subplot(4,1,2)
plt.title('field (Z, X, Y)')
cmap = 'viridis'
fieldshow(fieldXYZ,SLIDE,cmap,'sagittal',-50,20,WIDTH)

# just for visiualization
plt.figure(15,figsize=(6,8))
plt.subplot(4,1,2)
plt.title('field (Z, X, Y)')
cmap = 'viridis'
fieldshow(fieldXYZ,SLIDE,cmap,'sagittal',-50,20,WIDTH)

"""mask for deformation fields"""
#  fieldMaskGaussian
dilationSigma = 4
fieldmaskgaussian, dilation_maskEdges = fieldMaskGaussian3D(fieldXYZ,mask,dilationSigma)

plt.figure(5,figsize=(6,8))
plt.subplot(4,1,3)
plt.title('field-Mask-Gaussian (Z, X, Y)')
cmap = 'viridis'
fieldshow(fieldmaskgaussian,SLIDE,cmap,'sagittal',-50,20,WIDTH)

# mask
plt.figure(5,figsize=(6,8))
plt.subplot(4,1,1)
plt.title('mask (original, Gaussian[WIDTH: %s])'%dilationSigma)
cmap = mpl.colors.ListedColormap([RED,PURPLE,BROWN,LIGHT_PURPLE,LIGHT_BROWN])  
maskshow(mask,dilation_maskEdges,SLIDE,cmap,'sagittal',WIDTH)

#  fieldGaussian
fieldgaussian = fieldGaussian3D(fieldXYZ)

plt.figure(5,figsize=(6, 8))
plt.subplot(4,1,4)
plt.title('field-Gaussian (Z, X, Y)')
cmap = 'viridis'
fieldshow(fieldgaussian,SLIDE,cmap,'sagittal',-50,20,WIDTH)

# fieldLaplace
fieldlaplace = fieldLaplace3D(fieldXYZ)

plt.figure(15,figsize=(6, 8))
plt.subplot(4,1,4)
plt.title('field-Laplace (Z, X, Y)')
cmap = 'viridis'
fieldshow(fieldlaplace,SLIDE,cmap,'sagittal',-50,20,WIDTH)

#  fieldMaskLaplace
dilationSigma = 4
fieldmasklaplace, dilation_maskEdges = fieldMaskLaplace3D(fieldXYZ, mask, dilationSigma)

plt.figure(15,figsize=(6, 8))
plt.subplot(4,1,3)
plt.title('field-mask-Laplace (Z, X, Y)')
cmap = 'viridis'
fieldshow(fieldmasklaplace,SLIDE,cmap,'sagittal',-50,20,WIDTH)

plt.figure(15, figsize=(6, 8))
plt.subplot(4,1,1)
plt.title('mask (original, Laplace[WIDTH: %s])'%dilationSigma)
cmap = mpl.colors.ListedColormap([RED,PURPLE,BROWN,LIGHT_PURPLE,LIGHT_BROWN])      
maskshow(mask,dilation_maskEdges,SLIDE,cmap,'sagittal')

"""
Warp module[backward]
"""
arryresult = warp_fieldtoimage3D(dst,fieldXYZ,[tem_THICK,tem_WIDTH,tem_LENGTH],MR.origin,MR.spacing,MR.direction) # ANTs field
arryresult_maskGaussian = warp_fieldtoimage3D(dst,fieldmaskgaussian,[tem_THICK, tem_WIDTH,tem_LENGTH],MR.origin,MR.spacing,MR.direction) # ANTs mask-Gussian-field
arryresult_Gaussian = warp_fieldtoimage3D(dst,fieldgaussian,[tem_THICK,tem_WIDTH,tem_LENGTH],MR.origin,MR.spacing,MR.direction) # ANTs nonmask-Gussian-field
arryresult_Laplace = warp_fieldtoimage3D(dst,fieldlaplace,[tem_THICK,tem_WIDTH,tem_LENGTH],MR.origin,MR.spacing,MR.direction) # ANTs field-Laplace
arryresult_maskLaplace = warp_fieldtoimage3D(dst,fieldmasklaplace,[tem_THICK, tem_WIDTH,tem_LENGTH],MR.origin,MR.spacing,MR.direction) # ANTs field-Laplace

plt.figure(4,figsize=(7,6))
plt.subplot(2,3,2)
plt.title('ourResult (overlap-average)')
resultshow(arryresult,SLIDE,WIDTH,plane='sagittal')
plt.figure(40,figsize=(7,6))
plt.subplot(2,3,2)
plt.title('ourResult (overlap-average)')
resultshow(arryresult,SLIDEAXIAL,WIDTH,plane='axial')

plt.figure(4, figsize=(7, 6))
plt.subplot(2, 3, 3)
plt.title('ourResult (nonmask-Gussian-field)')
resultshow(arryresult_Gaussian,SLIDE,WIDTH,plane='sagittal')
plt.figure(40,figsize=(7,6))
plt.subplot(2,3,3)
plt.title('ourResult (nonmask-Gussian-field)')
resultshow(arryresult_Gaussian,SLIDEAXIAL,WIDTH,plane='axial')

plt.figure(4,figsize=(7,6))
plt.subplot(2,3,4)
plt.title('ourResult (mask-Gussian-field)')
resultshow(arryresult_maskGaussian,SLIDE,WIDTH,plane='sagittal')
plt.figure(40,figsize=(7,6))
plt.subplot(2,3,4)
plt.title('ourResult (mask-Gussian-field)')
resultshow(arryresult_maskGaussian,SLIDEAXIAL,WIDTH,plane='axial')

plt.figure(4,figsize=(7,6))
plt.subplot(2,3,5)
plt.title('ourResult (nonmask-Laplace-field)')
resultshow(arryresult_Laplace,SLIDE,WIDTH,plane='sagittal')
plt.figure(40,figsize=(7,6))
plt.subplot(2,3,5)
plt.title('ourResult (nonmask-Laplace-field)')
resultshow(arryresult_Laplace,SLIDEAXIAL,WIDTH,plane='axial')

plt.figure(4,figsize=(7,6))
plt.subplot(2,3,6)
plt.title('ourResult (mask-Laplace-field)')
resultshow(arryresult_maskLaplace,SLIDE,WIDTH,plane='sagittal')
plt.subplot(2,3,6)
plt.title('ourResult (mask-Laplace-field)')
resultshow(arryresult_maskLaplace,SLIDEAXIAL,WIDTH,plane='axial')

outs = ants.registration(fix_img,move_img,type_of_transform='Affine')  # whole image using ants
reg_img = outs['warpedmovout'] 
arry = reg_img.numpy()
niisave(arry,origin=LS.origin,spacing=LS.spacing,dire=LS.direction,path='../result/save/wholeImageantAffineResult.nii.gz')

outs = ants.registration(fix_img,move_img,type_of_transform ='SyN')  # whole image using ants
reg_img = outs['warpedmovout'] 
arry = reg_img.numpy()
niisave(arry,origin=LS.origin,spacing=LS.spacing,dire=LS.direction,path='../result/save/wholeImageantSyNResult.nii.gz')

'''nifty save'''
#niisave(norm(arry1+arry2+arry3+arry4), origin = sitk_LS.GetOrigin(), spacing = sitk_LS.GetSpacing(), dire=sitk_LS.GetDirection(), path = '../result/save/antAffinePiecewiseResult.nii.gz')
niisave(arry1234,origin=LS.origin,spacing=LS.spacing,dire=LS.direction,path='../result/save/antAffinePiecewiseResult.nii.gz')
niisave(arryresult,origin=LS.origin,spacing=LS.spacing,dire=LS.direction,path='../result/save/overlapFieldPiecewiseResult.nii.gz')
niisave(arryresult_maskGaussian,origin=LS.origin,spacing=LS.spacing,dire=LS.direction,path='../result/save/maskGaussianPiecewiseResult.nii.gz')
niisave(arryresult_Gaussian,origin=LS.origin,spacing=LS.spacing,dire=LS.direction,path='../result/save/GaussianPiecewiseResult.nii.gz')
niisave(arryresult_Laplace,origin=LS.origin,spacing=LS.spacing,dire=LS.direction,path='../result/save/LaplacePiecewiseResult.nii.gz')
niisave(arryresult_maskLaplace,origin=LS.origin,spacing=LS.spacing,dire=LS.direction,path='../result/save/maskLaplacePiecewiseResult.nii.gz')

niisave(mask,origin=LS.origin,spacing=LS.spacing,dire=LS.direction,path='../result/save/mask.nii.gz')

niisave(fieldXYZ[:,:,:,0],origin=LS.origin,spacing=LS.spacing,dire=LS.direction,path='../result/save/Zfield.nii.gz')
niisave(fieldXYZ[:,:,:,1],origin=LS.origin,spacing=LS.spacing,dire=LS.direction,path='../result/save/Xfield.nii.gz')
niisave(fieldXYZ[:,:,:,2],origin=LS.origin,spacing=LS.spacing,dire=LS.direction,path='../result/save/Yfield.nii.gz')

niisave(MR.arry.transpose(2,1,0),origin=LS.origin,spacing=LS.spacing,dire=LS.direction,path='../result/save/mr.nii.gz')

'''syn entire'''

outslist = [[],[],[],[],[],[]]
reg_imglist = [[],[],[],[],[],[]]
arrylist = [[],[],[],[],[],[]]
for i in range(0,SEGMENTS):
    outslist[i] = ants.registration(fix_imglist[i],move_imglist[i],type_of_transform =regisMethod)  
    reg_imglist[i] = outslist[i]['warpedmovout'] 
    arrylist[i] = reg_imglist[i].numpy()


entireOutlist = [[],[],[],[],[],[]]
entireReg_imglist = [[],[],[],[],[],[]]
entireArrylist = [[],[],[],[],[],[]]
entireRegisMethod = 'SyN'
resultArry = [arry1234,arryresult,arryresult_Gaussian,arryresult_maskGaussian,arryresult_Laplace,arryresult_maskLaplace]
saveName = ['SYN_antAffinePiecewiseResult',\
            'SYN_overlapFieldPiecewiseResult',\
            'SYN_GaussianPiecewiseResult',\
            'SYN_maskGaussianPiecewiseResult',\
            'SYN_LaplacePiecewiseResult',\
            'SYN_maskLaplacePiecewiseResult']
for i in range(0,SEGMENTS):
    entireOutlist[i] = ants.registration(fix_img,ants.from_numpy(resultArry[i]),type_of_transform=entireRegisMethod)
    entireReg_imglist[i] = entireOutlist[i]['warpedmovout'] 
    entireArrylist[i] = entireReg_imglist[i].numpy()
    niisave(entireArrylist[i],origin=LS.origin,spacing=LS.spacing,dire=LS.direction,path='../result/save/%s.nii.gz'%saveName[i])

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
sys.path.insert(0, 'C:/Users/fangyan/Desktop/PiecewiseRegistration')
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