# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 03:28:53 2020

@author: fangyan
"""
import numpy as np
from basicProcessing import denoiseLabels
from basicProcessing import denoiseLabels3D

def postProcessing(labels_MR, labels_LS, ifmorphology):
    # MR
    dilation_mr0 = np.zeros_like(labels_MR)
    dilation_mr1 = np.zeros_like(labels_MR)
    dilation_mr2 = np.zeros_like(labels_MR)
    dilation_mr3 = np.zeros_like(labels_MR)
    maxContour_mr0 = np.zeros_like(labels_MR)
    maxContour_mr1 = np.zeros_like(labels_MR)
    maxContour_mr2 = np.zeros_like(labels_MR)
    maxContour_mr3 = np.zeros_like(labels_MR)
    
    for i in range(0, labels_MR.shape[0]):
        mr0 = labels_MR[i,:,:].copy()
        mr0[mr0!=1] = 0
        mr0[mr0==1] = 1
        dilation_mr0[i,:,:], maxContour_mr0[i,:,:] = denoiseLabels(mr0, ifmorphology)
        
        mr1 = labels_MR[i,:,:].copy()
        mr1[mr1!=2] = 0
        mr1[mr1==2] = 1
        dilation_mr1[i,:,:], maxContour_mr1[i,:,:] = denoiseLabels(mr1, ifmorphology)
        
        mr2 = labels_MR[i,:,:].copy()
        mr2[mr2!=3] = 0
        mr2[mr2==3] = 1
        dilation_mr2[i,:,:], maxContour_mr2[i,:,:] = denoiseLabels(mr2, ifmorphology)
        
        mr3 = labels_MR[i,:,:].copy()
        mr3[mr3!=4] = 0
        mr3[mr3==4] = 1
        dilation_mr3[i,:,:], maxContour_mr3[i,:,:] = denoiseLabels(mr3, ifmorphology)
    
    mr = np.zeros_like(labels_MR) 
    dilation_mr0[dilation_mr0==1] = 1
    dilation_mr1[dilation_mr1==1] = 1 + 1
    dilation_mr2[dilation_mr2==1] = 2 + 1
    dilation_mr3[dilation_mr3==1] = 3 + 1
#    mr = mr + (dilation_mr0 + dilation_mr1 + dilation_mr2 + dilation_mr3 - 1)
    mr = mr + (dilation_mr0 + dilation_mr1 + dilation_mr2 + dilation_mr3)
#    mr[mr==6] = 3 # temp
#    mr[mr==-1] = 4
    mr[mr==0] = 5
    
    maxContour_mr = maxContour_mr0 + maxContour_mr1 + maxContour_mr2 + maxContour_mr3;
    
    # LS
    dilation_ls0 = np.zeros_like(labels_LS)
    dilation_ls1 = np.zeros_like(labels_LS)
    dilation_ls2 = np.zeros_like(labels_LS)
    dilation_ls3 = np.zeros_like(labels_LS)
    maxContour_ls0 = np.zeros_like(labels_LS)
    maxContour_ls1 = np.zeros_like(labels_LS)
    maxContour_ls2 = np.zeros_like(labels_LS)
    maxContour_ls3 = np.zeros_like(labels_LS)
    for i in range(0, labels_LS.shape[0]):
        ls0 = labels_LS[i,:,:].copy()
        ls0[ls0!=1] = 0
        ls0[ls0==1] = 1
        dilation_ls0[i,:,:], maxContour_ls0[i,:,:] = denoiseLabels(ls0, ifmorphology)
        
        ls1 = labels_LS[i,:,:].copy()
        ls1[ls1!=2] = 0
        ls1[ls1==2] = 1
        dilation_ls1[i,:,:], maxContour_ls1[i,:,:] = denoiseLabels(ls1, ifmorphology)
        
        ls2 = labels_LS[i,:,:].copy()
        ls2[ls2!=3] = 0
        ls2[ls2==3] = 1
        dilation_ls2[i,:,:], maxContour_ls2[i,:,:] = denoiseLabels(ls2, ifmorphology)
        
        ls3 = labels_LS[i,:,:].copy()
        ls3[ls3!=4] = 0
        ls3[ls3==4] = 1
        dilation_ls3[i,:,:], maxContour_ls3[i,:,:] = denoiseLabels(ls3, ifmorphology)
    
    ls = np.zeros_like(labels_LS)
    dilation_ls0[dilation_ls0==1] = 1 
    dilation_ls1[dilation_ls1==1] = 1 + 1
    dilation_ls2[dilation_ls2==1] = 2 + 1
    dilation_ls3[dilation_ls3==1] = 3 + 1
#    ls = ls + (dilation_ls0 + dilation_ls1 + dilation_ls2 + dilation_ls3 - 1)
    ls = ls + (dilation_ls0 + dilation_ls1 + dilation_ls2 + dilation_ls3)
##    ls[ls==4] = 5
#    ls[ls==-1] = 4
    ls[ls==0] = 5
    
    maxContour_ls = maxContour_ls0 + maxContour_ls1 + maxContour_ls2 + maxContour_ls3
    
    mr = mr.astype('int8')
    ls = ls.astype('int8')
    
    return mr,maxContour_mr,ls,maxContour_ls,dilation_mr0,dilation_mr1,dilation_mr2,dilation_mr3,dilation_ls0,dilation_ls1,dilation_ls2,dilation_ls3


def postProcessing3D(labels_MR, labels_LS, ifmorphology):
    # MR
    dilation_mr0 = np.zeros_like(labels_MR)
    dilation_mr1 = np.zeros_like(labels_MR)
    dilation_mr2 = np.zeros_like(labels_MR)
    dilation_mr3 = np.zeros_like(labels_MR)
    maxContour_mr0 = np.zeros_like(labels_MR)
    maxContour_mr1 = np.zeros_like(labels_MR)
    maxContour_mr2 = np.zeros_like(labels_MR)
    maxContour_mr3 = np.zeros_like(labels_MR)
    
    mr0 = labels_MR.copy()
    mr0[mr0!=1] = 0
    mr0[mr0==1] = 1
    dilation_mr0, maxContour_mr0 = denoiseLabels3D(mr0, ifmorphology)
    
    mr1 = labels_MR.copy()
    mr1[mr1!=2] = 0
    mr1[mr1==2] = 1
    dilation_mr1, maxContour_mr1 = denoiseLabels3D(mr1, ifmorphology)
    
    mr2 = labels_MR.copy()
    mr2[mr2!=3] = 0
    mr2[mr2==3] = 1
    dilation_mr2, maxContour_mr2 = denoiseLabels3D(mr2, ifmorphology)
    
    mr3 = labels_MR.copy()
    mr3[mr3!=4] = 0
    mr3[mr3==4] = 1
    dilation_mr3, maxContour_mr3 = denoiseLabels3D(mr3, ifmorphology)
    
    mr = np.zeros_like(labels_MR) 
    dilation_mr0[dilation_mr0==1] = 1
    dilation_mr1[dilation_mr1==1] = 1 + 1
    dilation_mr2[dilation_mr2==1] = 2 + 1
    dilation_mr3[dilation_mr3==1] = 3 + 1
#    mr = mr + (dilation_mr0 + dilation_mr1 + dilation_mr2 + dilation_mr3 - 1)
    mr = mr + (dilation_mr0 + dilation_mr1 + dilation_mr2 + dilation_mr3)
#    mr[mr==6] = 3 # temp
#    mr[mr==-1] = 4
    mr[mr==0] = 5
    
    maxContour_mr = maxContour_mr0 + maxContour_mr1 + maxContour_mr2 + maxContour_mr3;
    
    # LS
    dilation_ls0 = np.zeros_like(labels_LS)
    dilation_ls1 = np.zeros_like(labels_LS)
    dilation_ls2 = np.zeros_like(labels_LS)
    dilation_ls3 = np.zeros_like(labels_LS)
    maxContour_ls0 = np.zeros_like(labels_LS)
    maxContour_ls1 = np.zeros_like(labels_LS)
    maxContour_ls2 = np.zeros_like(labels_LS)
    maxContour_ls3 = np.zeros_like(labels_LS)

    ls0 = labels_LS.copy()
    ls0[ls0!=1] = 0
    ls0[ls0==1] = 1
    dilation_ls0, maxContour_ls0 = denoiseLabels3D(ls0, ifmorphology)
    
    ls1 = labels_LS.copy()
    ls1[ls1!=2] = 0
    ls1[ls1==2] = 1
    dilation_ls1, maxContour_ls1 = denoiseLabels3D(ls1, ifmorphology)
    
    ls2 = labels_LS.copy()
    ls2[ls2!=3] = 0
    ls2[ls2==3] = 1
    dilation_ls2, maxContour_ls2 = denoiseLabels3D(ls2, ifmorphology)
    
    ls3 = labels_LS.copy()
    ls3[ls3!=4] = 0
    ls3[ls3==4] = 1
    dilation_ls3, maxContour_ls3 = denoiseLabels3D(ls3, ifmorphology)
    
    ls = np.zeros_like(labels_LS)
    dilation_ls0[dilation_ls0==1] = 1 
    dilation_ls1[dilation_ls1==1] = 1 + 1
    dilation_ls2[dilation_ls2==1] = 2 + 1
    dilation_ls3[dilation_ls3==1] = 3 + 1
#    ls = ls + (dilation_ls0 + dilation_ls1 + dilation_ls2 + dilation_ls3 - 1)
    ls = ls + (dilation_ls0 + dilation_ls1 + dilation_ls2 + dilation_ls3)
##    ls[ls==4] = 5
#    ls[ls==-1] = 4
    ls[ls==0] = 5
    
    maxContour_ls = maxContour_ls0 + maxContour_ls1 + maxContour_ls2 + maxContour_ls3
    
    mr = mr.astype('int8')
    ls = ls.astype('int8')
    
    return mr,maxContour_mr,ls,maxContour_ls,dilation_mr0,dilation_mr1,dilation_mr2,dilation_mr3,dilation_ls0,dilation_ls1,dilation_ls2,dilation_ls3