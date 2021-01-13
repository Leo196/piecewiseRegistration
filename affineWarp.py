# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 17:37:40 2020

@author: fangyan
"""
import numpy as np
import math
from scipy.interpolate import RegularGridInterpolator
import operator
import scipy.ndimage
import gc # clear memory

def shulie(first,end,step):
    x = []
    for i in np.arange(first, end,step):
        x.append(i)
    return x


def affine(subimg,ty,tx,theta):
    
    a = math.cos(math.radians(theta))
    b = math.sin(math.radians(theta))
    c = -math.sin(math.radians(theta))
    d = math.cos(math.radians(theta))
    matrix = np.float32([[a, b, ty], [c, d, tx]])
    matrix = np.row_stack([matrix,np.array([0, 0, 1])])
    
    return matrix


def warpAffine(im, A, output_shape): # backward, affine matrix
    """ Warps (h,w) image im using affine (3,3) matrix A
    producing (output_shape[0], output_shape[1]) output image
    with warped = A*input, where warped spans 1...output_size.
    Uses nearest neighbor interpolation."""

    x = np.zeros((output_shape[0], output_shape[1]))
    for i in range(0, output_shape[0]):
        for j in range(0, output_shape[1]):
            M = np.linalg.inv(A).dot(np.array([i,j,1]))
            p, q = M[0], M[1]
            rp = int(round(p))
            rq = int(round(q))
            if rp<im.shape[0] and rp>=0 and rq<im.shape[1] and rq>=0:
                x[i, j] = im[rp, rq]

    return x


#def warp_field(A, output_shape): # A refers to the inverse of affine matrix
#
#    x = np.zeros((output_shape[0], output_shape[1]))
#    y = np.zeros((output_shape[0], output_shape[1]))
#    for i in range(0, output_shape[0]):
#        for j in range(0, output_shape[1]):
##            M = np.linalg.inv(A).dot(np.array([i, j, 1]))
#            M = A.dot(np.array([i, j, 1]))
#            x[i, j] = i - M[0]           
#            y[i, j] = j - M[1]
#        
#    return np.stack((x, y) , axis = 0)

def warp_field(A, output_shape): # A refers to the inverse of affine matrix

    y = np.zeros((output_shape[0], output_shape[1]))
    x = np.zeros((output_shape[0], output_shape[1]))
    for i in range(0, output_shape[0]):   # i,j belong to fixed image
        for j in range(0, output_shape[1]):
#            M = np.linalg.inv(A).dot(np.array([i, j, 1]))
            M = A.dot(np.array([i, j, 1]))
            y[i, j] = i - M[0]          
            x[i, j] = j - M[1] 
        
    return np.stack((y, x) , axis = 0)


def warp_field3D(A, output_shape, origin, spacing, direction): # A refers to the inverse of affine matrix

#    x = np.zeros((output_shape[1], output_shape[2])) # 378 * 256
#    y = np.zeros((output_shape[1], output_shape[2])) # 378 * 256
#    z = np.zeros((output_shape[1], output_shape[2])) # 378 * 256
    field = np.zeros((output_shape[0], output_shape[1], output_shape[2], 3)) # 378 * 256 * thick * 3 [x,y,z]
    for i in range(0, output_shape[1]):
        for j in range(0, output_shape[2]):
            for k in range(0, output_shape[0]):
#                M = np.linalg.inv(A).dot(np.array([k,i, j, 1]))
                M = A.dot(np.array([k,i, j, 1]))
                
                field[k,i,j,0] = k - M[0]
                field[k,i,j,1] = i - M[1] 
                field[k,i,j,2] = j - M[2]
                
    return field


#def warp_fieldtoimage(im, field, output_shape): # backward, field matrix
#
#    x = np.zeros((output_shape[0], output_shape[1]))  # i,j from target image
#    for i in range(0, output_shape[0]):
#        for j in range(0, output_shape[1]):
#            dsx = field[0, i, j] 
#            dsy = field[1, i, j]
#            p = i - dsx 
#            q = j - dsy 
#            rp = int(round(p))
#            rq = int(round(q))
#            if rp<im.shape[0] and rp>=0 and rq<im.shape[1] and rq>=0:
#                x[i, j] = im[rp, rq]
#                
#    return x

def warp_fieldtoimage(im, field, output_shape): # backward, field matrix

    x = np.zeros((output_shape[0], output_shape[1]))  # i,j from target image
    for i in range(0, output_shape[0]):
        for j in range(0, output_shape[1]):
            dsy = field[0, i, j] 
            dsx = field[1, i, j]
            p = i - dsy 
            q = j - dsx
            rp = int(round(p))
            rq = int(round(q))
            if rp<im.shape[0] and rp>=0 and rq<im.shape[1] and rq>=0:
                x[i, j] = im[rp, rq]
                
    return x

def warp_fieldtoimage3D(im, field, output_shape, origin, spacing, direction): # backward, field matrix 5,378,256
# https://www.codenong.com/16217995/

    im = im.astype('float32')
    field = field.astype('float32')
    spacing = spacing.astype('float32')
    x = np.zeros((output_shape[0], output_shape[1], output_shape[2]))  # k,i,j from target image
    a = []
    b = []
    c = []

    for k in range(0, output_shape[0]):
        for i in range(0, output_shape[1]):
            for j in range(0, output_shape[2]):
                dsz = field[k, i, j, 0]  # image coordinate
                dsy = field[k, i, j, 1] 
                dsx = field[k, i, j, 2]
    
                r = k - dsz
                p = i - dsy
                q = j - dsx
                
                a.append(r)
                b.append(p)
                c.append(q)
                
    coords = np.array([a,b,c])
#    idx = coords / spacing[(slice(None),) + (None,)*(coords.ndim-1)]
    x = scipy.ndimage.map_coordinates(im, coords, order=3).reshape(output_shape[0],output_shape[1],output_shape[2]) # 0:nearest; 1:linear; 3:spline
    x[x<0]=0
    
    # release memory
    del a,b,c,coords,im,field
    gc.collect()
    
    return x

#def warp_fieldtoimage3D(im, field, output_shape, origin, spacing, direction): # backward, field matrix 5,378,256
#    
#    x = np.zeros((output_shape[0], output_shape[1], output_shape[2]))  # k,i,j from target image
#    for i in range(0, output_shape[1]):
#        for j in range(0, output_shape[2]):
#            for k in range(0, output_shape[0]):
##                k = int(round(k*spacing[0]+origin[0])) # img coordinate => physical coordinate
##                i = int(round(i*spacing[1]+origin[1]))
##                j = int(round(j*spacing[2]+origin[2]))
#                
#                dsz = field[k, i, j, 0]  # physical coordinate
#                dsy = field[k, i, j, 1] 
#                dsx = field[k, i, j, 2]
#
#                r = k - dsz
#                p = i - dsy
#                q = j - dsx
##                r = (r-origin[0])/spacing[0]
##                p = (p-origin[1])/spacing[1]
##                q = (q-origin[2])/spacing[2]
#                
#                rr = int(round(r))
#                rp = int(round(p))
#                rq = int(round(q))
#                
#                if rr<im.shape[0] and rr>=0 and rp<im.shape[1] and rp>=0 and rq<im.shape[2] and rq>=0:
#                    x[k, i, j] = im[rr, rp, rq]
#                
#    return x