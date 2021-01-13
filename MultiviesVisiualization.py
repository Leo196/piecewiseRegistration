# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 12:01:07 2020

@author: fangyan
"""
# sub-images_axial
mrImg1 = img1[:,:,SLIDE].copy()
mrImg1 = np.fliplr(mrImg1)
mrImg1 = mrImg1.transpose((1,0))
lsImg1 = dst1[:,:,SLIDE].copy()
lsImg1 = np.fliplr(lsImg1)
lsImg1 = lsImg1.transpose((1,0))

plt.figure(31,figsize=(6, 8))
plt.subplot(4, 1, 1)
plt.title('subImg1(MR, Light-sheet)')
plt.ylim(0, LENGTH-1)
plt.axis('off') 
plt.imshow(np.hstack((mrImg1, lsImg1)),cmap = 'gray') 

mrImg2 = img2[:,:,SLIDE].copy()
mrImg2 = np.fliplr(mrImg2)
mrImg2 = mrImg2.transpose((1,0))
lsImg2 = dst2[:,:,SLIDE].copy()
lsImg2 = np.fliplr(lsImg2)
lsImg2 = lsImg2.transpose((1,0))

plt.figure(31,figsize=(6, 8))
plt.subplot(4, 1, 2)
plt.title('subImg2(MR, Light-sheet)')
plt.ylim(0, LENGTH-1)
plt.axis('off') 
plt.imshow(np.hstack((mrImg2, lsImg2)),cmap = 'gray') 

mrImg3 = img3[:,:,SLIDE].copy()
mrImg3 = np.fliplr(mrImg3)
mrImg3 = mrImg3.transpose((1,0))

lsImg3 = dst3[:,:,SLIDE].copy()
lsImg3 = np.fliplr(lsImg3)
lsImg3 = lsImg3.transpose((1,0))

plt.figure(31,figsize=(6, 8))
plt.subplot(4, 1, 3)
plt.title('subImg3(MR, Light-sheet)')
plt.ylim(0, LENGTH-1)
plt.axis('off') 
plt.imshow(np.hstack((mrImg3, lsImg3)),cmap = 'gray') 

mrImg4 = img4[:,:,SLIDE].copy() # be careful
mrImg4 = np.fliplr(mrImg4)
mrImg4 = mrImg4.transpose((1,0))
lsImg4 = dst4[:,:,SLIDE].copy()
lsImg4 = np.fliplr(lsImg4)
lsImg4 = lsImg4.transpose((1,0))

plt.figure(31,figsize=(6, 8))
plt.subplot(4, 1, 4)
plt.title('subImg4(MR, Light-sheet)')
plt.ylim(0, LENGTH-1)
plt.axis('off') 
plt.imshow(np.hstack((mrImg4, lsImg4)),cmap = 'gray') 


# result_axial
test = arry1[:,:,SLIDE].copy()
test = np.fliplr(test)
test = test.transpose((1,0))
plt.figure(30,figsize=(6, 8))
plt.subplot(4, 1, 1)
plt.title('result1(MR, Light-sheet)')
plt.ylim(0, 256-1)
plt.axis('off') 
plt.imshow(test,cmap = 'gray')

test = arry2[:,:,SLIDE].copy()
test = np.fliplr(test)
test = test.transpose((1,0))
plt.figure(30,figsize=(6, 8))
plt.subplot(4, 1, 2)
plt.title('result2(MR, Light-sheet)')
plt.ylim(0, 256-1)
plt.axis('off') 
plt.imshow(test,cmap = 'gray')

test = arry3[:,(378-40-255):378-40,SLIDE].copy()
test = np.fliplr(test)
test = test.transpose((1,0))
plt.figure(30,figsize=(6, 8))
plt.subplot(4, 1, 3)
plt.title('result3(MR, Light-sheet)')
plt.ylim(0, 256-1)
plt.axis('off') 
plt.imshow(test,cmap = 'gray')

test = arry4[:,(378-255):378,SLIDE].copy()
test = np.fliplr(test)
test = test.transpose((1,0))
plt.figure(30,figsize=(6, 8))
plt.subplot(4, 1, 4)
plt.title('result4(MR, Light-sheet)')
plt.ylim(0, 256-1)
plt.axis('off') 
plt.imshow(test,cmap = 'gray')

# displacement field_axial
plt.figure(50,figsize=(6, 8))
plt.subplot(4,1,1)
plt.title('field (Z, Y, X) of 1')
show_fieldz = field1[:,:,SLIDE,0].copy()
show_fieldz = np.fliplr(show_fieldz)
show_fieldz = show_fieldz.transpose((1,0))
show_fieldy = field1[:,:,SLIDE,1].copy()
show_fieldy = np.fliplr(show_fieldy)
show_fieldy = show_fieldy.transpose((1,0))
show_fieldx = field1[:,:,SLIDE,2].copy()
show_fieldx = np.fliplr(show_fieldx)
show_fieldx = show_fieldx.transpose((1,0))
plt.ylim(0, WIDTH-1)
mat = plt.imshow(np.hstack((show_fieldz, show_fieldy, show_fieldx)), vmin=-50, vmax=20,cmap='viridis') 
plt.axis('off') 
ax = plt.gca()
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(mat, cax = cax, spacing='uniform')

plt.figure(50,figsize=(6, 8))
plt.subplot(4,1,2)
plt.title('field (Z, Y, X) of 1')
show_fieldz = field2[:,:,SLIDE,0].copy()
show_fieldz = np.fliplr(show_fieldz)
show_fieldz = show_fieldz.transpose((1,0))
show_fieldy = field2[:,:,SLIDE,1].copy()
show_fieldy = np.fliplr(show_fieldy)
show_fieldy = show_fieldy.transpose((1,0))
show_fieldx = field2[:,:,SLIDE,2].copy()
show_fieldx = np.fliplr(show_fieldx)
show_fieldx = show_fieldx.transpose((1,0))
plt.ylim(0, WIDTH-1)
mat = plt.imshow(np.hstack((show_fieldz, show_fieldy, show_fieldx)), vmin=-50, vmax=20,cmap='viridis') 
plt.axis('off') 
ax = plt.gca()
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(mat, cax = cax, spacing='uniform')

plt.figure(50,figsize=(6, 8))
plt.subplot(4,1,3)
plt.title('field (Z, Y, X) of 1')
show_fieldz = field3[:,:,SLIDE,0].copy()
show_fieldz = np.fliplr(show_fieldz)
show_fieldz = show_fieldz.transpose((1,0))
show_fieldy = field3[:,:,SLIDE,1].copy()
show_fieldy = np.fliplr(show_fieldy)
show_fieldy = show_fieldy.transpose((1,0))
show_fieldx = field3[:,:,SLIDE,2].copy()
show_fieldx = np.fliplr(show_fieldx)
show_fieldx = show_fieldx.transpose((1,0))
plt.ylim(0, WIDTH-1)
mat = plt.imshow(np.hstack((show_fieldz, show_fieldy, show_fieldx)), vmin=-50, vmax=20,cmap='viridis') 
plt.axis('off') 
ax = plt.gca()
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(mat, cax = cax, spacing='uniform')

plt.figure(50,figsize=(6, 8))
plt.subplot(4,1,4)
plt.title('field (Z, Y, X) of 1')
show_fieldz = field4[:,:,SLIDE,0].copy()
show_fieldz = np.fliplr(show_fieldz)
show_fieldz = show_fieldz.transpose((1,0))
show_fieldy = field4[:,:,SLIDE,1].copy()
show_fieldy = np.fliplr(show_fieldy)
show_fieldy = show_fieldy.transpose((1,0))
show_fieldx = field4[:,:,SLIDE,2].copy()
show_fieldx = np.fliplr(show_fieldx)
show_fieldx = show_fieldx.transpose((1,0))
plt.ylim(0, WIDTH-1)
mat = plt.imshow(np.hstack((show_fieldz, show_fieldy, show_fieldx)), vmin=-50, vmax=20,cmap='viridis') 
plt.axis('off') 
ax = plt.gca()
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(mat, cax = cax, spacing='uniform')

# displacement field_sagittal

plt.figure(51,figsize=(6, 8))
plt.subplot(4,1,1)
plt.title('field (Z, Y, X) of 1')
show_fieldz = field2[SLIDE,:,:,0].copy()
show_fieldz = np.fliplr(show_fieldz)
show_fieldz = show_fieldz.transpose((1,0))
show_fieldy = field2[SLIDE,:,:,1].copy()
show_fieldy = np.fliplr(show_fieldy)
show_fieldy = show_fieldy.transpose((1,0))
show_fieldx = field2[SLIDE,:,:,2].copy()
show_fieldx = np.fliplr(show_fieldx)
show_fieldx = show_fieldx.transpose((1,0))
plt.ylim(0, WIDTH-1)
mat = plt.imshow(np.hstack((show_fieldz, show_fieldy, show_fieldx)), vmin=-50, vmax=20,cmap='viridis') 
plt.axis('off') 
ax = plt.gca()
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(mat, cax = cax, spacing='uniform')

plt.figure(51,figsize=(6, 8))
plt.subplot(4,1,2)
plt.title('field (Z, Y, X) of 1')
show_fieldz = field3[SLIDE,:,:,0].copy()
show_fieldz = np.fliplr(show_fieldz)
show_fieldz = show_fieldz.transpose((1,0))
show_fieldy = field3[SLIDE,:,:,1].copy()
show_fieldy = np.fliplr(show_fieldy)
show_fieldy = show_fieldy.transpose((1,0))
show_fieldx = field3[SLIDE,:,:,2].copy()
show_fieldx = np.fliplr(show_fieldx)
show_fieldx = show_fieldx.transpose((1,0))
plt.ylim(0, WIDTH-1)
mat = plt.imshow(np.hstack((show_fieldz, show_fieldy, show_fieldx)), vmin=-50, vmax=20,cmap='viridis') 
plt.axis('off') 
ax = plt.gca()
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(mat, cax = cax, spacing='uniform')

plt.figure(51,figsize=(6, 8))
plt.subplot(4,1,3)
plt.title('field (Z, Y, X) of 1')
show_fieldz = field4[SLIDE,:,:,0].copy()
show_fieldz = np.fliplr(show_fieldz)
show_fieldz = show_fieldz.transpose((1,0))
show_fieldy = field4[SLIDE,:,:,1].copy()
show_fieldy = np.fliplr(show_fieldy)
show_fieldy = show_fieldy.transpose((1,0))
show_fieldx = field4[SLIDE,:,:,2].copy()
show_fieldx = np.fliplr(show_fieldx)
show_fieldx = show_fieldx.transpose((1,0))
plt.ylim(0, WIDTH-1)
mat = plt.imshow(np.hstack((show_fieldz, show_fieldy, show_fieldx)), vmin=-50, vmax=20,cmap='viridis') 
plt.axis('off') 
ax = plt.gca()
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(mat, cax = cax, spacing='uniform')