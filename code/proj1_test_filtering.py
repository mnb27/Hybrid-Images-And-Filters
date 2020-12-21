"""
% this script has test cases to help you test my_imfilter() which you will
% write. You should verify that you get reasonable output here before using
% your filtering to construct a hybrid image in proj1.m. The outputs are
% all saved and you can include them in your writeup. You can add calls to
% imfilter() if you want to check that my_imfilter() is doing something
% similar. """

import matplotlib.pyplot as plt
import os
import cv2
from os.path import join
import numpy as np
from PIL import Image
import matplotlib.image as mpimg
from skimage.transform import rescale, resize, downscale_local_mean

from my_imfilter import my_imfilter
from scipy import signal
from skimage import img_as_float


#%% close all figures
plt.close('all')

#%% Setup
'''
img_path = join(''.join([PATH_INPUT_IMAGES, '/', 'bicycle.bmp']))
test_image = mpimg.imread(img_path);
'''

test_image = mpimg.imread('../data/cat.bmp');
test_image = resize(test_image, (test_image.shape[0] // 2, test_image.shape[1] // 2), anti_aliasing=True)#resizing to speed up testing
plt.figure(1)
plt.imshow(test_image)
plt.show()


#%% This filter should do nothing regardless of the padding method you use.
""" Identity filter """

identity_filter = np.asarray([[0,0,0],[0,1,0],[0,0,0]]);
identity_image  = my_imfilter(test_image, identity_filter)

plt.figure(2)
plt.title("Identity filter")
plt.imshow(identity_image);
mpimg.imsave('Results/identity_image.jpg',np.clip(identity_image, 0, 1.0));
#

#%% This filter should remove some high frequencies
""" Small blur with a box filter """

blur_filter = np.asarray([[1,1,1],[1,1,1],[1,1,1]]);
blur_filter = blur_filter / np.sum(blur_filter); # making the filter sum to 1
#
blur_image = my_imfilter(test_image, blur_filter);
#
plt.figure(3) 
plt.title("Small blur with a box filter")
plt.imshow(blur_image);
plt.show()
mpimg.imsave('Results/blur_image.jpg',np.clip(blur_image, 0, 1.0));
#

#%% Large blur
""" This blur would be slow to do directly, so we instead use the fact that
     Gaussian blurs are separable and blur sequentially in each direction. """

large_1d_blur_filter = np.asarray([])# import values from fspecial('Gaussian', [25 1], 10) here
large_blur_image = my_imfilter(test_image, large_1d_blur_filter);
large_blur_image = my_imfilter(large_blur_image, large_1d_blur_filter_transpose) #implement large_1d_blur_filter_transpose
#
plt.figure(4) 
plt.title("Large blur")
plt.imshow(large_blur_image);
plt.show()
mpimg.imsave('Results/large_blur_image.jpg', np.clip(large_blur_image, 0, 1.0)); 
#
#% %If you want to see how slow this would be to do naively, try out this
#% %equivalent operation:
#% tic %tic and toc run a timer and then print the elapsted time
#% large_blur_filter = fspecial('Gaussian', [25 25], 10);
#% large_blur_image = my_imfilter(test_image, large_blur_filter);
#% toc 
#
#%% Oriented filter (Sobel Operator)
""" Edge Filter """
sobel_filter = np.asarray([[-1,0,1],[-2,0,2],[-1,0,1]]) #should respond to horizontal gradients
sobel_image = my_imfilter(test_image, sobel_filter);
#
# 0.5 added because the output image is centered around zero otherwise and mostly black
plt.figure(5)
plt.title("Sobel Edge Filter")
plt.imshow(sobel_image + 0.5)
plt.show()
mpimg.imsave('Results/sobel_image.jpg',np.clip(sobel_image + 0.5, 0, 1.0)) 
#
#
#%% High pass filter (Discrete Laplacian)
""" Laplacian Filter """
laplacian_filter = np.asarray([[0,1,0],[1,-4,1],[0,1,0]])
laplacian_image = my_imfilter(test_image, laplacian_filter)
# 0.5 added because the output image is centered around zero otherwise and mostly black
plt.figure(6)
plt.title("High pass filter (Discrete Laplacian)")
plt.imshow(laplacian_image + 0.5)
plt.show()
mpimg.imsave('Results/laplacian_image.jpg', np.clip(laplacian_image + 0.5, 0, 1.0))
#
#%% High pass "filter" alternative
""" High pass filter example we saw in class """
high_pass_image = test_image - blur_image #simply subtract the low frequency content
plt.figure(7)
plt.title("High pass filter alternative")
plt.imshow(high_pass_image + 0.5);
plt.show()
mpimg.imsave('Results/high_pass_image.jpg',np.clip(high_pass_image + 0.5, 0, 1.0))

'''
f, axarr = plt.subplots(2,3)
axarr[0,0].imshow(identity_image)
axarr[0,1].imshow(blur_image)
axarr[0,2].imshow(large_blur_image)
axarr[1,0].imshow(sobel_image + 0.5)
axarr[1,1].imshow(laplacian_image + 0.5)
axarr[1,2].imshow(high_pass_image + 0.5)
'''

""" adopted from code by James Hays (GATech)"""