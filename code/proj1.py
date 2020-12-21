"""
% Before trying to construct hybrid images, it is suggested that you
% implement my_imfilter.m and then debug it using proj1_test_filtering.m

% Debugging tip: You can split your MATLAB code into cells using "%%"
% comments. The cell containing the cursor has a light yellow background,
% and you can press Ctrl+Enter to run just the code in that cell. This is
% useful when projects get more complex and slow to rerun from scratch
"""

import matplotlib.pyplot as plt
import os
import cv2
from os.path import join
import numpy as np
from PIL import Image
import matplotlib.image as mpimg
from skimage.transform import rescale, resize, downscale_local_mean


from skimage import img_as_float
from scipy import signal
from my_imfilter import my_imfilter
from vis_hybrid_image import vis_hybrid_image


#%% close all figures
plt.close('all') # closes all figures


# bicycle motorcycle
# cat dog
# einstein marilyn
# bird plane
# fish submarine

'''
ROOT = os.getcwd()
PATH_INPUT_IMAGES = join(ROOT, "data")
PATH_OUTPUT_IMAGES = join(ROOT, "Results")

dog_img_path = join(''.join([PATH_INPUT_IMAGES, '/', 'dog.bmp']))
cat_img_path = join(''.join([PATH_INPUT_IMAGES, '/', 'cat.bmp']))

image1 = mpimg.imread(dog_img_path)
image2 = mpimg.imread(cat_img_path)

'''


#%% Setup
#% read images and convert to floating point format
image1 = mpimg.imread('../data/dog.bmp')
image2 = mpimg.imread('../data/cat.bmp')

image1 = img_as_float(image1) #will provide the low frequencies
image2 = img_as_float(image2) #will provide the high frequencies


"""
% Several additional test cases are provided for you, but feel free to make
% your own (you'll need to align the images in a photo editor such as
% Photoshop). The hybrid images will differ depending on which image you
% assign as image1 (which will provide the low frequencies) and which image
% you asign as image2 (which will provide the high frequencies)
"""

""" %% Filtering and Hybrid Image construction """
# Try for values between 1-10 and see which outputs the best hybrid image
cutoff_frequency = 7

"""This is the standard deviation, in pixels, of the 
% Gaussian blur that will remove the high frequencies from one image and 
% remove the low frequencies from another image (by subtracting a blurred
% version from the original version). You will want to tune this for every
% image pair to get the best results. """

#filter=[] insert values from fspecial('Gaussian', cutoff_frequency*4+1, cutoff_frequency) here
filter = np.reshape(np.asarray(signal.get_window(('gaussian', cutoff_frequency), cutoff_frequency*4 + 1)), (cutoff_frequency*4 + 1, 1)) ;
filter = filter/np.sum(filter)
filter_transpose = np.transpose(filter)

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE BELOW. Use my_imfilter to create 'low_frequencies' and
% 'high_frequencies' and then combine them to create 'hybrid_image'
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Remove the high frequencies from image1 by blurring it. The amount of
% blur that works best will vary with different image pairs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
"""

#low_frequencies = 
low_frequencies = my_imfilter(image1, filter)
low_frequencies = my_imfilter(low_frequencies, filter_transpose)

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Remove the low frequencies from image2. The easiest way to do this is to
% subtract a blurred version of image2 from the original version of image2.
% This will give you an image centered at zero with negative values.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""

#high_frequencies = 
low_frequencies_2 = my_imfilter(image2, filter)
low_frequencies_2 = my_imfilter(low_frequencies_2, filter_transpose)
high_frequencies = image2 - low_frequencies_2

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Combine the high frequencies and low frequencies
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""
#hybrid_image = 
hybrid_image = low_frequencies + high_frequencies


#%% Visualize and save outputs

imgLr = np.clip(low_frequencies,0,1.0)
imgRr = np.clip(high_frequencies + 0.5,0,1.0)

f = plt.figure()
f.add_subplot(1,2, 1)
plt.title("Low Frequency")
plt.imshow(imgLr)
f.add_subplot(1,2, 2)
plt.title("High Frequency")
plt.imshow(imgRr)
plt.show(block=True)

vis = vis_hybrid_image(hybrid_image) #see function script vis_hybrid_image.py
plt.figure(3)
plt.imshow(np.clip(vis,0,1))
plt.show()


'''
plt.figure(1)
plt.imshow(low_frequencies)
plt.figure(2)
plt.imshow(high_frequencies + 0.5);
vis = vis_hybrid_image(hybrid_image) #see function script vis_hybrid_image.py
plt.figure(3)
plt.imshow(vis)

# outputPath = join(''.join([PATH_OUTPUT_IMAGES, '/']))

mpimg.imsave('Results/low_frequencies.jpg',np.clip(low_frequencies, 0, 1.0)) 
mpimg.imsave('Results/high_frequencies.jpg',np.clip(high_frequencies + 0.5, 0, 1.0)) 
mpimg.imsave('Results/hybrid_image.jpg',np.clip(hybrid_image, 0, 1.0))
mpimg.imsave('Results/hybrid_image_scales.jpg',np.clip(vis, 0, 1.0))

'''