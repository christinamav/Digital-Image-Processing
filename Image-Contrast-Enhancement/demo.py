from PIL import Image
import matplotlib.pyplot as plt
from global_hist_eq import *
from adaptive_hist_eq import *
from math import floor, ceil

# set the filepath to the image file
filename = "input_img.png"

# read the image into a PIL entity
img = Image.open(fp=filename)

# keep only the Luminance component of the image
bw_img = img.convert("L")

# obtain the underlying np array
img_array = np.array(bw_img)

# ----------------------------------------------------------------------------------------------------------------------

# show input image
plt.figure(1)

plt.imshow(img_array, cmap="gray")
plt.title("Input Image")

# plot histogram of input image
img_array_1d = img_array.flatten()

plt.figure(2)
plt.hist(img_array_1d, bins=NUM_LEVELS)
plt.title("Histogram of Input Image")

# ----------------------------------------------------------------------------------------------------------------------

# GLOBAL HISTOGRAM EQUALIZATION

# calculate equalization transform of input image
eq_transform = get_equalization_transform_of_img(img_array)

intensity_values = range(NUM_LEVELS)

plt.figure(3)
plt.step(intensity_values, eq_transform, where='post', )
plt.title("Global Histogram Equalization transform\nof Input Image")

# perform global histogram equalization to image
equalized_img_array = perform_global_hist_equalization(img_array)

# show output image
plt.figure(4)
plt.imshow(equalized_img_array, cmap="gray")
plt.title("Output Image of \nGlobal Histogram Equalization transform")

# plot histogram of output image
equalized_img_array_1d = equalized_img_array.flatten()

plt.figure(5)
plt.hist(equalized_img_array_1d, bins=NUM_LEVELS)
plt.title("Histogram of Output Image of \nGlobal Histogram Equalization transform")

# ----------------------------------------------------------------------------------------------------------------------

# ADAPTIVE HISTOGRAM EQUALIZATION

# image dimensions
(m, n) = img_array.shape

# contextual region height
region_len_h = 64
# contextual region width
region_len_w = 48

input_img_array = img_array

# check if image dimensions are divided exactly by respective
# contextual region dimensions

# if image height is not divisible by contextual region height
h_res = m % region_len_h
if h_res != 0:
    # crop image height
    input_img_array = input_img_array[floor(h_res/2): m - ceil(h_res/2), :]
# if image width is not divisible by contextual region width
w_res = n % region_len_w
if w_res != 0:
    # crop image width
    input_img_array = input_img_array[:, floor(w_res/2): n - ceil(w_res/2)]

# perform adaptive histogram equalization to image
equalized_img_array = perform_adaptive_hist_equalization(input_img_array, region_len_h, region_len_w)

# show output image
plt.figure(6)
plt.imshow(equalized_img_array, cmap="gray")
plt.title("Output Image of \nAdaptive Histogram Equalization transform")

# plot histogram of output image
equalized_img_array_1d = equalized_img_array.flatten()

plt.figure(7)
plt.hist(equalized_img_array_1d, bins=NUM_LEVELS)
plt.title("Histogram of Output Image of\n Adaptive Histogram Equalization transform")

# ----------------------------------------------------------------------------------------------------------------------

# ADAPTIVE HISTOGRAM EQUALIZATION WITHOUT THE USE OF BI-LINEAR INTERPOLATION

# perform adaptive histogram equalization to image
equalized_img_array = perform_no_interpolation_ahe(input_img_array, region_len_h, region_len_w)

# show output image
plt.figure(8)
plt.imshow(equalized_img_array, cmap="gray")
plt.title("Output Image of \nAdaptive Histogram Equalization transform\nwithout use of Bi-linear Interpolation")

# plot histogram of output image
equalized_img_array_1d = equalized_img_array.flatten()

plt.figure(9)
plt.hist(equalized_img_array_1d, bins=NUM_LEVELS)
plt.title("Histogram of Output Image of \nAdaptive Histogram Equalization transform\nwithout use of Bi-linear "
          "Interpolation")

plt.show()

# ----------------------------------------------------------------------------------------------------------------------
