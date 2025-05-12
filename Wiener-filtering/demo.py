from scipy.ndimage import convolve
import hw3_helper_utils
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

import inverse_filtering
import wiener_filtering
import optimal_wiener_parameter

# ----------------------------------------------------------------------------------------------------------------------

# LOAD IMAGE

# set the filepath to the image file
filename = "cameraman.tif"

# read the image into a PIL entity
img = Image.open(fp=filename)

# keep only the Luminance component of the image
bw_img = img.convert("L")

# obtain the underlying np array
img_array = np.array(bw_img, dtype=np.uint8)

# ----------------------------------------------------------------------------------------------------------------------

# IMAGE PREPROCESSING

# normalize intensity value to be in the range [0, 1]
x = img_array.astype(np.float64)
x /= 255

# image shape
m, n = x.shape

# ----------------------------------------------------------------------------------------------------------------------

# DISTORTION AND NOISE SYSTEM MODEL

# set noise level
noise_level = 0.2

# fixed seed to generate the same random numbers across different runs
np.random.seed(0)

# create white noise
v = noise_level*np.random.randn(*x.shape)

# variance of noise signal
noise_var = np.var(v)

# set length and angle of motion blur filter
length = 20
angle = 30

# create motion blur filter
h = hw3_helper_utils.create_motion_blur_filter(length, angle)

# distortion filter shape
l, p = h.shape

# obtain the filtered image
y0 = convolve(x, h, mode='wrap')

# generate the noisy image
y = y0 + v

# ----------------------------------------------------------------------------------------------------------------------

# APPLY INVERSE FILTER H^-1

# apply inverse filter on the output of the distortion/noise system
x_inv = inverse_filtering.my_inverse_filter(y, h)

# apply inverse filter on the output of the distortion system only
x_inv0 = inverse_filtering.my_inverse_filter(y0, h)

# ----------------------------------------------------------------------------------------------------------------------

# APPLY WIENER FILTER

# calculate power of original image
power_x = np.var(x)

# set parameter k of the Wiener Filter
k = optimal_wiener_parameter.calculate_k(x_inv0, y, h, power_x, noise_var)
print("Optimal value of Wiener filter k parameter: ", k)

# apply Wiener Filter
x_hat = wiener_filtering.my_wiener_filter(y, h, k)

# ----------------------------------------------------------------------------------------------------------------------

# PLOT PROCESSING RESULTS

fig, axs = plt.subplots(nrows=2, ncols=3)

# plot original image
axs[0, 0].imshow(x, cmap='gray')
axs[0, 0].set_title("Original image x")

# plot distorted image
axs[0, 1].imshow(y0, cmap='gray')
axs[0, 1].set_title("Clean image y0")

# plot distorted and noisy image
axs[0, 2].imshow(y, cmap='gray')
axs[0, 2].set_title("Blurred and noisy image y")

# plot output image of inverse filtering on distorted image
axs[1, 0].imshow(x_inv0, cmap='gray')
axs[1, 0].set_title("Inverse filtering noiseless output x_inv0")

# plot output image of inverse filtering on distorted and noisy image
axs[1, 1].imshow(x_inv, cmap='gray')
axs[1, 1].set_title("Inverse filtering noisy output x_inv")

# plot output image of wiener filtering on distorted and noisy image
axs[1, 2].imshow(x_hat, cmap='gray')
axs[1, 2].set_title("Wiener filtering output x_hat")

plt.show()

# ----------------------------------------------------------------------------------------------------------------------
