import numpy as np

# number of quantization levels of 8-bit grayscale digital images
NUM_LEVELS = 256


def get_equalization_transform_of_img(img_array: np.ndarray):
    """ Returns the histogram equalization transform of an 8-bit grayscale
        image with 256 quantization levels

        :param img_array: 8-bit grayscale input image
        :type img_array: numpy.ndarray(dtype= numpy.uint8)

        :returns: equalization transform of input image
        :rtype: numpy.ndarray(dtype= numpy.uint8, shape= (1, 256))
    """

    # total number of samples of input image
    n = img_array.size

    # reshape <img_array> to be a 1-dimensional array
    img_array_1d = img_array.flatten()

    # compute non-normalized histogram of image <img_array>
    histogram = np.zeros(NUM_LEVELS, dtype=np.uint64)
    for pixel in img_array_1d:
        # count how many times an intensity value appears in the image
        histogram[pixel] += 1

    # normalize histogram
    histogram = np.divide(histogram, n)

    # compute cumulative distribution function of input image
    cdf = [np.sum(histogram[:(i+1)]) for i in range(NUM_LEVELS)]

    # compute equalization transform of input image
    equalization_transform = np.round((cdf - cdf[0])/(1 - cdf[0])*(NUM_LEVELS-1))

    # return equalization transform as a numpy array
    return equalization_transform


def perform_global_hist_equalization(img_array: np.ndarray):
    """ Returns the output image of global histogram equalization transform 
        on an 8-bit grayscale image with 256 quantization levels

        :param img_array: 8-bit grayscale input image
        :type img_array: numpy.ndarray(dtype= numpy.uint8)

        :returns: equalized output image
        :rtype: numpy.ndarray(dtype= numpy.uint8)
    """

    # get the equalization transform of input image
    equalization_transform = get_equalization_transform_of_img(img_array)

    # apply the equalization transform to the input image
    equalized_img = np.array([equalization_transform[level] for level in img_array])

    return equalized_img
