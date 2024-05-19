import numpy as np
import global_hist_eq


def calculate_eq_transformations_of_regions(img_array: np.ndarray, region_len_h: int, region_len_w: int):
    """ Returns the histogram equalization transform of each contextual region
        that the input image is split into

        :param img_array: 8-bit grayscale input image
        :type img_array: numpy.ndarray(dtype= numpy.uint8)
        :param region_len_h: height (in number of pixels) of each contextual
            region of image
        :type region_len_h: int
        :param region_len_w: width (in number of pixels) of each contextual
            region of image
        :type region_len_w: int
        :returns: equalization transform of each contextual region
        :rtype: Dict[Tuple, numpy.ndarray]
    """

    # find input image dimensions
    m = img_array.shape[0]
    n = img_array.shape[1]

    # split the input image into contextual regions
    # each contextual region can be represented by a tuple
    # containing the indices of the upper left pixel of the region
    regions = [(i, j) for i in range(0, m, region_len_h) for j in range(0, n, region_len_w)]

    # initialize return variable as a dictionary
    region_to_eq_transform = {}

    # compute equalization transform for each contextual region
    for region in regions:
        # find image array that corresponds to region of input image
        img_region_array = img_array[region[0]:(region[0] + region_len_h), region[1]:(region[1] + region_len_w)]
        # compute equalization transform on the above image array
        eq_transform = global_hist_eq.get_equalization_transform_of_img(img_region_array)
        # update the dictionary with the key-value pair {region: region transform}
        region_to_eq_transform.update({region: eq_transform})

    return region_to_eq_transform


def perform_adaptive_hist_equalization(img_array: np.ndarray, region_len_h: int, region_len_w: int):
    """ Returns the adaptive histogram equalization transform of input image,
        using contextual regions with height <region_len_h> and width <region_len_w>

        :param img_array: 8-bit grayscale input image
        :type img_array: numpy.ndarray(dtype= numpy.uint8)
        :param region_len_h: height (in number of pixels) of each contextual
            region of image
        :type region_len_h: int
        :param region_len_w: width (in number of pixels) of each contextual
            region of image
        :type region_len_w: int
        :returns: 8-bit grayscale output image of transform
        :rtype: numpy.ndarray(dtype= numpy.uint8)
    """

    # find input image dimensions
    m = img_array.shape[0]
    n = img_array.shape[1]

    # initialize output image as a numpy ndarray
    equalized_img = np.zeros((m, n))

    # compute histogram equalization transform for each contextual region
    # of input image
    region_to_eq_transform = calculate_eq_transformations_of_regions(img_array, region_len_h, region_len_w)

    # set a list of contextual regions
    regions = region_to_eq_transform.keys()

    # find contextual centers of each region
    # initialize a dictionary to hold key-value pairs {contextual region: contextual center}
    region_to_center = {}
    # iterate through all contextual regions of the input image
    for region in regions:
        # find center of contextual region
        center = (region[0] + region_len_h // 2, region[1] + region_len_w // 2)
        # update dictionary with the new key-value pair
        region_to_center.update({region: center})

    # iterate through all image pixels
    for i in range(m):
        for j in range(n):
            # find the contextual region each pixel resides in
            region = ((i // region_len_h) * region_len_h, (j // region_len_w) * region_len_w)

            # find the contextual center of this region
            region_center = region_to_center[region]

            # find the equalization transform of this contextual region
            eq_transform = region_to_eq_transform[region]

            # check in which category the pixel falls in
            if (i, j) == region_center:
                # pixel (i, j) is a contextual center

                # use the equalization transform of the contextual region
                # that the pixel resides in
                equalized_img[i, j] = eq_transform[img_array[i, j]]
            elif region[0] == 0 and i < region_center[0] or \
                    region[0] == m - region_len_h and i > region_center[0] or \
                    region[1] == 0 and j < region_center[1] or \
                    region[1] == n - region_len_w and j > region_center[1]:
                # pixel (i, j) is an outer point

                # use the equalization transform of the contextual region
                # that the pixel resides in
                equalized_img[i, j] = eq_transform[img_array[i, j]]
            else:
                # pixel (i, j) is an inner point

                # use bi-linear interpolation between the equalization transforms
                # of the 4 adjacent contextual regions

                # find the 4 adjacent contextual centers depending on the relative position
                # between the pixel (i, j) and the center of the contextual region it resides in

                # initialize adjacent contextual centers with default value
                h_minus = None
                h_plus = None
                w_minus = None
                w_plus = None

                if region[1] == n - region_len_w and j == region_center[1]:
                    # pixels in the rightmost contextual regions, that are collinear
                    # with their contextual centers
                    if i > region_center[0]:
                        h_minus = region_center[0]
                        w_plus = region_center[1]
                        h_plus = h_minus + region_len_h
                        w_minus = w_plus - region_len_w
                    elif i < region_center[0]:
                        h_plus = region_center[0]
                        w_plus = region_center[1]
                        h_minus = h_plus - region_len_h
                        w_minus = w_plus - region_len_w
                elif region[0] == m - region_len_h and i == region_center[0]:
                    # pixels in the lowermost contextual regions, that are collinear
                    # with their contextual centers
                    if j > region_center[1]:
                        h_plus = region_center[0]
                        w_minus = region_center[1]
                        h_minus = h_plus - region_len_h
                        w_plus = w_minus + region_len_w
                    elif j < region_center[1]:
                        h_plus = region_center[0]
                        w_plus = region_center[1]
                        h_minus = h_plus - region_len_h
                        w_minus = w_plus - region_len_w
                else:
                    # all other pixels
                    if i >= region_center[0] and j >= region_center[1]:
                        h_minus = region_center[0]
                        w_minus = region_center[1]
                        h_plus = h_minus + region_len_h
                        w_plus = w_minus + region_len_w

                    if i >= region_center[0] and j < region_center[1]:
                        h_minus = region_center[0]
                        w_plus = region_center[1]
                        h_plus = h_minus + region_len_h
                        w_minus = w_plus - region_len_w

                    if i < region_center[0] and j >= region_center[1]:
                        h_plus = region_center[0]
                        w_minus = region_center[1]
                        h_minus = h_plus - region_len_h
                        w_plus = w_minus + region_len_w

                    if i < region_center[0] and j < region_center[1]:
                        h_plus = region_center[0]
                        w_plus = region_center[1]
                        h_minus = h_plus - region_len_h
                        w_minus = w_plus - region_len_w

                # find corresponding adjacent contextual regions
                region_minus_minus = (h_minus - region_len_h // 2, w_minus - region_len_w // 2)
                region_minus_plus = (h_minus - region_len_h // 2, w_plus - region_len_w // 2)
                region_plus_minus = (h_plus - region_len_h // 2, w_minus - region_len_w // 2)
                region_plus_plus = (h_plus - region_len_h // 2, w_plus - region_len_w // 2)

                # compute factors for bi-linear interpolation between transforms
                a = (j - w_minus) / (w_plus - w_minus)
                b = (i - h_minus) / (h_plus - h_minus)

                # compute level of pixel (i, j) in the output image
                equalized_img[i, j] = np.round(
                        (1 - a) * (1 - b) * region_to_eq_transform[region_minus_minus][img_array[i, j]] +
                        (1 - a) * b * region_to_eq_transform[region_plus_minus][img_array[i, j]] +
                        a * (1 - b) * region_to_eq_transform[region_minus_plus][img_array[i, j]] +
                        a * b * region_to_eq_transform[region_plus_plus][img_array[i, j]]
                )

    return equalized_img


def perform_no_interpolation_ahe(img_array: np.ndarray, region_len_h: int, region_len_w: int):
    """ Returns the adaptive histogram equalization transform of input image,
        using contextual regions with height <region_len_h> and width <region_len_w>,
        without using bi-linear interpolation to compute values of each pixel in the
        output image

        :param img_array: 8-bit grayscale input image
        :type img_array: numpy.ndarray(dtype= numpy.uint8)
        :param region_len_h: height (in number of pixels) of each contextual
            region of image
        :type region_len_h: int
        :param region_len_w: width (in number of pixels) of each contextual
            region of image
        :type region_len_w: int
        :returns: 8-bit grayscale output image of transform
        :rtype: numpy.ndarray(dtype= numpy.uint8)
    """

    # find input image dimensions
    m = img_array.shape[0]
    n = img_array.shape[1]

    # initialize output image as a numpy ndarray
    equalized_img = np.zeros((m, n))

    # compute histogram equalization transform for each contextual region
    # of input image
    region_to_eq_transform = calculate_eq_transformations_of_regions(img_array, region_len_h, region_len_w)

    for i in range(m):
        for j in range(n):
            # find the contextual region each pixel resides in
            region = ((i // region_len_h) * region_len_h, (j // region_len_w) * region_len_w)

            # find the equalization transform of this contextual region
            eq_transform = region_to_eq_transform[region]

            # perform the above transform on the pixel
            equalized_img[i, j] = eq_transform[img_array[i, j]]

    return equalized_img
