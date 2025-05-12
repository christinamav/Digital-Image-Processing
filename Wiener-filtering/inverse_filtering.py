import numpy as np


def my_inverse_filter(y: np.ndarray, h: np.ndarray):
    """
    Returns an estimation of an original 2-dimensional signal x, that has been distorted
    based on the model:
    y = h*x + v,
    where v is a 2-dimensional noise signal and h is the impulse response of the distortion
    system.
    The estimation is performed using Inverse Filtering.

    :param y: 2-dimensional array representing the distorted grayscale image in the spatial
            field
    :param h: 2-dimensional array representing the impulse response of the
            distortion system
    :return: 2-dimensional image of shape identical to the input image y, output of the inverse
    filter in the spatial field
    """

    # shape of input image y
    m, n = y.shape
    # shape of impulse response h
    l, p = h.shape

    # perform zero-padding on the impulse response h
    h_padded = np.zeros((m, n))
    h_padded[:l, :p] = h

    # perform Discrete Fourier Transform on the padded versions of the input image y and
    # the impulse response h, using the Fast Fourier Transform
    y_dft = np.fft.fft2(y)
    h_dft = np.fft.fft2(h_padded)

    # compute the inverse of the distortion system transfer function (element-wise inverse)
    # handle the division by zero or near-zero values for numerical stability
    epsilon = 1e-10
    h_dft_inv = 1/(h_dft + epsilon)

    # output of inverse filtering in the spatial frequency field
    x_hat_dft = np.multiply(h_dft_inv, y_dft)

    # compute Inverse Discrete Fourier Transform on the output of the filter
    x_hat = np.fft.ifft2(x_hat_dft)

    # extract only the real part of the output image
    x_hat = np.real(x_hat)

    return x_hat
