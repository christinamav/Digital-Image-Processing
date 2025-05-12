import numpy as np
import inverse_filtering
import wiener_filtering
from scipy.ndimage import convolve
import matplotlib.pyplot as plt


def calculate_k(x_inv0: np.ndarray, y: np.ndarray, h: np.ndarray, power_x: float, noise_var: float):
    """
    Returns the optimal value of the k parameter of the Wiener filter that
    minimizes the Mean Squared Error of the estimation of the original image x.
    The estimation is performed based on the distorted/ noisy image:
    y = h*x + v
    where h is the distortion filter and v is the noise signal.

    :param x_inv0: output of inverse filtering on the distorted noiseless image
    :param y: output image of the distortion/ noise model
    :param h: distortion filter applied on original image x
    :param power_x: power (variance) of the original image x
    :param noise_var: variance of noise signal v
    :return: optimal value of parameter k (float) of the Wiener filter
    """

    # the mse is the mean squared error between the original image x and
    # the estimated image x_hat
    # It is observed that the estimated image is shifted compared to the
    # original image x. To account for this, instead of considering the
    # original image in the mse calculation, we consider the image x_inv0,
    # which is the output of the inverse filtering performed on the noiseless
    # distorted image y0.

    # the value of parameter k can be approximated by the Signal-to-Noise-Ratio
    # of the original image to the noise signal

    # calculate SNR
    k0 = power_x / noise_var
    print("Signal-to-Noise-Ratio: ", k0)

    # generate values around the SNR
    # number of values to generate
    num = 1000
    # the range of k values is configured by the following
    # parameters a, b (they can be reconfigured by the user to provide
    # desirable results)
    a = 10
    b = 4
    k_values = np.linspace(k0/a, b*k0, num)

    # initialize mse
    mse = np.zeros((num, ))

    for i in range(num):
        k = k_values[i]
        # apply Wiener filter on the distorted/ noisy image y
        x_hat = wiener_filtering.my_wiener_filter(y, h, k)

        # compute the mse for every value of k
        mse[i] = np.mean((x_inv0 - x_hat)**2)

    # after having computed the MSE for a set of values of parameter k,
    # we need to find which value of k minimizes it

    # index of the minimum value of the mse
    idx_opt = np.argmin(mse)

    # optimal value of parameter k
    k_opt = k_values[idx_opt]

    # plot the mse in relation to parameter k
    plt.figure()
    plt.plot(k_values, mse)
    plt.plot(k_opt, mse[idx_opt], marker='o', markersize=5, markeredgecolor="red", markerfacecolor="red")
    plt.text(k_opt, mse[idx_opt], '({:.2f}, {:.4f})'.format(k_opt, mse[idx_opt]))
    plt.title("Mean Squared Error Line")
    plt.xlabel("parameter k")
    plt.ylabel("Mean Squared Error")
    
    return k_opt
