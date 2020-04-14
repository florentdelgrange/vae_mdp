# Probability density functions
import numpy as np
from scipy import special as sp


def bin_concrete(temperature, alpha, x):
    """
    Binary concrete probability density function

    :param temperature: temperature of the concrete density function (in (0, inf))
    :param alpha: location (in (0, inf))
    :param x: binary concrete random variables (data points) (in (0, 1)). Bernoulli random variables when the
              temperature converges to 0.
    :return: the (binary concrete) probability of x given the temperature and location alpha
    """
    return temperature * alpha * np.power(x, - temperature - 1) * np.power(1 - x, - temperature - 1) / \
           np.square(alpha * np.power(x, (- temperature)) + np.power((1 - x), (- temperature)))


def log_bin_concrete(temperature, alpha, x):
    """
    Binary concrete log-probability

    :param temperature: temperature of the concrete density function (in (0, inf))
    :param alpha: location (in (0, inf))
    :param x: binary concrete random variables (data points) (in (0, 1))
    :return: the (binary concrete) log-probability of x given the temperature and location alpha
    """
    return np.log(temperature) + np.log(alpha) - (temperature + 1) * (np.log(x) + np.log(1 - x)) + \
           2 * temperature * np.log(1 - x) - \
           2 * np.log1p(alpha * np.power(1 - x, temperature) / np.power(x, temperature))


def log_diff_bin_concrete(temperature, alpha_0, alpha_1, x):
    """
    Difference of two binary concrete log-probability with same temperature but different locations

    :param temperature: temperature of both concrete density functions (in (0, inf))
    :param alpha_0: location of the first probability density function (in (0, inf))
    :param alpha_1: location of the (substracted) second density function (in (0, inf))
    :param x: binary concrete random variables (data points) (in (0, 1))
    :return: compute log p_0(x) - log p_1(x) where p_i is a binary concrete density function with location a_i
    """
    heated_x = np.power(1 - x, temperature) / np.power(x, temperature)
    return np.log(alpha_0) - np.log(alpha_1) + 2 * np.log1p(alpha_1 * heated_x) - 2 * np.log1p(alpha_0 * heated_x)


def sample_bin_concrete(temperature, alpha, logistic_noise):
    """
    Sample a binary concrete random variable

    :param temperature: temperature of the concrete density function (in (0, inf))
    :param alpha: location of the concrete density function (in (0, inf))
    :param logistic_noise (sampled from log(U) - log(1 - U), where U is a uniform random variable in (0, 1))
    :return: a sampled binary concrete random variable from the corresponding density function with temperature and
             location alpha
    """
    return sp.expit(logistic_noise + np.log(alpha) / temperature)
