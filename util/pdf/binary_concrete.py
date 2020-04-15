import numpy as np
from scipy import special as sp
from util.pdf import logistic


def density(temperature, alpha, x):
    """
    Binary concrete probability density function

    :param temperature: temperature of the concrete density function (in (0, inf))
    :param alpha: location (in (0, inf))
    :param x: binary concrete random variables (in (0, 1)). Bernoulli random variables when the
              temperature converges to 0.
    :return: the (binary concrete) probability of x given the temperature and location alpha
    """
    return temperature * alpha * np.power(x, - temperature - 1) * np.power(1 - x, - temperature - 1) / \
           np.square(alpha * np.power(x, (- temperature)) + np.power((1 - x), (- temperature)))


def log_density(temperature, log_alpha, x):
    """
    Binary concrete log-probability

    :param temperature: temperature of the concrete density function (in (0, inf))
    :param log_alpha: log of the location of the distribution
    :param x: binary concrete random variables (in (0, 1))
    :return: the (binary concrete) log-probability of x given the temperature and location alpha
    """
    return np.log(temperature) + log_alpha - (temperature + 1) * np.log(x) + (temperature - 1) * np.log(1 - x) - \
           2 * np.log1p(np.exp(log_alpha - temperature * (np.log(x) - np.log(1 - x))))


def log_logistic_density(temperature, log_alpha, x):
    """
    Binary concrete log-probability of a logistic random variable x (i.e., before applying the sigmoid)

    :param temperature: temperature of the concrete density function (in (0, inf))
    :param log_alpha: log of the location of the distribution
    :param x: binary concrete random variables (in (0, 1))
    :return: the (binary concrete) log-probability of x given the temperature and location alpha
    """
    return np.log(temperature) - temperature * x + log_alpha - 2 * np.log1p(np.exp(- temperature * x + log_alpha))


def log_diff(temperature, log_alpha_0, log_alpha_1, x):
    """
    Difference between two binary concrete log-probability with same temperature but different locations

    :param temperature: temperature of both concrete density functions (in (0, inf))
    :param log_alpha_0: log of the location of the first probability density function
    :param log_alpha_1: log of the location of the (subtracted) second density function
    :param x: binary concrete random variables (in (0, 1))
    :return: compute log p_0(x) - log p_1(x) where p_i is a binary concrete density function with location a_i
    """
    return log_alpha_0 - log_alpha_1 + 2 * \
           (np.log1p(np.exp(log_alpha_1 - temperature * (np.log(x) - np.log(1 - x)))) -
            np.log1p(np.exp(log_alpha_0 - temperature * (np.log(x) - np.log(1 - x)))))


def sample(temperature, log_alpha, logistic_noise):
    """
    Sample a binary concrete random variable

    :param temperature: temperature of the binary concrete density function (in (0, inf))
    :param log_alpha: log of the location of the binary concrete density function
    :param logistic_noise (sampled from log(U) - log(1 - U), where U is a uniform random variable in (0, 1))
    :return: a sampled binary concrete random variable from the corresponding density function with temperature and
             location alpha
    """
    return sp.expit(logistic.sample(temperature, log_alpha, logistic_noise))
