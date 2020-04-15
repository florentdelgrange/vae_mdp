import numpy as np


def density(temperature, mu, x):
    """
    Logistic probability density function

    :param temperature: temperature of the logistic function (in (0, inf))
    :param mu: location (in the binary logistic case, mu is log_alpha)
    :param x: logistic random variables (in (0, 1)).
    :return: the logistic probability of x given the temperature and location mu
    """
    return temperature * np.exp(- temperature * x + mu) / np.square(1 + np.exp(- temperature * x + mu))


def log_density(temperature, mu, x):
    """
    Log-Logistic probability density function

    :param temperature: temperature of the logistic function (in (0, inf))
    :param mu: location (in the binary logistic case, mu is log_alpha)
    :param x: logistic random variables (in (0, 1)).
    :return: the log-logistic probability of x given the temperature and location mu
    """
    return np.log(temperature) + (- temperature * x + mu) - 2 * np.log1p(np.exp(mu - temperature * x))


def log_diff(temperature, mu_0, mu_1, x):
    """
    Difference between two logistic log-probability with same temperature but different locations

    :param temperature: temperature of both logistic density functions (in (0, inf))
    :param mu_0: location of the first probability density function
    :param mu_1: location of the (subtracted) second density function
    :param x: logistic random variables (in (0, 1))
    :return: compute log p_0(x) - log p_1(x) where p_i is a logistic density function with location mu_i
    """
    return mu_0 - mu_1 + 2 * (np.log1p(np.exp(mu_1 - temperature * x)) - np.log1p(np.exp(mu_0 - temperature * x)))


def sample(temperature, mu, logistic_noise):
    """
    Sample a logistic random variable

    :param temperature: temperature of the logistic density function (in (0, inf))
    :param mu: log of the location of the logistic density function
    :param logistic_noise (sampled from log(U) - log(1 - U), where U is a uniform random variable in (0, 1))
    :return: a sampled logistic random variable from the corresponding density function with temperature and
             location alpha
    """
    return (logistic_noise + mu) / temperature
