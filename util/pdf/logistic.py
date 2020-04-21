import numpy as np


def density(temperature, mu, x, square=np.square, exp=np.exp):
    """
    Logistic probability density function

    :param temperature: temperature of the logistic function (in (0, inf))
    :param mu: location (in the binary logistic case, mu is log_alpha)
    :param x: logistic random variables (in (0, 1)).
    :param square: square function
    :param exp: exp function
    :return: the logistic probability of x given the temperature and location mu
    """
    return temperature * exp(- temperature * x + mu) / square(1 + exp(- temperature * x + mu))


def log_density(temperature, mu, x, log=np.log, exp=np.exp, log1p=np.log1p):
    """
    Log-Logistic probability density function

    :param temperature: temperature of the logistic function (in (0, inf))
    :param mu: location (in the binary logistic case, mu is log_alpha)
    :param x: logistic random variables (in (0, 1)).
    :param log: log function
    :param exp: exp function
    :param log1p: log1p function
    :return: the log-logistic probability of x given the temperature and location mu
    """
    return log(temperature) + (- temperature * x + mu) - 2 * log1p(exp(mu - temperature * x))


def log_diff(temperature, mu_0, mu_1, x, log=np.log, exp=np.exp, log1p=np.log1p):
    """
    Difference between two logistic log-probability with same temperature but different locations

    :param temperature: temperature of both logistic density functions (in (0, inf))
    :param mu_0: location of the first probability density function
    :param mu_1: location of the (subtracted) second density function
    :param x: logistic random variables (in (0, 1))
    :param log: log function
    :param exp: exp function
    :param log1p: log1p function
    :return: compute log p_0(x) - log p_1(x) where p_i is a logistic density function with location mu_i
    """
    return mu_0 - mu_1 + 2 * (log1p(exp(mu_1 - temperature * x)) - log1p(exp(mu_0 - temperature * x)))


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
