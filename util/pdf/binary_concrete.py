import numpy as np
from scipy import special as sp
from util.pdf import logistic


def density(temperature, alpha, x, power=np.power, square=np.square):
    """
    Binary concrete probability density function

    :param temperature: temperature of the concrete density function (in (0, inf))
    :param alpha: location (in (0, inf))
    :param x: binary concrete random variables (in (0, 1)).
              Bernoulli random variables when the temperature converges to 0.
    :param power: power function
    :param square: square function
    :return: the (binary concrete) probability of x given the temperature and location alpha
    """
    return temperature * alpha * power(x, - temperature - 1) * power(1 - x, - temperature - 1) / \
           square(alpha * power(x, (- temperature)) + power((1 - x), (- temperature)))


def log_density(temperature, log_alpha, x, log=np.log, exp=np.exp, log1p=np.log1p):
    """
    Binary concrete log-probability

    :param temperature: temperature of the concrete density function (in (0, inf))
    :param log_alpha: log of the location of the distribution
    :param x: binary concrete random variables (in (0, 1))
    :param log: log function
    :param exp: exp function
    :param log1p: log1p function
    :return: the (binary concrete) log-probability of x given the temperature and location alpha
    """
    return log(temperature) + log_alpha - (temperature + 1) * log(x) + (temperature - 1) * log(1 - x) - \
           2 * log1p(exp(log_alpha - temperature * (log(x) - log(1 - x))))


def log_logistic_density(temperature, log_alpha, x, log=np.log, exp=np.exp, log1p=np.log1p):
    """
    Binary concrete log-probability of a logistic random variable x (i.e., before applying the sigmoid)

    :param temperature: temperature of the concrete density function (in (0, inf))
    :param log_alpha: log of the location of the distribution
    :param x: binary concrete random variables (in (0, 1))
    :param log: log function
    :param exp: exp function
    :param log1p: log1p function
    :return: the (binary concrete) log-probability of x given the temperature and location alpha
    """
    return log(temperature) - temperature * x + log_alpha - 2 * log1p(exp(- temperature * x + log_alpha))


def log_diff(temperature, log_alpha_0, log_alpha_1, x, log=np.log, exp=np.exp, log1p=np.log1p):
    """
    Difference between two binary concrete log-probability with same temperature but different locations

    :param temperature: temperature of both concrete density functions (in (0, inf))
    :param log_alpha_0: log of the location of the first probability density function
    :param log_alpha_1: log of the location of the (subtracted) second density function
    :param x: binary concrete random variables (in (0, 1))
    :param log: log function
    :param exp: exp function
    :param log1p: log1p function
    :return: compute log p_0(x) - log p_1(x) where p_i is a binary concrete density function with location a_i
    """
    return log_alpha_0 - log_alpha_1 + 2 * \
           (log1p(exp(log_alpha_1 - temperature * (log(x) - log(1 - x)))) -
            log1p(exp(log_alpha_0 - temperature * (log(x) - log(1 - x)))))


def sample(temperature, log_alpha, logistic_noise, sigmoid=sp.expit):
    """
    Sample a binary concrete random variable

    :param temperature: temperature of the binary concrete density function (in (0, inf))
    :param log_alpha: log of the location of the binary concrete density function
    :param logistic_noise (sampled from log(U) - log(1 - U), where U is a uniform random variable in (0, 1))
    :param sigmoid: sigmoid function
    :return: a sampled binary concrete random variable from the corresponding density function with temperature and
             location alpha
    """
    return sigmoid(logistic.sample(temperature, log_alpha, logistic_noise))
