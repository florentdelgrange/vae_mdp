import numpy as np
from scipy import special as sp
from util.pdf import logistic


def density(temperature, alpha, power=np.power, square=np.square):
    """
    Binary concrete probability density function

    :param temperature: scale of the concrete density function (in (0, inf))
    :param alpha: location (in (0, inf))
    :param power: power function
    :param square: square function
    :return: the binary concrete density function of scale temperature and location alpha
    """
    # :param x: binary concrete random variable (in (0, 1)).
    #           Bernoulli random variable when the temperature converges to 0.
    return lambda x: temperature * alpha * power(x, - temperature - 1) * power(1 - x, - temperature - 1) / \
                     square(alpha * power(x, (- temperature)) + power((1 - x), (- temperature)))


def log_density(temperature, log_alpha, log=np.log, exp=np.exp, log1p=np.log1p):
    """
    Binary concrete log-density function

    :param temperature: scale of the concrete density function (in (0, inf))
    :param log_alpha: log of the location of the distribution
    :param log: log function
    :param exp: exp function
    :param log1p: log1p function
    :return: the binary concrete log-density function of scale temperature and location alpha
    """
    # :param x: binary concrete random variable (in (0, 1))
    return lambda x: log(temperature) + log_alpha - (temperature + 1) * log(x) + (temperature - 1) * log(1 - x) - \
                     2 * log1p(exp(log_alpha - temperature * (log(x) - log(1 - x))))


def log_logistic_density(temperature, log_alpha, log=np.log, exp=np.exp, log1p=np.log1p):
    """
    Binary concrete logistic log-density function. Gives the probability of a logistic random variable x, i.e., a binary
    concrete random variable before applying the sigmoid.

    :param temperature: scale of the concrete density function (in (0, inf))
    :param log_alpha: log of the location of the distribution
    :param log: log function
    :param exp: exp function
    :param log1p: log1p function
    :return: the binary concrete log-density function of scale temperature and location alpha
    """
    # :param x: logistic random variable (in (0, 1))
    return lambda x: log(temperature) - temperature * x + log_alpha - 2 * log1p(exp(- temperature * x + log_alpha))


def log_diff(temperature, log_alpha_0, log_alpha_1, log=np.log, exp=np.exp, log1p=np.log1p):
    """
    Difference between two binary concrete log-probability with same temperature but different locations

    :param temperature: scale of both concrete density functions (in (0, inf))
    :param log_alpha_0: log of the location of the first probability density function
    :param log_alpha_1: log of the location of the (subtracted) second density function
    :param log: log function
    :param exp: exp function
    :param log1p: log1p function
    :return: the log-diff density log p_0(x) - log p_1(x)
             such that p_i is a binary concrete density function with scale temperature and location mu_i
    """
    # :param x: binary concrete random variable (in (0, 1))
    return lambda x: log_alpha_0 - log_alpha_1 + 2 * \
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
