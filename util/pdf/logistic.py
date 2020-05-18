import numpy as np
import tensorflow_probability


def density(temperature, mu, square=np.square, exp=np.exp):
    """
    Logistic probability density function

    :param temperature: scale of the logistic function (in (0, inf))
    :param mu: location (in the binary logistic case, mu is log_alpha)
    :param square: square function
    :param exp: exp function
    :return: the logistic density function of scale temperature and location mu
    """
    # :param x: logistic random variable (in (0, 1)).
    return lambda x: temperature * exp(- temperature * x + mu) / square(1 + exp(- temperature * x + mu))


def log_density(temperature, mu, log=np.log, exp=np.exp, log1p=np.log1p, tfp=True):
    """
    Log-Logistic probability density function

    :param temperature: scale of the logistic function (in (0, inf))
    :param mu: location (in the binary logistic case, mu is log_alpha)
    :param log: log function
    :param exp: exp function
    :param log1p: log1p function
    :param tfp: use Tensorflow probability, less sensitive to overflow
    :return: the log-logistic density function of scale temperature and location mu
    """
    # :param x: logistic random variable (in (0, 1))
    if tfp:
        return lambda x: tensorflow_probability.distributions.Logistic(
            scale=1/temperature, loc=mu/temperature
        ).log_prob(x)
    else:
        return lambda x: log(temperature) + (- temperature * x + mu) - 2 * log1p(exp(mu - temperature * x))


def log_diff(temperature, mu_0, mu_1, exp=np.exp, log1p=np.log1p):
    """
    Difference between two logistic log-probability with same scale (temperature) but different locations

    :param temperature: scale of both logistic density functions (in (0, inf))
    :param mu_0: location of the first probability density function
    :param mu_1: location of the (subtracted) second density function
    :param exp: exp function
    :param log1p: log1p function
    :return: the log-diff density log p_0(x) - log p_1(x)
             such that p_i is a logistic density function with scale temperature and location mu_i
    """
    # :param x: logistic random variable (in (0, 1))
    return lambda x: mu_0 - mu_1 + 2 * (log1p(exp(mu_1 - temperature * x)) - log1p(exp(mu_0 - temperature * x)))


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
