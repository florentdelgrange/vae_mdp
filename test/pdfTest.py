import unittest
from util.pdf import binary_concrete, logistic
import numpy as np
import scipy.special as sp


class PdfTestCase(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(PdfTestCase, self).__init__(*args, **kwargs)
        # Uniform random variables
        U = np.array([np.random.uniform(1e-12) for _ in range(100)])
        # Logistic noise
        self.L = np.log(U) - np.log(np.ones(U.size) - U)
        self.temperatures = [1 / 2, 2 / 3]

    def test_log_bin_diff(self):
        # binary concrete random variables
        for temperature in self.temperatures:
            for alpha in [np.random.uniform(1e-12, 1e5) for _ in range(100)]:
                # binary concrete random variables
                X = binary_concrete.sample(temperature, np.log(alpha), self.L)
                alpha_1 = np.random.uniform(1e-12, 1e5)
                for x in X:
                    self.assertAlmostEqual(
                        np.log(binary_concrete.density(temperature, alpha)(x)) -
                        np.log(binary_concrete.density(temperature, alpha_1)(x)),
                        binary_concrete.log_diff(temperature, np.log(alpha), np.log(alpha_1))(x))

    def test_log_logistic_diff(self):
        for temperature in self.temperatures:
            for log_alpha in [np.log(np.random.uniform(1e-12, 1e5)) for _ in range(100)]:
                # binary concrete random variables
                X = logistic.sample(temperature, log_alpha, self.L)
                log_alpha_1 = np.log(np.random.uniform(1e-12, 1e5))
                for x in X:
                    self.assertAlmostEqual(
                        np.log(logistic.density(temperature, log_alpha)(x)) -
                        np.log(logistic.density(temperature, log_alpha_1)(x)),
                        logistic.log_diff(temperature, log_alpha, log_alpha_1)(x))

    def test_log_logistic(self):
        for log_alpha in [np.log(np.random.uniform(1e-12, 1e5)) for _ in range(100)]:
            # binary concrete random variables
            X = logistic.sample(self.temperatures[0], log_alpha, self.L)
            log_alpha_1 = np.log(np.random.uniform(1e-12, 1e5))
            for x in X:
                """
                The difference between two log binary probabilities is prone to underflow, so we set 2 places since a
                large error is possible.
                """
                self.assertAlmostEqual(
                    binary_concrete.log_density(self.temperatures[0], log_alpha)(sp.expit(x)) -
                    binary_concrete.log_density(self.temperatures[1], log_alpha_1)(sp.expit(x)),
                    binary_concrete.log_logistic_density(self.temperatures[0], log_alpha)(x) -
                    logistic.log_density(self.temperatures[1], log_alpha_1,)(x), 2)


if __name__ == '__main__':
    unittest.main()
