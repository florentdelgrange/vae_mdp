import unittest
from util import pdf
import numpy as np


class PdfTestCase(unittest.TestCase):
    def test_log_bin(self):
        # Uniform random variables
        U = np.array([np.random.uniform(1e-12) for _ in range(100)])
        temperatures = [1/2, 2/3]
        # binary concrete random variables
        for temperature in temperatures:
            for alpha in [np.random.uniform(1e-12, 1e5) for _ in range(100)]:
                # logistic noises
                L = np.log(U) - np.log(np.ones(U.size) - U)
                # binary concrete random variables
                X = pdf.sample_bin_concrete(temperature, alpha, L)
                alpha_1 = np.random.uniform(1e-12, 1e5)
                self.assertTrue(np.allclose(
                    np.log(pdf.bin_concrete(temperature, alpha, X)) - np.log(pdf.bin_concrete(temperature, alpha_1, X)),
                    pdf.log_diff_bin_concrete(temperature, alpha, alpha_1, X)))


if __name__ == '__main__':
    unittest.main()
