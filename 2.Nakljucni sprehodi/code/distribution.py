# Source to sampling with custom distribution https://stackoverflow.com/questions/3510475/generate-random-numbers-according-to-distributions (15.10.2024)
import numpy as np
from scipy.integrate import quad
from matplotlib import pyplot as plt
from scipy.stats.sampling import NumericalInversePolynomial


class distribution:
    def __init__(self, mi):
        self.mi = mi

    def support(self):
        return (0.01, 10)

    def pdf(self, x):
        return x**(-self.mi)


if __name__ == '__main__':
    for i in np.linspace(1, 4, 10):

        dist = distribution(i)
        gen = NumericalInversePolynomial(dist)

        # compute the missing normalizing constant to plot the pdf
        const_pdf = quad(dist.pdf, *dist.support())[0]

        r = gen.rvs(size=int(1e7))
        x = np.linspace(r.min(), r.max(), 500)

        # show histogram together with the pdf
        plt.plot(x, dist.pdf(x) / const_pdf)
        plt.hist(r, density=True, bins=100)
        plt.yscale('log')
    plt.show()
