# Source to sampling with custom distribution https://stackoverflow.com/questions/3510475/generate-random-numbers-according-to-distributions (15.10.2024)
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats.sampling import NumericalInversePolynomial
from scipy import stats
from scipy.optimize import curve_fit
import time

np.random.seed(1234)

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Computer Modern Serif",
    "font.size": 14
})

np.random.seed(1234)


class distribution:
    def __init__(self, mi):
        self.mi = mi

    def support(self):
        return (1e-5, 1e200)

    def pdf(self, x):
        return x**(-self.mi)


dist = distribution(1.5)
ro = NumericalInversePolynomial(dist)

N = 10000

X, Y = [0], [0]

for _ in range(N):
    l = ro.rvs()
    phi = np.random.uniform(0, 2 * np.pi)

    X.append(l * np.cos(phi))
    Y.append(l * np.sin(phi))

X = np.cumsum(X)
Y = np.cumsum(Y)

plt.plot(X, Y, 'o-')
plt.xlabel('x')
plt.ylabel('y')
plt.title(f'Naključni sprehod po {N} korakih')
plt.show()
