import numpy as np
from scipy.stats.sampling import NumericalInversePolynomial
from matplotlib import pyplot as plt
from scipy.integrate import quad


plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Computer Modern Serif",
    "font.size": 14
})


class MyDist:
    def __init__(self, a):
        self.a = a

    def support(self):
        return (2, 100)

    def pdf(self, x):
        return x**(-self.a)


dist = MyDist(1.5)
gen = NumericalInversePolynomial(dist)

# compute the missing normalizing constant to plot the pdf
const_pdf = quad(dist.pdf, *dist.support())[0]

r = gen.rvs(size=50000)
x = np.linspace(r.min(), r.max(), 500)

# show histogram together with the pdf
plt.plot(x, dist.pdf(x) / const_pdf, label='Verjetnostna porazdelitev')
plt.hist(r, density=True, bins=100, label='Histogram')
plt.xlabel('x')
plt.ylabel(r'$\rho(x)$')
plt.title('Verjetnostna porazdelitev in njen histogram')
plt.legend()
plt.show()
