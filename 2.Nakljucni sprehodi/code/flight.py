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


class distribution:
    def __init__(self, mi):
        self.mi = mi

    def support(self):
        return (1e-6, 1e200)

    def pdf(self, x):
        return x**(-self.mi)


def sim_flight(mi):
    dist = distribution(mi)
    ro = NumericalInversePolynomial(dist)

    N = 100000
    M = 1000

    all_steps = ro.rvs(size=(M, N))
    all_norms = np.cumsum(all_steps, axis=1)

    mads = stats.median_abs_deviation(all_norms, axis=0)
    lnmads2 = 2*np.log(mads)[100:]
    lnN = np.log(np.arange(1, N + 1))[100:]

    line = curve_fit(lambda x, a, b: a * x + b, lnN, lnmads2)
    gamma = line[0][0]

    plt.scatter(lnN, lnmads2, s=1, label='simulacija')
    plt.plot(lnN, line[0][0] * lnN + line[0][1], '--', c='gray', label='Fit')
    plt.xlabel(r'$\ln(N)$')
    plt.ylabel(r'$2\ln(\mathrm{MAD}^2)$')
    plt.title(
        r'Odvisnost $\ln(\mathrm{MAD}^2)$ od $\ln(N)$ pri $\mu = $' + str(mi))
    plt.legend()
    plt.show()

    return gamma


sim_flight(1.5)

start_time = time.time()
mis = np.arange(1.2, 4, 0.2)
gammas = np.array([[sim_flight(mi)**-1 for _ in range(10)] for mi in mis])
gammas_avg = np.mean(gammas, axis=1)
end_time = time.time()

print("Time elapsed: ", end_time - start_time, 's')


def theory(x):
    return np.where(x < 3, (1 / 2) * x - 1/2, 1)


plt.errorbar(mis, gammas_avg, yerr=np.std(gammas, axis=1), fmt='o',
             color='blue', ecolor='gray', elinewidth=1, capsize=3, capthick=1, label='simulacija')
plt.plot(mis, theory(mis), '--', c='gray', label='teorija')
plt.xlabel(r'$\mu$')
plt.ylabel(r'$\gamma^{-1}$')
plt.title(r'Odvisnost $\gamma^{-1}(\mu)$ za polete')
plt.legend()
plt.show()

# Time elapsed:  1018.4987487792969 s
