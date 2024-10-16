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
        return (1e-5, 1e200)

    def pdf(self, x):
        return x**(-self.mi)


def sim_flight(mi):
    dist = distribution(mi)
    ro = NumericalInversePolynomial(dist)

    N = 100000
    M = 100

    all_steps = ro.rvs(size=(M, N))
    all_norms = np.cumsum(all_steps, axis=1)
    all_times = all_norms

    times = np.median(all_times, axis=0)
    mads = stats.median_abs_deviation(all_norms, axis=0)
    lnmads2 = 2*np.log(mads)[200:]
    lnt = np.log(times)[200:]
    lnN = lnt

    line = curve_fit(lambda x, a, b: a * x + b, lnN, lnmads2)
    gamma = line[0][0]

    return gamma


start_time = time.time()
mis = np.arange(1.2, 4, 0.2)
gammas = np.array([[sim_flight(mi) for _ in range(10)] for mi in mis])
gammas_avg = np.mean(gammas, axis=1)
end_time = time.time()

print("Time elapsed: ", end_time - start_time, 's')


def theory(x):
    return [(-1 * i + 4) if 2 <= i <= 3 else 2 if i < 2 else 1 for i in x]


plt.errorbar(mis, gammas_avg, yerr=np.std(gammas, axis=1), fmt='o',
             color='blue', ecolor='gray', elinewidth=1, capsize=3, capthick=1, label='simulacija')
plt.plot(mis, theory(mis), '--', c='gray', label='teorija')
plt.xlabel(r'$\mu$')
plt.ylabel(r'$\gamma$')
plt.title(r'Odvisnost $\gamma(\mu)$ za sprehode')
plt.legend()
plt.show()

# Time elapsed:   s
