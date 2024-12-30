import numpy as np
import matplotlib.pyplot as plt
from diffeq import rk4
from scipy.linalg import eigh
from scipy.optimize import newton, bisect, fsolve
import time
plt.style.use('customStyle.mplstyle')
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

times = {}

# Solving Schrödinger equation in finite or infinite potential well
matrix = np.array([[0, 1], [1, 0]])


def f(y, t, lam):
    return np.array([y[1], -lam * y[0]])


def time_df(N):
    x = np.linspace(0, 1, N)

    def F(E, *args):
        y0 = np.array(args)
        sol = rk4(f, y0, x, E)
        return sol[-1, 0]
    # Rešitve razdelimo na sode in lihe
    Es = []

    y0 = np.array([1, 0])
    E0 = 0

    time1 = time.time()
    while len(Es) < 5:
        E = newton(F, E0, args=(y0), maxiter=1000)
        E0 = E * 2
        if E in Es:
            break
        Es.append(E)

        sol = rk4(f, y0, x, Es[-1])
        parity = y0.dot([1, -1])
        sol = np.vstack((parity * sol[::-1], sol))

        y0 = matrix.dot(y0)
    time2 = time.time()

    E = 2*np.array(Es)  # rescale to units
    time2 = time.time()
    # Expected energy levels
    expected = np.array([(n+1)**2 * np.pi**2 / 2 for n in range(5)])
    err_E = np.abs(E[:6] - expected)

    return time2 - time1, err_E


M = 10
samples = range(100, 3001, 100)
err_Es = {}

for N in samples:
    times[N] = []
    err_Es[N] = []

for i in range(M):
    for N in samples:
        print(i+1, N)
        dt, err_E = time_df(N)
        times[N].append(dt)
        err_Es[N].append(err_E)

t = []
err_t = []
combined_err = []
combined_err2 = []
# print(times)
for N in samples:
    t.append(np.mean(times[N]))
    combined_err.append(np.mean(err_Es[N]))
    err_t.append(np.std(times[N]) / np.sqrt(M))
    combined_err2.append(np.std(err_Es[N]) / np.sqrt(M))


plt.plot(samples, t, c=colors[0])
plt.fill_between(samples, np.array(t) - np.array(err_t),
                 np.array(t) + np.array(err_t), alpha=0.3, color=colors[0])
plt.xlabel('Število točk $N$')
plt.ylabel('Čas reševanja [s]')
plt.tight_layout()
plt.savefig('strelska_time.pdf')
plt.show()

plt.plot(samples, combined_err, c=colors[0])
plt.fill_between(samples, np.array(combined_err) - np.array(combined_err2),
                 np.array(combined_err
                          ) + np.array(combined_err2), alpha=0.3, color=colors[0])
plt.xlabel('Število točk $N$')
plt.ylabel(r'Napaka izračuna energij $<\delta E>$')
plt.tight_layout()
plt.yscale('log')
plt.savefig('strelska_err.pdf')
plt.show()

combined_err2 = np.array(combined_err2)
combined_err = np.array(combined_err)

plt.fill_between(t, combined_err - combined_err2,
                 combined_err + combined_err2, alpha=0.3, color=colors[0])
plt.plot(t, combined_err, c=colors[0])
plt.xlabel('Čas reševanja [s]')
plt.ylabel(r'Napaka izračuna energij $<\delta E>$')
plt.tight_layout()
plt.yscale('log')
plt.savefig('strelska_err_time.pdf')
plt.show()
