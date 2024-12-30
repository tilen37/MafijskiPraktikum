import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
import time
plt.style.use('customStyle.mplstyle')
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

# Solving Schrödinger equation in finite or infinite potential well

# Constants
a = 2
N = 1000
V0 = 1e2

times = {}


def time_df(N):
    h = a/N
    A = np.diag(np.ones(N-1), -1) + np.diag(np.ones(N-1), 1) + \
        np.diag((-2)*np.ones(N))

    # Rešujem HY = EY, kjer so E lastne vrednosti operatorja H

    # Infinite well
    x = np.linspace(-a/2, a/2, N)
    V = np.zeros(N) * np.eye(N)

    # Finite well
    # x = np.linspace(-a/2 * 1.25, a/2 * 1.25, N)
    # V = np.r_[V0 * np.ones(N//10), np.zeros(N//10*8),
    #           V0 * np.ones(N//10)] * np.eye(N)

    H = -0.5 * A / h**2 + V

    time1 = time.time()
    E, states = eigh(H)
    E = 2*E  # rescale to units
    time2 = time.time()
    # Expected energy levels
    expected = [(n+1)**2 * np.pi**2 / 4 for n in range(6)]
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


# plt.errorbar(samples, t, yerr=err_t, fmt='o', capsize=5, c=colors[0])
plt.plot(samples, t, c=colors[0])
plt.fill_between(samples, np.array(t) - np.array(err_t),
                 np.array(t) + np.array(err_t), alpha=0.3, color=colors[0])
plt.xlabel('Število točk $N$')
plt.ylabel('Čas reševanja [s]')
plt.tight_layout()
plt.savefig('diferencna_time.pdf')
plt.show()

plt.plot(samples, combined_err, c=colors[0])
plt.fill_between(samples, np.array(combined_err) - np.array(combined_err2),
                 np.array(combined_err
                          ) + np.array(combined_err2), alpha=0.3, color=colors[0])
plt.xlabel('Število točk $N$')
plt.ylabel(r'Napaka izračuna energij $<\delta E>$')
plt.yscale('log')
plt.tight_layout()
plt.savefig('diferencna_err.pdf')
plt.show()

combined_err2 = np.array(combined_err2)
combined_err = np.array(combined_err)

plt.fill_between(t, combined_err - combined_err2,
                 combined_err + combined_err2, alpha=0.3, color=colors[0])
plt.plot(t, combined_err, c=colors[0])
plt.xlabel('Čas reševanja [s]')
plt.ylabel(r'Napaka izračuna energij $<\delta E>$')
plt.yscale('log')
plt.tight_layout()
plt.savefig('diferencna_err_time.pdf')
plt.show()
