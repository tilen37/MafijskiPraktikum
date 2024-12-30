import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from quspin.operators import hamiltonian
from quspin.basis import spinless_fermion_basis_1d
import time
plt.style.use('customStyle.mplstyle')
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

# Solving Schrödinger equation in finite or infinite potential well

# Constants
a = 2
N = L = 1000
V0 = 1e2

times = {}


def time_df(N):
    # Parameters
    h = 2.0/L  # Lattice spacing (well width = 2)
    J = - 1.0/(2*h**2)  # Hopping strength to match -∇²/2m

    # Define basis
    basis = spinless_fermion_basis_1d(L, Nf=1)

    def V(x):
        return 0 if -1 < x - 1 < 1 else J*1e5   # infinite well V0 = 1e5J
        return 0 if -.5 < x - 1 < .5 else J*3e-3  # finite well V0 = 3e-3J

    # Define hopping terms with correct scaling
    hop_left = [[J, i, i+1] for i in range(L-1)]    # left hopping
    hop_right = [[J, i+1, i] for i in range(L-1)]   # right hopping
    # on-site potential is 0 inside inf well
    diag = [[V(i*h), i] for i in range(L)]          # on-site potential
    static = [["+-", hop_left], ["+-", hop_right], ["n", diag]]

    # Create Hamiltonian
    H = hamiltonian(static, [], basis=basis, dtype=np.float64)

    # Solve and plot
    time1 = time.time()
    E, states = H.eigh()
    time2 = time.time()

    # Remove ground state
    states = states[:, 1:]
    E = E[1:]
    # Rescale energies to match expected values (probably from 1/2 in -∇²/2m term)
    E = 2 * E

    # Calculate error
    expected = np.array([np.pi**2 / 4 * i**2 for i in range(1, 6)])
    err_E = np.mean(np.abs(E[:5] - expected))

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
plt.savefig('quspin_time.pdf')
plt.show()

plt.plot(samples, combined_err, c=colors[0])
plt.fill_between(samples, np.array(combined_err) - np.array(combined_err2),
                 np.array(combined_err
                          ) + np.array(combined_err2), alpha=0.3, color=colors[0])
plt.xlabel('Število točk $N$')
plt.ylabel(r'Napaka izračuna energij $<\delta E>$')
plt.yscale('log')
plt.tight_layout()
plt.savefig('quspin_err.pdf')
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
plt.savefig('quspin_err_time.pdf')
plt.show()
