import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
import time
plt.style.use('customStyle.mplstyle')
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

# Solving Schrödinger equation in finite or infinite potential well

# Constants
a = 4
N = 2000
h = a/N
A = np.diag(np.ones(N-1), -1) + np.diag(np.ones(N-1), 1) + \
    np.diag((-2)*np.ones(N))

x = np.linspace(-a/2, a/2, N)

# Rešujem HY = EY, kjer so E lastne vrednosti operatorja H

# Infinite well
V = np.zeros(N) * np.eye(N)

# Finite well
V0 = 1e2
# x = np.linspace(-a/2 * 1.25, a/2 * 1.25, N)
V = np.r_[V0 * np.ones(N//4), np.zeros(N//2),
          V0 * np.ones(N//4)] * np.eye(N)

H = -0.5 * A / h**2 + V

time1 = time.time()
E, states = eigh(H)
time2 = time.time()
print(f"Time: {time2-time1:.4f} s")

# Rescale energies to my units
E = 2*E
for n in range(5):
    psi = states[:, n]
    psi /= np.sqrt(np.trapezoid(psi**2, x))
    psi *= 4.5  # Scale for better visualization
    plt.plot(x, E[n] + psi, label=f'n={n+1}, E={E[n]:.2f}')
    expected = (n+1)**2 * np.pi**2 / 4  # Expected energy levels
    print(f"State {n+1}: E = {E[n]:.2f}, Expected = {expected}")

    plt.hlines(E[n], -2, 2, linestyles='dashed', colors='gray', alpha=0.5)
    plt.annotate(
        f'$E_{n}$', (1.09, E[n]), textcoords="offset points", xytext=(0, 10), ha='center', color=colors[n])
    plt.fill_between(x, E[n] + psi, E[n], alpha=0.3)

plt.hlines(0, -a/2, a/2, colors='black')
plt.vlines([-a/2, a/2], -1, V0, colors='black')
plt.hlines(V0, -2, -a/2, colors='black')
plt.hlines(V0, a/2, 2, colors='black')
plt.annotate(f'$V_0$', (1.09, V0), textcoords="offset points",
             xytext=(0, 10), ha='center', color='black')
plt.xlabel('Pozicija $x$')
plt.ylabel(r'Valovna funkcija $\Psi + E_{\rm n}$')
plt.title('Prvih 5 lastnih stanj končne potencialne jame')
plt.xlim(-1.2, 1.2)
plt.ylim(-1.15, V0 * 1.1)
plt.tight_layout()
# plt.savefig(r'8RobniProblemLastnihVrednosti/fd_fwell.pdf')
plt.show()

# fd_infwell: \Delta t = 0.1423s (from 0.11s to 0.19s)
# fd_fwell: \Delta t = 0.1567s (from 0.13s to 2s)
