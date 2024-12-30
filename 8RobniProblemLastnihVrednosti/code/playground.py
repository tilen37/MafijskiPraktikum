from quspin.operators import hamiltonian
from quspin.basis import spinless_fermion_basis_1d
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import time
plt.style.use('customStyle.mplstyle')
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

# Parameters
L = 2000  # Number of lattice sites
a = 4/L  # Lattice spacing (well width = 2)
J = - 1.0/(2*a**2)  # Hopping strength to match -∇²/2m

# Define basis
basis = spinless_fermion_basis_1d(L, Nf=1)


def V(x):
    return 0 if -1 < x - 2 < 1 else 1e2  # finite well V0 = 1e2
    return 0 if -1 < x - 1 < 1 else J*1e2   # infinite well V0 = 1e5J


x = np.linspace(-1, 1, L)
x = np.linspace(-1.25, 1.25, L)

# Define hopping terms with correct scaling
hop_left = [[J, i, i+1] for i in range(L-1)]    # left hopping
hop_right = [[J, i+1, i] for i in range(L-1)]   # right hopping
# on-site potential is 0 inside inf well
diag = [[V(i*a), i] for i in range(L)]          # on-site potential
static = [["+-", hop_left], ["+-", hop_right], ["n", diag]]

# Create Hamiltonian
H = hamiltonian(static, [], basis=basis, dtype=np.float64)

# Solve and plot
time1 = time.time()
E, states = H.eigh()
time2 = time.time()
print(f"Time: {time2-time1:.2f} s")

# Remove ground state
print(E[:5])
# states = states[:, 1:]
# E -= E[0]
# E = E[1:]
# Rescale energies to match expected values (probably from 1/2 in -∇²/2m term)
E = 2 * E
E = E - E[0] + 2.038  # Set the first energy level to the expected value

print(f'Number of states: {len(E)}')
for n in range(5):
    psi = states[:, n]
    psi /= np.sqrt(np.trapezoid(psi**2, x))
    psi *= 3.5  # Scale for better visualization
    # We only plot every second point, to compensate for fermion doubling in tight-binding model
    psi = psi[::2]
    x1 = x[::2]
    plt.plot(x1, E[n] + psi, label=f'n={n+1}, E={E[n]:.2f}')
    expected = (n+1)**2 * np.pi**2 / 4  # Expected energy levels
    print(f"State {n+1}: E = {E[n]:.2f}, Expected = {expected}")

    plt.hlines(E[n], -2, 2, linestyles='dashed', colors='gray', alpha=0.5)
    plt.annotate(
        f'$E_{n}$', (1.075, E[n]), textcoords="offset points", xytext=(0, 10), ha='center', color=colors[n])
    plt.fill_between(x1, E[n] + psi, E[n], alpha=0.3)

plt.hlines(0, -1, 1, colors='black')
plt.vlines([-1, 1], -1, E[n] * 1.1, colors='black')
plt.xlabel('Pozicija $x$')
plt.ylabel('Valovna funkcija $\Psi$')
plt.title('Prvih 5 lastnih stanj neskončne potencialne jame')
plt.xlim(-1.15, 1.15)
plt.tight_layout()
plt.show()
