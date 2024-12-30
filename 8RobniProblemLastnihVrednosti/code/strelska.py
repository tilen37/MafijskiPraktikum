import numpy as np
import matplotlib.pyplot as plt
# import tensorflow as tf
from diffeq import rk4
from scipy.optimize import newton, bisect, fsolve
import time
plt.style.use('customStyle.mplstyle')
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

# Solving Schrödinger equation in finite or infinite potential well
# a = 2

# RP:
# for infinite well
# psi(-1) = 0
# psi(1) = 0


def f(y, t, lam):
    return np.array([y[1], -lam * y[0]])


x = np.linspace(0, 1, 1000)
x1 = np.concatenate((-x[::-1], x))
E = 1


def F(E, *args):
    y0 = np.array(args)
    sol = rk4(f, y0, x, E)
    return sol[-1, 0]


# Rešitve razdelimo na sode in lihe
Es = []

matrix = np.array([[0, 1], [1, 0]])
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
    sol /= np.sqrt(np.trapezoid(sol[:, 0]**2, x1))
    sol *= 3  # Scale for better visualization
    plt.plot(x1, Es[-1] + sol[:, 0], label=f'E={len(Es)}')
    plt.hlines(Es[-1], -2, 2, linestyles='dashed', colors='gray', alpha=0.5)
    plt.annotate(
        f'$E_{len(Es)}$', (1.075, Es[-1]), textcoords="offset points", xytext=(0, 10), ha='center', color=colors[len(Es) - 1])
    plt.fill_between(x1, Es[-1] + sol[:, 0], Es[-1], alpha=0.3)
    expected = (len(Es))**2 * np.pi**2 / 4  # Expected energy levels
    print(f"State {len(Es)}: E = {Es[-1]:.3f}, Expected = {expected:.3f}")

    y0 = matrix.dot(y0)
time2 = time.time()
print(time2 - time1)

plt.hlines(0, -1, 1, colors='black')
plt.vlines([-1, 1], -1, Es[-1] * 1.1, colors='black')
plt.xlabel('Pozicija $x$')
plt.ylabel('Valovna funkcija $\Psi$')
plt.title('Prvih 5 lastnih stanj neskončne potencialne jame')
plt.xlim(-1.15, 1.15)
plt.tight_layout()
plt.show()

print(Es)
print([n**2 * np.pi**2 / 4 for n in range(1, 7)])
