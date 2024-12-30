import numpy as np
import matplotlib.pyplot as plt
from uncertainties import ufloat

# Solving Schrödinger equation in finite or infinite potential well

# Constants
# a = 2
N = 1000
h = 1/N
A = np.diag(np.ones(N-3), -1) + np.diag(np.ones(N-3), 1) + \
    np.diag((-2)*np.ones(N-2))


def f(x, y, yprime, lam):  # y'' = - lam * y
    return - lam * y


def finite_difference(E, alpha, beta):
    def V(Y0):
        assert Y0.shape[0] == N - 2
        Y = np.r_[[alpha], Y0, [beta]]
        return h*h * np.array([f(h * i, Y[i], 1/2/h * Y[i+1] - Y[i-1], E) for i in range(1, N-1)]) - np.r_[[alpha], np.zeros(N-4), [beta]]

    def F(Y0):
        return A.dot(Y0) - V(Y0)

    # Find the root of F with Newton's method
    Y = np.ones(N-2)
    Y = np.array([np.sin(np.pi * i**2 / N) for i in range(1, N-1)])
    for i in range(10000):
        Y = Y - np.linalg.inv(A).dot(F(Y))
        if np.linalg.norm(F(Y)) < 1e-10:
            print(f"Converged in {i} iterations.")
            break

    Y /= np.sqrt(np.trapezoid(Y**2, np.linspace(-1, 1, N-2)))
    lam = (A.dot(Y) / Y)
    lam = ufloat(np.mean(lam), np.std(lam))
    return Y, lam


sol, E = finite_difference(5, 0, 0)

print(E)
x = np.linspace(-1, 1, N)
plt.plot(x, np.r_[0, sol, 0])
plt.show()


# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.linalg import eigh
# import time
# plt.style.use('customStyle.mplstyle')
# colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


# # Parameters
# a = 2.0  # Well width
# N = 1000  # Grid points
# h = a/N   # Grid spacing
# x = np.linspace(-a/2, a/2, N+1)


# def solve_schrodinger():
#     # Construct Hamiltonian matrix (-∇²/2 + V)
#     # Second derivative operator
#     diag = np.ones(N-1) * (-2.0)/(h*h)
#     offdiag = np.ones(N-2)/(h*h)
#     H = np.diag(diag) + np.diag(offdiag, k=1) + np.diag(offdiag, k=-1)

#     # Add potential (zero inside well, infinite outside is handled by BC)
#     V = np.zeros(N-1)  # V=0 inside well

#     # Solve eigenvalue problem
#     E, psi = eigh(H)  # Get first 5 states
#     # Set the first energy level to the expected value
#     E = E - E[0] + np.pi**2 / 4

#     # Normalize wavefunctions
#     for i in range(5):
#         norm = np.sqrt(h * np.sum(psi[:, i]**2))
#         psi[:, i] /= norm

#     return E, psi


# # Solve and plot
# E, psi = solve_schrodinger()

# # Plot results
# x_plot = x[1:-1]  # Interior points only

# for n in range(5):
#     plt.plot(x_plot, psi[:, n] + E[n],
#              label=f'n={n+1}, E={E[n]:.2f}')
#     plt.axhline(y=E[n], color='gray', linestyle='--', alpha=0.3)

# plt.axvline(x=-a/2, color='k', linewidth=2)
# plt.axvline(x=a/2, color='k', linewidth=2)
# plt.xlabel('$x$')
# plt.ylabel(r'$\Psi(x) + E_{\rm n}$')
# plt.title('Wavefunctions for Infinite Square Well')
# plt.show()

# # Print energies
# print("\nEnergy levels:")
# for n, En in enumerate(E[:5]):
#     expected = (n+1)**2 * np.pi**2/(a**2)
#     print(f"n={n+1}: E={En:.4f}, Expected={expected:.4f}")


# # def J(E):
# #     A_temp = -0.5
# #     B_temp = -0.5
# #     C_temp = 1 - h*h/2 * E

# #     return np.diag(np.ones(N-2), -1) * A_temp + np.diag(np.ones(N-2), 1) * B_temp + np.diag(np.ones(N-1)) * C_temp


# # def G(Y, E):
# #     return Y - np.linalg.inv(J(E)) @ F(Y, E)


# # # Začetni približek
# # Y = np.ones(N-1)
# # E = 5

# # E = newton(F, E, args=(E,), maxiter=1000)

# # for i in range(100):
# #     Y = G(Y, E)
# #     if np.linalg.norm(F(Y, E)) < 1e-10:
# #         print(f"Converged in {i} iterations.")
# #         break

# # print(E)
# # Y = np.r_[alpha, Y, beta]
# # plt.plot(x, np.r_[alpha, Y, beta])
# # plt.show()
