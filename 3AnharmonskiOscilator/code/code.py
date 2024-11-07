# Matrix Generation
from scipy.special import hermite
import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Computer Modern Serif",
    "font.size": 14
})


def matricni_element(i, j):
    return 1/2 * np.sqrt(i + j + 1) * (abs(i - j) == 1)


def matricni_element2(i, j):
    return 1/2 * (np.sqrt(j*(j-1)) * (i == j-2) + (2*j+1) * (i == j) + np.sqrt((j+1)*(j+2)) * (i == j+2))


def optimized_factorial_division(i, j):
    if i == j:
        return 1
    if i > j:
        result = 1
        for k in range(j + 1, i + 1):
            result *= k
        return result
    else:
        result = 1
        for k in range(i + 1, j + 1):
            result *= k
        return 1 / result


def matricni_element4(i, j):
    return 1/16 * np.sqrt(2.0**(i-j) * optimized_factorial_division(i, j)) * (
        (i == j+4) +
        4*(2*j+3) * (i == j+2) +
        12 * (2*j**2 + 2*j + 1) * (i == j) +
        16 * j * (2*j**2 - 3*j + 1) * (i == j-2) +
        16 * j * (j**3 - 6*j**2 + 11*j - 6) * (i == j-4)
    )


def matrix_of_dim(n, lam):
    H = np.zeros((n, n))
    H0 = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            H0[i, j] = (i == j) * ((i+1) + 0.5)
            H[i, j] = lam * matricni_element2(i, j)

    H = np.matrix(H) @ np.matrix(H) + np.matrix(H0)
    return H


# if __name__ == "__main__":
#     l = 1  # parameter lambda
#     N = 4000  # dimenzija

#     H0 = np.zeros((N, N))
#     H1 = np.zeros((N, N))
#     H2 = np.zeros((N, N))
#     # H3 = np.zeros((N, N))

#     for i in range(N):
#         for j in range(N):
#             H0[i, j] = (i == j) * ((i+1) + 0.5)
#             H1[i, j] = matricni_element(i, j)
#             H2[i, j] = matricni_element2(i, j)
#             # H3[i, j] = matricni_element4(i, j)

#     H0 = np.matrix(H0)
#     H1 = np.matrix(H1)
#     H2 = np.matrix(H2)
#     # H3 = np.matrix(H3)

#     H1 = H1 @ H1
#     H2 = H2 @ H2

#     H1 = H1 @ H1

#     # # H1 = H1[1*N//6:5*N//6, 1*N//6:5*N//6]
#     # # H2 = H2[1*N//6:5*N//6, 1*N//6:5*N//6]
#     # # H3 = H3[1*N//6:5*N//6, 1*N//6:5*N//6]

#     # print('H1 == H2', H1 == H2)
#     # # print('H1 == H3', H1 == H3)
#     # # print('H2 == H3', H2 == H3)

#     print(pd.DataFrame(l * H1))
#     print('-' * 100)
#     print(pd.DataFrame(l * H2))
#     # # print('-' * 100)
#     # # print(pd.DataFrame(l * H3))

#     # # Plot each matrix as a heatmap to visualize their structures
#     # fig, axes = plt.subplots(1, 3, figsize=(15, 5))

#     # for ax, mat, title in zip(
#     #     axes, [H1, H2, H3],
#     #     [r'$q$', r'$q^2$', r'$q^4$']
#     # ):
#     #     cax = ax.matshow(mat, cmap='viridis')
#     #     fig.colorbar(cax, ax=ax)
#     #     ax.set_title(title)

#     # plt.show()

####################################################################################################
# N = 1000

# import time

# print('Start matrix generation')
# H = matrix_of_dim(N, 1)

# print('Start numpy diagonalization')
# time1 = time.time()
# diag = np.linalg.eigh(H)[0]
# time2 = time.time()

# print('Time:', time2 - time1, 's')


# print('Start scipy diagonalization')
# time1 = time.time()
# diag = np.linalg.eigh(H)[0]
# time2 = time.time()

# print('Time:', time2 - time1, 's')

# from diag import trid_householder

# print('Start Hausholder diagonalization')
# time1 = time.time()
# diag = trid_householder(H)[0]
# time2 = time.time()

# print('Time:', time2 - time1, 's')

# from diag import qr

# print('Start QR diagonalization')
# time1 = time.time()
# diag = qr(H)[0]
# time2 = time.time()

# print('Time:', time2 - time1, 's')
####################################################################################################
# N = 1000


# def plot(l):
#     H = matrix_of_dim(N, l)

#     # print(pd.DataFrame(H))

#     H_diag = np.linalg.eigh(H)[0]

#     # print(H_diag)

#     plt.plot(H_diag, label=f'$\lambda = {l}$')


# lambda_range = np.linspace(0, 1, 5)
# print(lambda_range)

# for l in lambda_range:
#     plot(l)

# H0 = np.zeros((N, N))
# for i in range(N):
#     for j in range(N):
#         H0[i, j] = (i == j) * ((i+1) + 0.5)

# H0_diag = np.linalg.eigh(H0)
# # print(H0_diag)
# plt.plot(H0_diag[0], '--', label=r'H_0')

# plt.xlabel('Energijski nivo (i)')
# plt.ylabel('$\\frac{E}{\hbar \omega}$')
# plt.legend()
# plt.xlim(0, 100)
# plt.ylim(0, 700)
# plt.show()

####################################################################################################
# l = 1
# E = []


# def plot(N):
#     H = matrix_of_dim(int(N), l)

#     # print(pd.DataFrame(H))

#     H_diag = np.linalg.eigh(H)[0]

#     E.append(H_diag)

#     # plt.plot(H_diag, label=f'H_{N}')


# Nrange = np.linspace(2, 40, 40)

# for N in Nrange:
#     plot(N)

# print(len(E[0]), len(Nrange))

# print(pd.DataFrame(E))
# print(E[:][0])

# En = []
# E0 = []

# for i in range(len(E)):
#     En.append(E[i][0])
#     E0.append(E[i][1])

# E = pd.DataFrame(E).to_numpy()
# print('shape', E.shape)

# for i in range(6):
#     plt.plot(Nrange, E[:, i])

# # Add annotations
# for i in range(6):
#     plt.annotate(f'$E_{i}$', xy=(
#         Nrange[-1], E[-1, i]), xytext=(5, 0), textcoords='offset points')

# plt.ylabel('$\\frac{E}{\hbar \omega}$')
# plt.xlabel('Dimenzija matrike (N)')
# plt.xlim(0, 45)
# plt.yscale('log')
# plt.show()

####################################################################################################
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import hermite

def plot_wavefunctions(eigenvectors, x_range=(-6, 6), num_points=1000, num_states=4, lambda_val=0):
    """
    Plot wavefunctions from eigenvectors in the Hermite polynomial basis.
    
    Parameters:
    -----------
    eigenvectors : np.ndarray
        Matrix of eigenvectors from diagonalization
    x_range : tuple
        Range of x values to plot (in units of sqrt(ℏ/mω))
    num_points : int
        Number of points for plotting
    num_states : int
        Number of states to plot
    lambda_val : float
        Anharmonicity parameter for title
    """
    x = np.linspace(x_range[0], x_range[1], num_points)
    
    # Create figure with subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs = axs.ravel()
    
    for state in range(num_states):
        wavefunction = np.zeros(num_points)
        
        # Reconstruct wavefunction from Hermite polynomials
        for n in range(len(eigenvectors)):
            # Normalization factor
            norm = 1.0 / np.sqrt(2**n * np.sqrt(np.pi) * np.math.factorial(n))
            
            # Hermite polynomial
            Hn = hermite(n)
            
            # Add contribution from each basis state
            wavefunction += eigenvectors[n, state] * norm * Hn(x) * np.exp(-x**2/2)
        
        # Plot the wavefunction
        axs[state].plot(x, wavefunction)
        # axs[state].grid(True)
        axs[state].set_xlabel(r'$x/(\hbar/m\omega)^{1/2}$')
        axs[state].set_ylabel(r'$\psi(x)$')
        axs[state].set_title(f'Stanje {state + 1}, $\lambda$ = {lambda_val}')
        axs[state].set_ylim(-1.5, 1.5)
    
    plt.tight_layout()
    return fig

def generate_basis_states(n_max, x_range=(-6, 6), num_points=1000):
    """
    Generate the first n_max harmonic oscillator basis states.
    """
    x = np.linspace(x_range[0], x_range[1], num_points)
    basis_states = []
    
    for n in range(n_max):
        # Normalization factor
        norm = 1.0 / np.sqrt(2**n * np.sqrt(np.pi) * np.math.factorial(n))
        # Hermite polynomial
        Hn = hermite(n)
        # Normalized wavefunction
        psi = norm * Hn(x) * np.exp(-x**2/2)
        basis_states.append(psi)
    
    return np.array(basis_states), x

H = matrix_of_dim(100, 1)
eigenvalues, eigenvectors = np.linalg.eigh(H)

# Assuming you have your eigenvectors from diagonalization in 'eigenvectors'
lambda_val = 0.1  # or whatever value you used
fig = plot_wavefunctions(eigenvectors, lambda_val=lambda_val)
plt.show()
