import numpy as np
import math
import pandas as pd


def matricni_element(i, j):
    return 1/2 * np.sqrt(i + j + 1) * (abs(i - j) == 1)


def matricni_element2(i, j):
    return 1/2 * (np.sqrt(j*(j-1)) * (i == j-2) + (2*j+1) * (i == j) + np.sqrt((j+1)*(j+2)) * (i == j+2))


def matricni_element4(i, j):
    return 1/16 * np.sqrt(2**(i-j) * math.factorial(i) / math.factorial(j)) * ((i == j+4) + 4*(2*j+3) * (i == j+2) + 12 * (2*j**2 + 2*j + 1) * (i == j) + 16 * j * (2*j**2 - 3*j + 1) * (i == j-2) + 16 * j * (j**3 - 6*j**2 + 11*j - 6) * (i == j-4))


def matrix_of_dim(n, lam):
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            H[i, j] = (i == j) * ((i+1) + 0.5) + lam * matricni_element4(i, j)
    H = np.matrix(H)
    return H


l = 1  # parameter lambda
N = 15  # dimenzija

if __name__ == "__main__":
    H0 = np.zeros((N, N))
    H1 = np.zeros((N, N))
    H2 = np.zeros((N, N))
    H3 = np.zeros((N, N))

    for i in range(N):
        for j in range(N):
            H0[i, j] = (i == j) * ((i+1) + 0.5)
            H1[i, j] = matricni_element(i, j)
            H2[i, j] = matricni_element2(i, j)
            H3[i, j] = matricni_element4(i, j)

    H0 = np.matrix(H0)
    H1 = np.matrix(H1)
    H2 = np.matrix(H2)
    H3 = np.matrix(H3)

    H1 = H1 @ H1
    H2 = H2 @ H2

    H1 = H1 @ H1

    print(pd.DataFrame(l * H1))
    print('-' * 100)
    print(pd.DataFrame(l * H2))
    print('-' * 100)
    print(pd.DataFrame(l * H3))

    import matplotlib.pyplot as plt

    # Plot each matrix as a heatmap to visualize their structures
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, mat, title in zip(
        axes, [np.linalg.matrix_power(
            H1, 4), np.linalg.matrix_power(H2, 2), H3],
        ['l * H1^4', 'l * H2^2', 'l * H3']
    ):
        cax = ax.matshow(mat, cmap='viridis')
        fig.colorbar(cax, ax=ax)
        ax.set_title(title)

    plt.show()
