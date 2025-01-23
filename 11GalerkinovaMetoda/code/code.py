import numpy as np
import matplotlib.pyplot as plt
from scipy.special import beta as B
from scipy.linalg import solve, solve_banded
from math import *
import time

# functions \xi**{2m+1} (1 - \xi)**n sin((2m+1) \phi)
# scalar product < m n | m' n'> = (m==m') * pi/2 * B(4 + 4m, 1 + n + n')

# matrix A = (m == m') * (-pi/2) * (n * n' * (3+4*m))/(2+4*m+n+n') * B(n+n'-1, 3+4*m)
# vector b = - 2/(2m' + 1) * B(2m'+3, n'+1)

M = 181
N = 181
print('Uporabljena M, N:', M, N)

time0 = time.time()
ab = np.zeros((2 * N - 1, M * N))
for i in range(M * N):
    for j in range(max(0, i - N + 1), min(M * N, i + N)):
        m, n = divmod(i, N)
        m0, n0 = divmod(j, N)
        if m == m0:
            ab[N - 1 + i - j, j] = - pi/2 * \
                ((n+1) * (n0+1) * (3 + 4*m)) / \
                (4 + 4*m + n + n0) * B(n + n0 + 1, 3 + 4*m)

b = np.zeros((M*N))
for m0 in range(M):
    for n0 in range(N):
        b[m0 * N + n0] = - 2 / (2*m0 + 1) * B(2*m0 + 3, n0 + 2)

a = solve_banded((N - 1, N - 1), ab, b)
C = - 32/pi * a@b
time1 = time.time()
print('time needed:', time1-time0)
print('banded', C)

weight
lotmo
zahtevnot