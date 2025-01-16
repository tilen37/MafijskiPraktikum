from cmath import *
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('customStyle.mplstyle')

N = 100
M = 10000

timeline = np.linspace(0, 1, M)
spaceline = np.linspace(0, 1, N)

dt = timeline[1] - timeline[0]
dx = spaceline[1] - spaceline[0]

# Potential


def V(i): return 1/2 * (dx*i)**2
def V(i): return 0


Vx = V(spaceline)

r = 1j * dt/(2*dx**2)
theta = 1/2 - 1/(12 * r)

A = np.diag(np.ones(N-1), -1) + np.diag(np.ones(N-1), 1) - \
    2 * np.diag(np.ones(N))

Al = 1 - theta * r * A
Ar = 1 + (1 - theta) * r * A

psi0 = np.exp(- (0.5 - spaceline)**2 * 64)
psi = psi0 = psi0 / np.linalg.norm(psi0)

plt.plot(spaceline, psi0)
# plt.plot(spaceline, Vx / np.max(Vx) * np.max(psi0))
# plt.show()

for i in range(1):
    RHS = np.dot(Ar, psi)
    # + np.r_[r/2, np.zeros(N-2), -r/2]
    psi = np.linalg.solve(Al, RHS)
    # psi = psi / np.linalg.norm(psi)

    if i % 10 == 0:
        plt.plot(spaceline, abs(psi))

plt.show()
