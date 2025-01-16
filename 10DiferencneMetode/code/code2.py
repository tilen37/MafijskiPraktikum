from matplotlib.colors import LinearSegmentedColormap
from cmath import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
plt.style.use('customStyle.mplstyle')

sigma0 = 1/20
k0 = 50 * pi
lam = 0.25
N = 1000
M = 3000
M = 1900

spaceline = np.linspace(-0.5, 1.5, N+1, endpoint=True)
dx = spaceline[1] - spaceline[0]

dt = 2 * dx**2
timeline = np.arange(0, dt*M, dt)
print(timeline.shape)

# Analitical solution


def real(x, t):
    sigma0_squared = 2 * sigma0**2
    complex_denom = 1 + 1j * t / sigma0_squared

    # Calculate the prefactor
    prefactor = (np.pi * sigma0_squared)**(-1/4) / np.sqrt(complex_denom)

    # Calculate the exponent terms
    pos_term = -(x - lam)**2 / (sigma0_squared * complex_denom)
    phase_term = 1j * (k0 * (x - lam) - k0**2 * t / 2) / complex_denom

    # Combine all terms
    psi = prefactor * np.exp(pos_term + phase_term)

    return psi


analitic = np.zeros((N+1, len(timeline)))
for i, t in enumerate(timeline):
    analitic[:, i] = abs(real(spaceline, t))**2
    # analitic[:, i] = abs(analitic[:, i])**2
    analitic[:, i] /= np.linalg.norm(analitic[:, i])
print(analitic.shape)


# solving
b = 1j * dt/(2 * dx**2)
a = -b/2
theta = 1/2 - 1/(12 * b)
theta = 1/2

A = a * (np.diag(np.ones(N-2), -1) +
         np.diag(np.ones(N-2), 1)) + b * np.eye(N-1)

Al = np.eye(N-1) - theta * A
Ar = np.eye(N-1) + (1 - theta) * A
A_eff = np.dot(np.linalg.inv(Al), Ar)

psi0 = (2*pi*sigma0**2)**-(1/4) * np.exp(-1j * k0 * (spaceline[1:-1] - lam)) * \
    np.exp(-(spaceline[1:-1] - lam)**2 / (2*sigma0**2))
psi = psi0 / np.linalg.norm(psi0)
# psi = analitic[1:-1, 0]

vfs = []
for _ in timeline:
    psi = np.dot(A_eff, psi)
    ro = abs(psi)**2
    ro /= np.linalg.norm(ro)

    vfs.append(np.r_[0, ro, 0])
    # if int(i/dt) % 200 == 0:
    #     plt.plot(spaceline, np.r_[0, abs(psi)**2, 0])
    # if i/T > 2:
    #     break
vfs = np.array(vfs).T

# plt.show()
# plt.close()

# # animacija
# fig, ax = plt.subplots()
# line, = ax.plot(spaceline, vfs[:, 0])
# line2, = ax.plot(spaceline, analitic[:, 0])
# # ax.set_ylim(0, .15)
# # ax.set_xlim(0, a)
# ax.set_xlabel('$x$')
# ax.set_ylabel(r'$\rho$')


# ind = 0
# psi = psi0


# def update(_):
#     global psi, ind
#     ind += 1
#     # for i in range(1):
#     #     psi = np.dot(A_eff, psi)
#     # psi = psi / np.linalg.norm(psi)
#     # line.set_ydata(np.r_[0, abs(psi)**2, 0])
#     line.set_ydata(vfs[:, ind])
#     line2.set_ydata(analitic[:, ind])
#     print(ind)
#     return line, line2,


# # Create animation
# anim = FuncAnimation(fig, update, frames=M-1,
#                      interval=10, blit=True)

# # Save animation
# writer = PillowWriter(fps=50)
# plt.title('Animacija rešitve prvih 100 korakov')
# plt.tight_layout()
# plt.show()
# plt.close()

# Compare to analytical solution
rmse = np.zeros(len(timeline))
for i in range(len(timeline)):
    rmse[i] = np.sqrt(np.mean((analitic[:, i] - vfs[:, i])**2))

# plt.plot(timeline, rmse)
# plt.xlabel(r'$t$')
# plt.ylabel(r'RMSE')
# plt.tight_layout()
# # plt.savefig(r'10DiferencneMetode/porocilo/rmse_free.pdf')
# plt.show()
# plt.close()

# Napaka v odv od prepotovane razdalje
razdalje = np.zeros(M)
for i in range(M):
    razdalje[i] = np.average(spaceline, axis=0, weights=analitic[:, i])

plt.plot(razdalje, rmse)
plt.xlabel(r'$< x >$')
plt.ylabel(r'RMSE')
plt.tight_layout()
plt.savefig(r'10DiferencneMetode/porocilo/rmse_free.pdf')
plt.show()
plt.close()

# # Create custom colormap
colors = ['#FFFFFF', '#FDD7CA', '#F3AB90', '#DF7F5D', '#C35B3E',
          '#A13A2A', '#7D1E1E', '#590F17', '#350813', '#0F0306']
custom_cmap = LinearSegmentedColormap.from_list('custom', colors, N=256)

plt.figure(figsize=(12, 8))
im = plt.pcolormesh(timeline, spaceline, vfs, cmap=custom_cmap, shading='auto')
plt.colorbar(im, label='Verjetnostna gostota')
plt.xlabel('$t$')
plt.ylabel('$x$')
plt.grid(False)
plt.tight_layout()
# plt.savefig(r'10DiferencneMetode/porocilo/evo_free.png')
plt.show()
