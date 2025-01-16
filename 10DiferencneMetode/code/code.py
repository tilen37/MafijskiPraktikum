from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import zoom
from cmath import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
plt.style.use('customStyle.mplstyle')

omega = 0.2
lam = 10.
k = omega**2
alpha = k**(1/4)
N = 300
M = 10
w = 10

# timeline = np.linspace(0, 1, M)
spaceline = np.linspace(-40, 40, N+1, endpoint=True)

# dt = timeline[1] - timeline[0]
dx = spaceline[1] - spaceline[0]

dt = 1/3 * dx**2
T = 2 * pi / omega
print(M*T)
timeline = np.arange(0, M*T, dt)

# Analitical solution
ksi_lam = alpha * lam


def real(x, t):
    omega = .1
    # Calculate the components inside the exponential
    term1 = -0.5 * (alpha * x - ksi_lam * np.cos(omega * t))**2
    term2 = -1j * (omega * t / 2 + alpha * x * ksi_lam * np.sin(omega * t) -
                   0.25 * ksi_lam * ksi_lam * np.sin(2 * omega * t))

    # Combine all terms
    psi = sqrt(alpha / pi) * np.exp(term1 + term2)

    return psi


analitic = np.zeros((N+1, len(timeline)))
for i, t in enumerate(timeline):
    analitic[:, i] = abs(real(spaceline, t))**2
    analitic[:, i] = analitic[:, i] * np.linalg.norm(analitic[:, i])

# Potentials
# Vx = 100 * (w/2 - spaceline[1:-1])**2
Vx = 1/2 * k * spaceline[1:-1]**2
# Vx = 0

b = 1j * dt/(2 * dx**2)
a = -b/2
theta = 1/2 - 1/(12 * b)
print(theta)
theta = 1/2

A = a * (np.diag(np.ones(N-2), -1) + np.diag(np.ones(N-2), 1)) + \
    (b + 1j * dt/2 * Vx) * np.eye(N-1)

Al = np.eye(N-1) - theta * A
Ar = np.eye(N-1) + (1 - theta) * A
A_eff = np.dot(np.linalg.inv(Al), Ar)

psi0 = sqrt(alpha / sqrt(pi)) * np.exp(- alpha **
                                       2 * (spaceline[1:-1] - lam)**2 / 2)
psi = psi0 / np.linalg.norm(psi0)

vfs = []
for i in np.arange(0, M*T, dt):
    psi = np.dot(A_eff, psi)
    psi = psi / np.linalg.norm(psi)

    vfs.append(np.r_[0, abs(psi)**2, 0])
    # if int(i/dt) % 200 == 0:
    #     plt.plot(spaceline, np.r_[0, abs(psi)**2, 0])
    # if i/T > 2:
    #     break
vfs = np.array(vfs).T

# plt.tight_layout()
# plt.show()
# plt.close()

# # animacija
# fig, ax = plt.subplots()
# line, = ax.plot(spaceline[1:-1], Vx / np.max(Vx), '--', c='gray')
# line, = ax.plot(spaceline, np.r_[0, abs(psi), 0])
# line2, = ax.plot(spaceline, analitic[:, 0])
# ax.set_ylim(0, .15)
# # ax.set_xlim(0, a)
# ax.set_xlabel('$x$')
# ax.set_ylabel(r'$\rho$')


# ind = 0


# def update(_):
#     global psi, ind
#     ind += 1
#     for i in range(100):
#         psi = np.dot(A_eff, psi)
#         # psi = psi / np.linalg.norm(psi)
# line.set_ydata(np.r_[0, abs(psi)**2, 0])
#     line2.set_ydata(analitic[:, ind*50])
#     return line, line2,


# # Create animation
# anim = FuncAnimation(fig, update, frames=int(M*T+1)//100,
#                      interval=100, blit=True)

# # Save animation
# writer = PillowWriter(fps=50)
# plt.title('Animacija rešitve prvih 100 korakov')
# plt.tight_layout()
# # plt.show()
# plt.close()

# print(vfs.shape, analitic.shape)
# # Compare to analytical solution
# rmse = np.zeros(len(spaceline))
# for i in range(len(rmse)):
#     rmse[i] = np.sqrt(np.mean((analitic[i, :] - vfs[i, :])**2))

# plt.plot(spaceline, rmse)
# plt.xlabel('$x$')
# plt.ylabel('RMSE')
# plt.tight_layout()
# # plt.savefig(r'10DiferencneMetode/porocilo/rmse_space.pdf')
# plt.show()
# plt.close()

# Plot as field

# Create custom colormap
# downsample_factor = 0.25
# vfs_downsampled = zoom(vfs, downsample_factor)

# # Adjust corresponding axes
# timeline_downsampled = np.linspace(
#     timeline[0], timeline[-1], vfs_downsampled.shape[1])
# spaceline_downsampled = np.linspace(
#     spaceline[0], spaceline[-1], vfs_downsampled.shape[0])

# # Plot
# colors = ['#FFFFFF', '#FDD7CA', '#F3AB90', '#DF7F5D', '#C35B3E',
#           '#A13A2A', '#7D1E1E', '#590F17', '#350813', '#0F0306']
# custom_cmap = LinearSegmentedColormap.from_list('custom', colors, N=256)

# plt.figure(figsize=(12, 8))
# # im = plt.pcolormesh(timeline_downsampled, spaceline_downsampled, vfs_downsampled,
# #                     cmap=custom_cmap, shading='auto')
# im = plt.pcolormesh(timeline, spaceline, vfs,
#                     cmap=custom_cmap, shading='auto')
# plt.colorbar(im, label='Verjetnostna gostota')
# plt.xlabel('$t$')
# plt.ylabel('$x$')
# plt.grid(False)
# plt.tight_layout()
# # plt.savefig(r'10DiferencneMetode/porocilo/evo.png')
# plt.show()
# plt.close()


samples = np.linspace(0.1, 1, 5, endpoint=False)
samples = [.5]
samples = [-.1, .1, .5, .9, 1.1]
samples = [.2, .4, .6, .8,]
samples = .5 + np.array(samples)-1.5 * 1j
samples = .5 - np.array([.75, 0.5, .25]) * 1j
print(samples)
errs = []

for theta in samples:
    A = a * (np.diag(np.ones(N-2), -1) + np.diag(np.ones(N-2), 1)) + \
        (b + 1j * dt/2 * Vx) * np.eye(N-1)

    Al = np.eye(N-1) - theta * A
    Ar = np.eye(N-1) + (1 - theta) * A
    A_eff = np.dot(np.linalg.inv(Al), Ar)

    psi0 = sqrt(alpha / sqrt(pi)) * np.exp(- alpha **
                                           2 * (spaceline[1:-1] - lam)**2 / 2)
    psi = psi0 / np.linalg.norm(psi0)

    vfs = []
    for i in np.arange(0, M*T, dt):
        psi = np.dot(A_eff, psi)
        psi = psi / np.linalg.norm(psi)

        vfs.append(np.r_[0, abs(psi)**2, 0])
        # if int(i/dt) % 200 == 0:
        #     plt.plot(spaceline, np.r_[0, abs(psi)**2, 0])
        # if i/T > 2:
        #     break
    vfs = np.array(vfs).T

    rmse = np.zeros(len(timeline))
    for i in range(len(timeline)):
        rmse[i] = np.sqrt(np.mean((analitic[:, i] - vfs[:, i])**2))

    errs.append(rmse)

sample_names = ['0,1', '0,3', '0,5', '0,7', '0,9']
sample_names = [r'0,5 - $\mathrm{i}$', r'0,5 - $\mathrm{i}$ 0,8',
                r'0,5 - $\mathrm{i}$ 0,6', r'0,5 - $\mathrm{i}$ 0,4']
sample_names = [r'0,5 - $\mathrm{i}$ 1,75',
                r'0,5 - $\mathrm{i}$ 0,5', r'0,5 - $\mathrm{i}$ 0,25']
for i, err in enumerate(errs):
    plt.plot(timeline, err, label=sample_names[i])

plt.tight_layout()
plt.legend(fontsize=18, loc='upper center', bbox_to_anchor=(
    0.5, 1.06), ncol=6, framealpha=1, edgecolor='black')
plt.xlabel('$t$')
plt.ylabel('RMSE')
plt.tight_layout()
plt.savefig(r'10DiferencneMetode/porocilo/theta3.png')
plt.show()
plt.close()
