# # Matematično Nihalo

# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.integrate import odeint
# from scipy.special import ellipk as K
# from scipy.special import ellipj as j
# from metode import *
# import time

# plt.style.use(r'customStyle.mplstyle')

# # Definiramo diferencialno enačbo
# # x''(t) = -b*x'(t) - c*sin(x(t))


# def f(y, t, b, c, d, omega):
#     x, v = y
#     ydt = np.array([v, - b * v - c*np.sin(x) + d*np.cos(omega * x)])
#     return ydt


# # Analitična rešitev
# def analiticna(y0, t, b, c, d, omega):
#     x0, v0 = y0
#     omega0 = np.sqrt(c)

#     sn, cn, dn, _ = j(K((np.sin(x0/2))**2) - omega0 * t, (np.sin(x0/2))**2)

#     x = 2 * np.arcsin(np.sin(x0/2) * sn)
#     v = 2 * omega0 * np.sin(x0/2) * dn / np.sqrt(1 -
#                                                  (np.sin(x0/2) * sn)**2) * cn

#     return np.array([x, v])


# # Definiramo začetne pogoje
# x0 = 1
# v0 = 0
# y0 = np.array([x0, v0])

# # Definiramo časovni vektor
# t = np.arange(0, int(1e6), 0.1)
# t0 = t
# t1 = t[t < 5]
# t2 = t[t < 1e3]
# t3 = t[t < 1e5]

# # Rešimo diferencialno enačbo
# b = 0     # Faktor dušenja
# c = 1     # Omega_0**2
# d = 0         # Amplituda vzbujanja
# omega = 1.5    # Frekvenca vzbujanja
# time0 = time.time()
# sol0 = odeint(f, y0, t0, args=(b, c, d, omega))
# time1 = time.time()
# print(f'Scipy: {time1 - time0} s, {len(t0)} steps')
# time0 = time.time()
# sol1 = euler(f, y0, t1, b, c, d, omega)
# time1 = time.time()
# print(f'euler: {time1 - time0} s, {len(t1)} steps')
# time0 = time.time()
# sol2 = mid(f, y0, t2, b, c, d, omega)
# time1 = time.time()
# print(f'mid: {time1 - time0} s, {len(t2)} steps')
# time0 = time.time()
# # sol3 = rk4(f, y0, t3, b, c, d, omega)
# # time1 = time.time()
# # print(f'rk4: {time1 - time0} s, {len(t3)} steps')

# time0 = time.time()
# real = analiticna(y0, t, b, c, d, omega).T
# time1 = time.time()
# print(f'analiticna: {time1 - time0} s, {len(t)} steps')

# # # Narišemo rešitev
# # plt.plot(t1, sol1[:, 0], label=r'Euler')
# # plt.plot(t2, sol2[:, 0], label=r'\textit{Midpoint}')
# # plt.plot(t3, sol3[:, 0], label=r'RK4')
# # # plt.plot(t0, sol0[:, 0], label=r'\texttt{SciPy}')
# # plt.plot(t, real[:, 0], label=r'Analitična')
# # plt.xlabel(r'$t$')
# # plt.ylabel(r'${\rm \theta}(t)$')
# # plt.ylim(-2, 2)
# # plt.tight_layout()
# # # plt.show()
# # plt.close()

# plt.rcParams['agg.path.chunksize'] = 10000
# # Narišemo fazni diagram
# plt.plot(sol1[:, 0], sol1[:, 1], label=r'Euler')
# plt.plot(sol2[:, 0], sol2[:, 1], label=r'\textit{Midpoint}')
# # plt.plot(sol3[:, 0], sol3[:, 1], label=r'RK4')
# plt.plot(sol0[:, 0], sol0[:, 1], label=r'\texttt{SciPy}')
# plt.plot(real[:, 0], real[:, 1], label=r'Analitična')
# plt.xlabel(r'$\theta$')
# plt.ylabel(r'$\dot{\theta}$')
# plt.legend(fontsize=12, loc='upper center', bbox_to_anchor=(
#     0.5, 1.04), ncol=6, framealpha=1, edgecolor='black')
# # plt.ylim(-2, 2)
# plt.tight_layout()
# plt.show()
# plt.close()

# # Narišemo energijo
# E0 = 1 - np.cos(sol0[:, 0]) + 0.5/c * sol0[:, 1]**2
# E1 = 1 - np.cos(sol1[:, 0]) + 0.5/c * sol1[:, 1]**2
# E2 = 1 - np.cos(sol2[:, 0]) + 0.5/c * sol2[:, 1]**2
# # E3 = 1 - np.cos(sol3[:, 0]) + 0.5/c * sol3[:, 1]**2

# E = 1 - np.cos(real[:, 0]) + 0.5/c * real[:, 1]**2

# plt.plot(t1, E1, label=r'Euler')
# # plt.plot(t3, E3, label=r'RK4')
# plt.plot(t2, E2, label=r'\textit{Midpoint}')
# plt.plot(t0, E0, label=r'\texttt{SciPy}')
# plt.plot(t, E, label=r'Analitična')
# plt.xlabel(r'$t$')
# plt.ylabel(r'$E$')
# plt.legend()
# # plt.ylim(0.45, 1e0)
# plt.yscale('log')
# plt.tight_layout()
# plt.show()
# plt.close()


# Matematično Nihalo
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.special import ellipk as K
from scipy.special import ellipj as j
from metode import *
import time

plt.style.use(r'customStyle.mplstyle')
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

# Definiramo diferencialno enačbo
# x''(t) = -b*x'(t) - c*sin(x(t))


def f(y, t, b, c, d, omega):
    x, v = y
    ydt = np.array([v, - b * v - c*np.sin(x) + d*np.cos(omega * x)])
    return ydt


# Analitična rešitev
def analiticna(y0, t, b, c, d, omega):
    x0, v0 = y0
    omega0 = np.sqrt(c)

    sn, cn, dn, _ = j(K((np.sin(x0/2))**2) - omega0 * t, (np.sin(x0/2))**2)

    x = 2 * np.arcsin(np.sin(x0/2) * sn)
    v = 2 * omega0 * np.sin(x0/2) * dn / np.sqrt(1 -
                                                 (np.sin(x0/2) * sn)**2) * cn

    return np.array([x, v])


# Definiramo začetne pogoje
x0 = 1.
v0 = 0.
y0 = np.array([x0, v0])

# Definiramo parametre
b = 0     # Faktor dušenja
c = 1     # Omega_0**2
d = 0         # Amplituda vzbujanja
omega = 1.5    # Frekvenca vzbujanja

# Definiramo časovni vektor
t = np.arange(0, int(1e7), 0.1)
t0 = t
t1 = t[t < 5]
t2 = t[t < 1e3]
t3 = t[t < 5e5]
t4 = t[t < 1e7]
t5 = t[t < 1e6]

time0 = time.time()
sol4 = verlet(f, y0, t4, b, c, d, omega)
time1 = time.time()

# print(f'Verlet: {time1 - time0} s, {len(t4)} steps')
# np.save('7NewtonovZakon/code/sol4.npy', sol4)

# time0 = time.time()
# sol5 = pefrl(f, y0, t5, b, c, d, omega)
# time1 = time.time()

# print(f'PEFRL: {time1 - time0} s, {len(t4)} steps')
# np.save('7NewtonovZakon/code/sol5.npy', sol5)

real = np.load(r'7NewtonovZakon/code/real.npy')
sol0 = np.load(r'7NewtonovZakon/code/sol0.npy')
sol1 = np.load(r'7NewtonovZakon/code/sol1.npy')
sol2 = np.load(r'7NewtonovZakon/code/sol2.npy')
sol3 = np.load(r'7NewtonovZakon/code/sol3.npy')
sol4 = np.load(r'7NewtonovZakon/code/sol4.npy')
sol5 = np.load(r'7NewtonovZakon/code/sol5.npy')

# Narišemo energijo

E = 1 - np.cos(real[:, 0]) + 0.5/c * real[:, 1]**2

E0 = 1 - np.cos(sol0[:, 0]) + 0.5/c * sol0[:, 1]**2
E1 = 1 - np.cos(sol1[:, 0]) + 0.5/c * sol1[:, 1]**2
E2 = 1 - np.cos(sol2[:, 0]) + 0.5/c * sol2[:, 1]**2
E3 = 1 - np.cos(sol3[:, 0]) + 0.5/c * sol3[:, 1]**2
E4 = 1 - np.cos(sol4[:, 0]) + 0.5/c * sol4[:, 1]**2
E5 = 1 - np.cos(sol5[:, 0]) + 0.5/c * sol5[:, 1]**2

print('Here')
time0 = time.time()
plt.rcParams['agg.path.chunksize'] = 10000
plt.plot(t[t < 5e6], (E)[t < 5e6], label=r'Analitična', c=colors[5])
plt.plot(t1, E1, label=r'Euler')
plt.plot(t2, E2, label=r'\textit{Midpoint}')
plt.plot(t3, E3, label=r'RK4')
plt.plot(t0[t < 5e6], E0[t < 5e6], label=r'\texttt{SciPy}')
plt.plot(t4[t < 5e6], E4[t < 5e6], label=r'Verlet')
plt.plot(t5, E5, label=r'PEFRL')
plt.xlabel(r'$t$')
plt.ylabel(r'$E$')
plt.legend(fontsize=12, loc='upper center', bbox_to_anchor=(
    0.5, 1.04), ncol=6, framealpha=1, edgecolor='black')
# plt.ylim(0.45, 1e0)
plt.xscale('log')
# plt.yscale('log')
plt.title('Odvisnost energije od časa pri uporabi različnih metod', pad=20)
# plt.savefig('energija.pdf')
plt.tight_layout()
time1 = time.time()
print(f'Time to plot: {time1 - time0} s')
plt.show()
plt.close()
