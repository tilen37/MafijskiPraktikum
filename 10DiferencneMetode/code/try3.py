import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from cmath import *

plt.style.use('customStyle.mplstyle')
colors = ['#FFFFFF',
          '#FDD7CA',
          '#F3AB90',
          '#DF7F5D',
          '#C35B3E',
          '#A13A2A',
          '#7D1E1E',
          '#590F17',
          '#350813',
          '#0F0306']
colors = colors[2:]

# začetni pogoj
T_0 = 100
a = 1
sigma = 0.1
D = 1e-4j


def f(x):
    return T_0 * np.exp(-(x - a/2)**2 / (sigma**2))


# def f(x):
#     return T_0 * np.sin(np.pi*x/a)


# diskretizacija
N = 1000
x = np.linspace(0, a, N, endpoint=False)
k = np.fft.fftfreq(N, d=a/N)
h = x[1] - x[0]

x1 = np.r_[x-a, x, x+a]
k1 = np.fft.fftfreq(3*N, d=a/N)

f0 = np.r_[-f(x), f(x), f(x)]
F0 = Hk = np.fft.fft(f0)

plt.plot(x1, f0)
plt.title('Začetni pogoj')
plt.xlabel('$x$')
plt.ylabel('$T$')
# plt.savefig('zrcaljeni_zp_sin.pdf')
# plt.show()
plt.close()

plt.plot(k1, F0)
plt.title('Fourierjeva transformacija začetnega pogoja')
plt.xlabel('$k$')
plt.ylabel('$|F_0|$')
# plt.show()
plt.close()

# c_k(t) = H_k * exp(-4pi² D k²/a² t)


def T_f(t):
    return Hk * np.exp(-4*np.pi**2 * D * k1**2 / a**2 * t)


T_f1 = T_f(20)

T = np.fft.ifft(T_f1)

samples = [0, 5, 10, 20, 50, 100]
for i in samples:
    T_f1 = T_f(i)
    T = np.fft.ifft(T_f1)
    plt.plot(x1, abs(T), label=f'{i}', c=colors[samples.index(i)])

plt.title(f'Rešitev skozi čas', pad=25)
plt.xlabel('$x$')
plt.ylabel('$T$')
plt.xlim(0, a)
plt.ylim(0, 105)
plt.legend(fontsize=18, loc='upper center', bbox_to_anchor=(
    0.5, 1.06), ncol=6, framealpha=1, edgecolor='black')
# plt.savefig(f'resitev_zrcaljena.pdf')
plt.show()
plt.close()

# # plt.plot(x1, T)
# plt.plot(x, T[N:2*N])
# plt.title('Rešitev po 20 korakih')
# plt.xlabel('$x$')
# plt.ylabel('$T$')
# plt.xlim(0, 1)
# plt.ylim(0, 100)
# # plt.savefig('dirichlet_neumann_sin.pdf')
# # plt.show()
# plt.close()


# animacija
fig, ax = plt.subplots()
# line, = ax.plot(x, T[N:2*N])
line, = ax.plot(x1, T)
ax.set_ylim(0, 100)
ax.set_xlim(0, a)
ax.set_xlabel('$x$')
ax.set_ylabel('$T$')


def update(frame):
    T_f1 = T_f(frame)
    T = np.fft.ifft(T_f1)
    # line.set_ydata(T[N:2*N])
    line.set_ydata(abs(T))
    # line.set_ydata(np.real(T))
    # line.set_ydata(np.imag(T))
    # line.set_ydata(T)
    return line,


# Create animation
anim = FuncAnimation(fig, update, frames=100,
                     interval=100, blit=True)

# Save animation
writer = PillowWriter(fps=10)
plt.title('Animacija rešitve prvih 100 korakov')
# anim.save('dirichlet_neumann_sin.gif', writer=writer)
plt.show()
plt.close()
