import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

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

# 2nd attempt

N = 100
D = 1e-3
a = 3
h = a/N
# začetni pogoj
T_0 = 100
sigma = 0.1


def f(x):
    return T_0 * np.exp(-(x - a/2)**2 / (sigma**2))


# def f(x):
#     return T_0 * np.sin(np.pi*x/a)


A = np.diag(4 * np.ones(N-1)) + np.diag(1 * np.ones(N-2), k=1) + \
    np.diag(1 * np.ones(N-2), k=-1)

B = 6*D/(h*h) * (np.diag(-2 * np.ones(N-1)) + np.diag(1 * np.ones(N-2), k=1) +
                 np.diag(1 * np.ones(N-2), k=-1))


x = np.linspace(h, a, N//3)

x1 = np.r_[x-a, x, x+a]
f1 = np.r_[-f(x), f(x), f(x)]


# f1_temp = np.array([f(i * h) for i in range(1, N//3+1)])
# f1 = np.r_[-f1_temp, f1_temp, f1_temp]
c1 = np.linalg.inv(A) @ f1

f0 = np.array([f(i * h) for i in range(1, N)])
c0 = np.linalg.inv(A) @ f0

dt = 1
t = 101


def implicit_Euler(A, B, dt, c0, t):
    solutions = [c0]

    for _ in np.arange(dt, t, dt):
        solutions.append(np.linalg.solve(
            A - dt/2 * B, (A + dt/2 * B) @ solutions[-1]))

    return np.array(solutions)


# solutions = implicit_Euler(A, B, dt, c0, t)
solutions = implicit_Euler(A, B, dt, c1, t)

# Dirichlet
# solutions = np.pad(solutions, ((0, 0), (1, 1)), mode='constant')
# solutions = np.pad(solutions, ((0, 0), (1, 1)), mode='reflect')
# Neumann
solutions = np.pad(solutions, ((0, 0), (2, 2)), mode='edge')

xline = np.linspace(0, a, N*10, endpoint=True)


def B_spline(k, x):
    if x < (k-2)*h:
        return 0
    elif x < (k-1)*h:
        return 1/h**3 * (x - (k-2)*h)**3
    elif x < (k * h):
        return 1/h**3 * ((x - (k-2)*h)**3 - 4 * (x - (k-1)*h)**3)
    elif x < (k+1)*h:
        return 1/h**3 * (((k+2)*h - x)**3 - 4 * ((k+1)*h - x)**3)
    elif x < (k+2)*h:
        return 1/h**3 * (((k+2)*h - x)**3)
    return 0


def T(t):
    sol = solutions[int(t/dt), :]
    # sol = c0
    # sol = np.r_[sol[0], 0, sol, 0, sol[-1]]

    T = np.zeros(len(xline))
    for i, xj in enumerate(xline):
        for k in range(-1, N+1):
            ck = sol[k+1]
            T[i] += ck * B_spline(k, xj)

    return T


samples = [0, 5, 10, 20, 50, 100]
for i in samples:
    T_f1 = T(i)
    plt.plot(xline-1, T_f1, label=f'{i}', c=colors[samples.index(i)])

plt.title(f'Rešitev skozi čas', pad=25)
plt.xlabel('$x$')
plt.ylabel('$T$')
plt.xlim(0, 1)
plt.ylim(0, 105)
plt.legend(fontsize=18, loc='upper center', bbox_to_anchor=(
    0.5, 1.06), ncol=6, framealpha=1, edgecolor='black')
plt.savefig(f'fem_neu_zrc.pdf')
plt.show()
plt.close()

plt.plot(xline-1, T(5))
plt.plot(x1-1, f1)
plt.title('Rešitev po 20 korakih')
plt.xlabel('$x$')
plt.ylabel('$T$')
plt.xlim(0, 1)
plt.ylim(0, 100)
# plt.savefig('fem_dirichlet.pdf')
# plt.show()
plt.close()


# animacija
fig, ax = plt.subplots()
line, = ax.plot(xline-1, T(0))
# line, = ax.plot(x1, f1)
ax.set_ylim(0, 100)
ax.set_xlim(0, 1)
ax.set_xlabel('$x$')
ax.set_ylabel('$T$')


def update(frame):
    line.set_ydata(T(frame))
    return line,


# Create animation
anim = FuncAnimation(fig, update, frames=100,
                     interval=100, blit=True)

# Save animation
writer = PillowWriter(fps=10)
plt.title('Animacija rešitve prvih 100 korakov')
anim.save('fem_neu_zrc.gif', writer=writer)
plt.show()
plt.close()
