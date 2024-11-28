from matplotlib.lines import Line2D
import mpmath as mp
from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt

mp.dps = 15
plt.style.use(r'customStyle.mplstyle')
c = colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


# Parametri
T_zun = -5
k = 0.1


def real(t, T0):
    # Ker je ta funkcija analitična rešitev, jo vzamem za referenco
    return T_zun + np.exp(-k*t) * (T0 - T_zun)


def f(t, T):
    return -k*(T - T_zun)


def rmse(x, y):
    return np.sqrt(np.mean((x - y)**2))


# Začetni pogoji
T0 = -15
T0 = 21

h = h_e = h_m = h_rk = .1
t = np.arange(0, 100, h)

##########################################################################
# Analitična rešitev
# plt.plot(t, real(t, T0), label='Analitična rešitev')
# plt.xlabel('Čas')
# plt.ylabel('Temperatura')
# plt.title(f'Analitična rešitev za $T(t=0) = {T0}$')
# # plt.savefig('6EnacbeHoda/porocilo/analiticno21.pdf')
# plt.show()

##########################################################################
# Eulerjeva metoda


def euler(f, t=t, T0=T0, h=h):
    h_e = h
    t_e = t

    T_e = np.zeros_like(t_e)
    T_e[0] = T0

    for i in range(1, len(t_e)):
        T_e[i] = T_e[i-1] + h_e*f(t_e[i-1], T_e[i-1])

    # plt.plot(t_e, real(t_e, T0), label='Analitična rešitev')
    # plt.plot(t_e, T_e, label='Eulerjeva metoda')
    # plt.legend()

    # print('Eulerjeva metoda rmse =', rmse(
    # real(t_e, T0), T_e), 'pri koraku h =', h_e)

    return t_e, T_e


##########################################################################
# Midpoint metoda
def midpoint(f, t=t, T0=T0, h=h):
    h_m = h
    t_m = t

    T_m = np.zeros_like(t_m)
    T_m[0] = T0

    for i in range(1, len(t_m)):
        k1 = f(t_m[i-1], T_m[i-1])
        k2 = f(t_m[i-1] + h_m/2, T_m[i-1] + h_m/2*k1)
        T_m[i] = T_m[i-1] + h_m*k2

    # plt.plot(t_m, real(t_m, T0), label='Analitična rešitev')
    # plt.plot(t_m, T_m, label='Midpoint metoda')
    # plt.legend()
    # plt.show()

    # print('Midpoint metoda rmse =', rmse(
        # real(t_m, T0), T_m), 'pri koraku h =', h_m)

    return t_m, T_m

##########################################################################
# RK4 metoda


def rk4(f, t=t, T0=T0, h=h):

    h_rk = h
    t_rk = t

    T_rk = np.zeros_like(t_rk)
    T_rk[0] = T0

    for i in range(1, len(t_rk)):
        k1 = f(t_rk[i-1], T_rk[i-1])
        k2 = f(t_rk[i-1] + h_rk/2, T_rk[i-1] + h_rk/2*k1)
        k3 = f(t_rk[i-1] + h_rk/2, T_rk[i-1] + h_rk/2*k2)
        k4 = f(t_rk[i-1] + h_rk, T_rk[i-1] + h_rk*k3)
        T_rk[i] = T_rk[i-1] + h_rk/6*(k1 + 2*k2 + 2*k3 + k4)

    # plt.plot(t_rk, real(t_rk, T0), label='Analitična rešitev')
    # plt.plot(t_rk, T_rk, label='RK4 metoda')
    # plt.legend()
    # plt.show()

    # print('RK4 metoda rmse =', rmse(real(t_rk, T0), T_rk), 'pri koraku h =', h_rk)

    return t_rk, T_rk

##########################################################################
# Builtin RK4
# sol = solve_ivp(f, [0, 10], [T0], t_eval=t)

# plt.plot(sol.t, sol.y[0], label=r'\textit{Scipy} RK4')
# plt.legend()
# plt.show()

# print('Builtin RK4 rmse =', rmse(
    # real(t, T0), sol.y[0]), 'pri koraku h =', h_rk)


###########################################################################
# Adams-Bashforth-Moulton metoda

def abm(f, t=t, T0=T0, h=h):
    h_abm = h
    t_abm = t

    T_abm = np.zeros_like(t_abm)
    T_abm[0] = T0

    # Začetni koraki z RK4
    for i in range(1, 4):
        k1 = f(t_abm[i-1], T_abm[i-1])
        k2 = f(t_abm[i-1] + h_abm/2, T_abm[i-1] + h_abm/2*k1)
        k3 = f(t_abm[i-1] + h_abm/2, T_abm[i-1] + h_abm/2*k2)
        k4 = f(t_abm[i-1] + h_abm, T_abm[i-1] + h_abm*k3)
        T_abm[i] = T_abm[i-1] + h_abm/6*(k1 + 2*k2 + 2*k3 + k4)

    # Adams-Bashforth-Moulton koraki
    for i in range(4, len(t_abm)):
        # Adams-Bashforth
        T_abm[i] = T_abm[i-1] + h_abm/24 * \
            (55*f(t_abm[i-1], T_abm[i-1]) - 59*f(t_abm[i-2], T_abm[i-2]) +
             37*f(t_abm[i-3], T_abm[i-3]) - 9*f(t_abm[i-4], T_abm[i-4]))

        # Adams-Moulton
        T_abm[i] = T_abm[i-1] + h_abm/24 * \
            (9*f(t_abm[i], T_abm[i]) + 19*f(t_abm[i-1], T_abm[i-1]) -
             5*f(t_abm[i-2], T_abm[i-2]) + f(t_abm[i-3], T_abm[i-3]))

    # plt.plot(t_abm, T_abm, label='Adams-Bashforth-Moulton metoda')

    # print('Adams-Bashforth-Moulton metoda rmse =',
        #   rmse(real(t_abm, T0), T_abm), 'pri koraku h =', h_abm)

    return t_abm, T_abm

##########################################################################
# plt.legend()
# plt.show()


def evaluateMethods(t, T0, h, i=0):
    t_e, T_e = euler(f, t, T0, h)
    t_m, T_m = midpoint(f, t, T0, h)
    t_rk, T_rk = rk4(f, t, T0, h)
    t_abm, T_abm = abm(f, t, T0, h)
    sol = solve_ivp(f, [np.min(t), np.max(t)], [T0], t_eval=t)

    # plt.plot(t, real(t, T0), label='Analitično')
    # plt.plot(t, T_e, label='Euler')
    # # plt.plot(t, T_m, label='Midpoint')
    # # plt.plot(t, T_rk, label='RK4')
    # # plt.plot(sol.t, sol.y[0], label=r'\textit{Scipy}')
    # # plt.plot(t, T_abm, label='Adams-Bashforth-Moulton')
    # plt.xlabel('Čas')
    # plt.ylabel('Temperatura')
    # plt.legend(loc='upper center', fontsize=12, shadow=False, framealpha=1,
    #            fancybox=True, bbox_to_anchor=(0.5, 1.04), ncol=6, edgecolor='black')
    # plt.title('Dobljene funkcije z različnimi metodami', pad=25)
    # plt.tight_layout()
    # plt.show()

    # Plot RSE
    plt.plot(t, np.abs(real(t_e, T0) - T_e),
             label='Euler', color=c[0])
    plt.plot(t, np.abs(real(t_m, T0) - T_m),
             label='Midpoint', color=c[1])
    plt.plot(t, np.abs(real(t_rk, T0) - T_rk), label='RK4', color=c[2])
    plt.plot(sol.t, np.abs(real(t_rk, T0) -
             sol.y[0]), label=r'\textit{Scipy}', color=c[6])
    plt.plot(t_abm, np.abs(real(t_abm, T0) - T_abm),
             label='Adams-Bashforth-Moulton', color=c[4])
    # plt.show()


# evaluateMethods(t, T0, h)
# plt.yscale('log')
# # plt.ylim(1e-2, 0.5)
# plt.xlabel('Čas')
# plt.ylabel('Absolutna Napaka')
# plt.legend(loc='upper center', fontsize=15, shadow=False, framealpha=1,
#            fancybox=True, bbox_to_anchor=(0.5, 1.05), ncol=6, edgecolor='black')
# plt.title('Napaka različnih numeričnih metod', pad=25)
# plt.tight_layout()
# # plt.savefig('6EnacbeHoda/porocilo/napake.pdf')
# plt.show()


# for i in range(2, 12, 2):
#     h = mp.mpf(0.1 + 0.1**i)
#     evaluateMethods(t, T0, h, i)

# plt.yscale('log')
# # plt.ylim(1e-2, 0.5)
# plt.xlabel('Čas')
# plt.ylabel('Absolutna Napaka')
# # plt.legend(loc='upper center', fontsize=15, shadow=False, framealpha=1,
#         #    fancybox=True, bbox_to_anchor=(0.5, 1.05), ncol=6, edgecolor='black')
# plt.title('Napaka različnih numeričnih metod', pad=25)
# plt.tight_layout()
# # plt.savefig('6EnacbeHoda/porocilo/napake.pdf')
# plt.show()


def create_error_progression_plot(t, T0, step_sizes):
    # plt.figure(figsize=(12, 6))

    # Use the provided color palette
    colors = ['#B32D2D', '#2F6E6A', '#D4A94D', '#3F89B4',
              '#A67F5D', '#598959', '#E96C4C', '#358585', '#B2785C']

    # Plot for each step size with a different color
    for i, h in enumerate(step_sizes):
        # Adjust t array for current step size
        t_current = np.arange(0, np.max(t), h)

        # Compute solutions
        _, T_e = euler(f, t_current, T0, h)
        _, T_m = midpoint(f, t_current, T0, h)
        _, T_rk = rk4(f, t_current, T0, h)
        _, T_abm = abm(f, t_current, T0, h)
        sol = solve_ivp(f, [np.min(t), np.max(t)], [T0], t_eval=t_current)

        # Compute absolute errors
        error_e = np.abs(real(t_current, T0) - T_e)
        error_m = np.abs(real(t_current, T0) - T_m)
        error_rk = np.abs(real(t_current, T0) - T_rk)
        error_abm = np.abs(real(t_current, T0) - T_abm)
        error_scipy = np.abs(real(t_current, T0) - sol.y[0])

        # Plot with alpha and color based on step size
        alpha = 1 - (i * 0.6 / len(step_sizes))  # Varying transparency
        plt.semilogy(t_current, error_e, label=f'Euler (h={h:.2e})',
                     color=colors[i], alpha=alpha, markersize=5)
        plt.semilogy(t_current, error_m, label=f'Midpoint (h={h:.2e})',
                     color=colors[i+1], alpha=alpha, markersize=5, linestyle='--')
        plt.semilogy(t_current, error_rk, label=f'RK4 (h={h:.2e})',
                     color=colors[i+2], alpha=alpha, markersize=5, linestyle=':')
        plt.semilogy(t_current, error_abm, label=f'ABM (h={h:.2e})',
                     color=colors[i+3], alpha=alpha, markersize=5, linestyle='-.')
        plt.semilogy(sol.t, error_scipy, label=f'SciPy (h={h:.2e})',
                     color=colors[i+4], alpha=alpha, markersize=5, linestyle=(0, (3, 1, 1, 1, 1, 1)))

    # Format x-axis ticks
    plt.xticks(np.arange(0, max(t)*1.05, 10), rotation=45)

    # Create custom legend
    euler_line = Line2D([], [], color=colors[0],
                        markersize=5)
    midpoint_line = Line2D([], [], color=colors[0],
                           markersize=5, linestyle='--')
    rk4_line = Line2D([], [], color=colors[0], markersize=5,
                      linestyle=':')
    abm_line = Line2D([], [], color=colors[0], markersize=5,
                      linestyle='-.')
    scipy_line = Line2D([], [], color=colors[0],
                        markersize=5, linestyle=(0, (3, 1, 1, 1, 1, 1)))
    first_legend = plt.legend([euler_line, midpoint_line, rk4_line, abm_line, scipy_line], [
                              'Euler', 'Midpoint', 'RK4', 'SciPy', 'ABM'], fontsize=12, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=6, framealpha=1, edgecolor='black')
    plt.gca().add_artist(first_legend)

    # Add second legend for step sizes
    step_size_lines = [Line2D([], [], color=colors[i], markersize=5,
                              linestyle='-') for i in range(len(step_sizes))]
    step_sizes_str = [f'h={h:.1f}' for h in step_sizes]
    plt.legend(step_size_lines, step_sizes_str, fontsize=12,
               loc='center right', bbox_to_anchor=(1.1, 0.5), ncol=1, framealpha=1, edgecolor='black')

    # Axis labels
    plt.xlabel('Čas')
    plt.ylabel('Absolutna napaka')

    # Plot title
    plt.title('Napaka različnih metod pri spreminjanju koraka', pad=30)

    plt.tight_layout()
    # plt.savefig('6EnacbeHoda/porocilo/hugekorak.pdf')
    plt.show()


# Define step sizes
step_sizes = [(0.1 + 0.1**i) for i in range(2, 9, 2)]
step_sizes = [0.1, 0.01, 0.001, 0.0001]
step_sizes = [1, 10, 33]

# Create the visualization
create_error_progression_plot(t, T0, step_sizes)
