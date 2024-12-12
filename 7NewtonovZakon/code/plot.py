import numpy as np
import matplotlib.pyplot as plt
plt.style.use('customStyle.mplstyle')

measured = np.array([10, 562, 31622, 1778279, 100000000])

t = np.arange(0, int(1e7), 0.1)
t0 = t
t1 = t[t < 5]
t2 = t[t < 1e3]
t3 = t[t < 5e5]
t4 = t[t < 1e7]
t5 = t[t < 1e7]

limits = [len(t0), len(t0), len(t1), len(t2), len(t3), len(t4), len(t5)]
mtimes = np.load(r'7NewtonovZakon/code/measured_times.npy')

for times, label, limit in zip(mtimes,
                               ['Analitična', 'SciPy', 'Euler',
                                'Midpoint', 'RK4', 'Verlet', 'PEFRL'],
                               limits):

    # Find the first point that exceeds our limit
    split_points = np.where(measured > limit)[0]
    if len(split_points) > 0:
        split_idx = split_points[0]
        # Plot solid line up to split point
        line = plt.plot(measured[:split_idx],
                        times[:split_idx], '-', label=label)[0]
        color = line.get_color()
        # Plot dashed line after split point
        plt.plot(measured[split_idx-1:],
                 times[split_idx-1:], '--', color=color)
    else:
        # If no split point found, plot entire line solid
        plt.plot(measured, times, '-', label=label)

plt.xlabel(r'Število korakov')
plt.ylabel(r'$t_{\rm izvajanja}$')
plt.legend(fontsize=15)
plt.xscale('log')
plt.yscale('log')
plt.title('Časovna zahtevnost različnih metod')
plt.savefig('7NewtonovZakon/porocilo/bigO.pdf')
plt.show()
