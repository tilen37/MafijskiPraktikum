import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Computer Modern Serif",
    "font.size": 14
})

num_freq = 30


def h(x): return sum([A * np.sin(omega * x) for omega, A in zip(
    [i for i in range(num_freq)], np.random.uniform(0.9, 1, num_freq))])


# def h(x, T=2):
#     x_mod = np.mod(x + 1, T) - 1    # Periodicno ponavljanje
#     x_mod = x                       # Brez periodicnega ponavljanja
#     return np.exp(-x_mod**2 / 0.05)


line = np.linspace(-1, 1, 10000)

samples = 50
t0 = (line[-1] - line[0]) / samples
print(t0)
nyquist = 1/(2*t0)
print('Nyquist', nyquist)


# def h(x):
#     return np.sin((nyquist*0.5)*x)


x = np.linspace(-1, 1, samples)
y = h(x)

Y = np.fft.fft(y, axis=0)
# Y = np.roll(Y, len(Y)//2)

x2 = np.linspace(-1, 1, samples//2)
x3 = np.linspace(-1, 1, samples//4)
y2 = np.fft.ifft(Y[:samples//2], axis=0)
y2 = np.fft.fftshift(y[::2])
y2 = np.roll(y2, len(y2)//2)

plt.plot(line, h(line), label='Original', c='r')
plt.plot(x2, y2, label='Rekonstrukcija', c='C0')
plt.legend(loc='upper center', fontsize=12, shadow=True,
           fancybox=True, bbox_to_anchor=(0.5, 1.07), ncol=2)
plt.grid(alpha=0.5)
plt.title('Rekonstrukcija signala', pad=23)
plt.savefig(r'4FourierovaAnaliza/porocilo/figs/mix_over.png')
plt.show()

# fig, ax = plt.subplots(3, 2, figsize=(20, 10))
# ax[0, 0].plot(line, h(line))
# ax[1, 0].plot(x, np.real(Y))
# ax[2, 0].plot(x, np.imag(Y))
# ax[0, 1].plot(x2, y2)
# ax[1, 1].plot(x2, np.real(y2))
# ax[2, 1].plot(x2, np.imag(y2))
# plt.show()
