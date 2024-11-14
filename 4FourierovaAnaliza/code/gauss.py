import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Computer Modern Serif",
    "font.size": 14
})


def h(x, T=2):
    x_mod = np.mod(x + 1, T) - 1    # Periodicno ponavljanje
    x_mod = x                       # Brez periodicnega ponavljanja
    return np.exp(-x_mod**2 / 0.05)


samples = 1000

x = np.linspace(-1, 1, samples)
y = h(x)

Y = np.fft.fft(y)
Y = np.fft.fftshift(Y)

freqs = np.fft.fftfreq(samples, d=(x[1] - x[0]))
freqs = np.fft.fftshift(freqs)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(x, y, label='Original Gaussovka', c='r')
plt.xlabel('$x$')
plt.ylabel('$A$')
plt.title('Originalna periodična Gaussova funkcija', pad=25)
plt.grid(True)
plt.legend(loc='upper center', fontsize=12, shadow=True,
           fancybox=True, bbox_to_anchor=(0.5, 1.07), ncol=2)

plt.subplot(1, 2, 2)
plt.plot(freqs, np.abs(Y), label='FFT Gaussovke', c='r')
plt.xlabel('$f$')
plt.ylabel('$A$')
plt.title('FFT periodične Gaussove funkcije', pad=25)
plt.grid(True)
plt.xlim(-50, 50)
plt.legend(loc='upper center', fontsize=12, shadow=True,
           fancybox=True, bbox_to_anchor=(0.5, 1.07), ncol=2)

plt.tight_layout()
plt.savefig('gauss.png')
# plt.show()
