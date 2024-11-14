# https://dsp.stackexchange.com/questions/60362/aliasing-when-interpolating-with-dft 13-11-2024

from numpy import arange, exp, pi, dot


def DFT(x):
    N = len(x)
    n = arange(N)
    k = n.reshape((N, 1))
    M = exp(-2j * pi * k * n / N)
    return dot(M, x) / N


def IDFT(X):
    N = len(X)
    n = arange(N)
    k = n.reshape((N, 1))
    M = exp(2j * pi * k * n / N)
    return dot(M, X) / N


if __name__ == '__main__':
    from time import time
    import matplotlib.pyplot as plt
    from numpy import arange, exp, pi, dot, linspace, sin, fft, outer, real

    N = 100
    t = linspace(-1, 1, N)

    print('Nyquist frequency:', 1/(2*(t[1] - t[0])))

    M = 200
    ti = linspace(-1, 1, M)

    # x = t * (t - 0.8) * (t + 1)
    x = sin(2 * pi * 4.7 * t)
    # x = exp(-t**2 / 0.01)
    x = x.reshape(-1, 1)

    time0 = time.time()
    X = fft.fft(x, axis=0)
    time1 = time.time()
    X = DFT(x)
    time2 = time.time()

    print('FFT:', time1 - time0)
    print('DFT:', time2 - time1)

    X = X.reshape(-1, 1)

    Nc = 40  # Number of Fourier coefficients, should not exceed N / 2

    # x2 = IDFT(X[:Nc])

    exp_term = exp(1j * 2 * pi / M *
                   outer(arange(M), arange(0, Nc)))
    x2 = X[0] + 2 * real(exp_term @ X[:Nc])
    x2 = x2.flatten() / N

    x3 = fft.ifft(X[:Nc], n=M, axis=0)

    # print(exp_term.shape)
    # print(X[1:Nc].shape)
    # print(x.shape)
    # print(x2.shape)
    # print(x3.shape)

    fig, ax = plt.subplots(3, 1)

    ax.flat[0].plot(t, x, label='Original', c='r')
    ax.flat[1].plot(ti, x2, label='Reconstructed', c='r')
    ax.flat[2].plot(ti, x3, label='Reconstructed', c='r')

    plt.show()
