from mpmath import mp, airyai, airybi
from matplotlib import pyplot as plt
import numpy as np


def P_s(x):
    result = 0

    def b_k(x):
        k = 0
        new = 1
        while True:
            yield new
            k += 1
            old = new
            new = -(6*k - mp.mpf(1)/2) * (6*k - mp.mpf(3)/2) * (6*k - mp.mpf(5)/2) * \
                (6*k - mp.mpf(7)/2) * (6*k - mp.mpf(9)/2) * (6*k - mp.mpf(11)/2) / \
                54**2 / (2*k) / (2*k - 1) / (2*k - mp.mpf(1)/2) / \
                (2*k - mp.mpf(3)/2) / x**2 * old
            if abs(old) < abs(new):
                break

    for b in b_k(x):
        result += b
    return result


def Q_s(x):
    result = 0

    def c_k(x):
        k = 0
        new = mp.mpf(5) / 72 / x
        while True:
            yield new
            k += 1
            old = new
            new = -(6*k + 3 - mp.mpf(1)/2) * (6*k + 3 - mp.mpf(3)/2) * (6*k + 3 - mp.mpf(5)/2) * (6*k + 3 - mp.mpf(7)/2) * (6*k + 3 - mp.mpf(
                9)/2) * (6*k + 3 - mp.mpf(11)/2) / 54**2 / (2*k + 1) / (2*k) / (2*k + 1 - mp.mpf(1)/2) / (2*k + 1 - mp.mpf(3)/2) / x**2 * old
            if abs(old) < abs(new):
                break

    for c in c_k(x):
        result += c
    return result


def Ai_asimptotic_n(x):
    ksi = 2/3 * abs(x)**(3/2)
    return 1/(mp.sqrt(mp.pi * mp.sqrt(-x))) * (Q_s(ksi) * mp.sin(ksi - mp.pi/4) + P_s(ksi) * mp.cos(ksi - mp.pi/4))


def Bi_asimptotic_n(x):
    ksi = 2/3 * abs(x)**(3/2)
    return 1/(mp.sqrt(mp.pi * mp.sqrt(-x))) * (- P_s(ksi) * mp.sin(ksi - mp.pi/4) + Q_s(ksi) * mp.cos(ksi - mp.pi/4))


if __name__ == '__main__':
    print('Ai asimptotic', Ai_asimptotic_n(-100))
    print('Ai asimptotic', Ai_asimptotic_n(-10))

    line_asimptotic_n = np.linspace(-100, -2, 50)
    line_asimptotic_n = [mp.mpf(x) for x in line_asimptotic_n]

    mineAiAsimptotic_n = [Ai_asimptotic_n(x) for x in line_asimptotic_n]
    libAiAsimptotic_n = [airyai(x) for x in line_asimptotic_n]
    aerrAiAsimptotic_n = [abs(mineAiAsimptotic_n[i] - libAiAsimptotic_n[i])
                          for i in range(len(mineAiAsimptotic_n))]
    rerrAiAsimptotic_n = [aerrAiAsimptotic_n[i] / abs(libAiAsimptotic_n[i])
                          for i in range(len(mineAiAsimptotic_n))]

    mineBiAsimptotic_n = [Bi_asimptotic_n(x) for x in line_asimptotic_n]
    libBiAsimptotic_n = [airybi(x) for x in line_asimptotic_n]
    aerrBiAsimptotic_n = [abs(mineBiAsimptotic_n[i] - libBiAsimptotic_n[i])
                          for i in range(len(mineBiAsimptotic_n))]
    rerrBiAsimptotic_n = [aerrAiAsimptotic_n[i] / abs(libBiAsimptotic_n[i])
                          for i in range(len(mineBiAsimptotic_n))]

    # plt.plot(line_asimptotic_n, libAiAsimptotic_n, label='lib Ai', linestyle='-.')
    # plt.plot(line_asimptotic_n, mineAiAsimptotic_n,
    #          label='mine Ai', linestyle='--')
    # plt.plot(line_asimptotic_n, libBiAsimptotic_n, label='lib Bi', linestyle='-.')
    # plt.plot(line_asimptotic_n, mineBiAsimptotic_n,
    #          label='mine Bi', linestyle='--')
    plt.plot(line_asimptotic_n, [1e-10 for _ in line_asimptotic_n],
             c='black', linestyle='dashed', label='Mejhna vrednost')
    plt.scatter(line_asimptotic_n, aerrAiAsimptotic_n,
                label='Ai Absolutna napaka', s=15)
    plt.scatter(line_asimptotic_n, rerrAiAsimptotic_n,
                label='Ai Relativna napaka', s=15)
    plt.scatter(line_asimptotic_n, aerrBiAsimptotic_n,
                label='Bi Absolutna napaka', s=15)
    plt.scatter(line_asimptotic_n, rerrBiAsimptotic_n,
                label='Bi Relativna napaka', s=15)
    plt.legend()
    plt.yscale('log')
    plt.title('Vrednosti napak pri asimptotskem razvoju za negativne x')
    plt.show()
