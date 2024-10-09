from mpmath import mp, airyai, airybi
from matplotlib import pyplot as plt
import numpy as np

mp.dps = 100


def L_s(x):
    result = mp.mpf(0)

    def a_k(x):
        k = mp.mpf('0')
        new = mp.mpf(1.)
        yield new
        while True:
            k += 1
            old = new
            new = old * (3*k - mp.mpf(1)/2) * (3*k - mp.mpf(5)/2) / 18 / k / x
            if abs(old) < abs(new):
                break
            yield new

    for a in a_k(x):
        result += a
    return result


def Ai_asimptotic_p(x):
    ksi = mp.mpf(2)/3 * abs(x)**(mp.mpf(3)/2)
    return mp.exp(-ksi) / (2 * mp.sqrt(mp.pi) * x**mp.mpf('0.25')) * L_s(-ksi)


def Bi_asimptotic_p(x):
    ksi = mp.mpf(2) / 3 * abs(x)**(mp.mpf(3)/2)
    return mp.exp(ksi) / (mp.sqrt(mp.pi * mp.sqrt(x))) * L_s(ksi)


if __name__ == '__main__':

    line_asimptotic_p = np.linspace(5, 100, 50)
    line_asimptotic_p = [mp.mpf(x) for x in line_asimptotic_p]

    mineAiAsimptotic_p = [Ai_asimptotic_p(x) for x in line_asimptotic_p]
    libAiAsimptotic_p = [airyai(x) for x in line_asimptotic_p]
    aerrAiAsimptotic_p = [abs(mineAiAsimptotic_p[i] - libAiAsimptotic_p[i])
                          for i in range(len(mineAiAsimptotic_p))]
    rerrAiAsimptotic_p = [aerrAiAsimptotic_p[i] / abs(libAiAsimptotic_p[i])
                          for i in range(len(mineAiAsimptotic_p))]

    mineBiAsimptotic_p = [Bi_asimptotic_p(x) for x in line_asimptotic_p]
    libBiAsimptotic_p = [airybi(x) for x in line_asimptotic_p]
    aerrBiAsimptotic_p = [abs(mineBiAsimptotic_p[i] - libBiAsimptotic_p[i])
                          for i in range(len(mineBiAsimptotic_p))]
    rerrBiAsimptotic_p = [aerrAiAsimptotic_p[i] / abs(libBiAsimptotic_p[i])
                          for i in range(len(mineBiAsimptotic_p))]

    print('Ai asimptotic', mineAiAsimptotic_p[-1])

    # plt.plot(line_asimptotic_p, libAiAsimptotic_p, label='lib Ai', linestyle='-.')
    # plt.plot(line_asimptotic_p, mineAiAsimptotic_p,
    #          label='mine Ai', linestyle='--')
    # plt.plot(line_asimptotic_p, libBiAsimptotic_p, label='lib Bi', linestyle='-.')
    # plt.plot(line_asimptotic_p, mineBiAsimptotic_p,
    #          label='mine Bi', linestyle='--')

    plt.plot(line_asimptotic_p, [1e-10 for _ in line_asimptotic_p],
             c='black', linestyle='dashed', label='Mejhna vrednost')
    plt.scatter(line_asimptotic_p, aerrAiAsimptotic_p,
                label='Ai Absolutna napaka', s=15)
    plt.scatter(line_asimptotic_p, rerrAiAsimptotic_p,
                label='Ai Relativna napaka', s=15)  # Not good enough
    plt.scatter(line_asimptotic_p, aerrBiAsimptotic_p,
                label='Bi Absolutna napaka', s=15)  # Not good enough
    plt.scatter(line_asimptotic_p, rerrBiAsimptotic_p,
                label='Bi Relativna napaka', s=15)
    plt.yscale('log')
    plt.legend()
    plt.title('Vrednosti napak pri asimptotskem razvoju za pozitivne x')
    plt.show()
