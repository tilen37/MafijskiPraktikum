from mpmath import mp, airyai, airybi
from matplotlib import pyplot as plt
import numpy as np

mp.dps = 52

# alpha = mp.mpf('0.355028053887817239')
alpha = airyai(0)
# beta = mp.mpf('0.258819403792806798')
beta = - airyai(0, derivative=1)
epsilon = mp.mpf('1e-50')


def f_rec(x):
    s = 0

    def a_k(x):
        k = 0
        new = 1
        while True:
            yield new
            k += 1
            old = new
            new = old * x**3 / (3*k * (3*k - 1))

    n = 0
    for a in a_k(x):
        s += a
        n += 1
        if n > 200:
            break
        if abs(a) < epsilon:
            break
    return s


def g_rec(x):
    s = 0

    def b_k(x):
        k = 0
        new = x
        while True:
            yield new
            k += 1
            old = new
            new = old * x**3 / (3*k * (3*k + 1))

    n = 0
    for b in b_k(x):
        n += 1
        s += b
        if n > 200:
            break
        if abs(b) < epsilon:
            break
    return s


def Ai_taylor(x):
    return alpha * f_rec(x) - beta * g_rec(x)


def Bi_taylor(x):
    return mp.sqrt(3) * (alpha * f_rec(x) + beta * g_rec(x))


if __name__ == '__main__':
    line_taylor = np.linspace(-40, 29, 30)
    line_taylor = [mp.mpf(x) for x in line_taylor]

    mineAiTaylor = [Ai_taylor(x) for x in line_taylor]
    libAiTaylor = [airyai(x) for x in line_taylor]
    aerrAiTaylor = [abs(mineAiTaylor[i] - libAiTaylor[i])
                    for i in range(len(mineAiTaylor))]
    rerrAiTaylor = [aerrAiTaylor[i] / abs(libAiTaylor[i])
                    for i in range(len(mineAiTaylor))]

    mineBiTaylor = [Bi_taylor(x) for x in line_taylor]
    libBiTaylor = [airybi(x) for x in line_taylor]
    aerrBiTaylor = [abs(mineBiTaylor[i] - libBiTaylor[i])
                    for i in range(len(mineBiTaylor))]
    rerrBiTaylor = [aerrBiTaylor[i] / abs(libBiTaylor[i])
                    for i in range(len(mineBiTaylor))]

    # x_0 = 1
    # print(Ai(x_0), airyai(x_0))

    # plt.plot(line_taylor, mineAiTaylor, c='r', linestyle='dashed')
    # plt.plot(line_taylor, libAiTaylor, c='g')
    # # plt.plot(line_taylor, mineBiTaylor, c='r', linestyle='dashed')
    # # plt.plot(line_taylor, libBiTaylor, c='g')
    # plt.legend(['Mine', 'Library'])
    # plt.show()

    plt.scatter(line_taylor, aerrAiTaylor, c='C0',
                label='Absolutna napaka Ai', s=15)
    plt.scatter(line_taylor, rerrAiTaylor, c='C2',
                label='Relativna napaka Ai', s=15)
    plt.scatter(line_taylor, aerrBiTaylor, c='C3',
                label='Absolutna napaka Bi', s=15)
    plt.scatter(line_taylor, rerrBiTaylor, c='C5',
                label='Relativna napaka Bi', s=15)
    plt.plot(line_taylor, [1e-10 for _ in line_taylor], c='gray',
             linestyle='dashed', label='Mejna vrednost')
    plt.title('Vrednosti napak pri Taylorjevem razvoju')
    plt.ylabel('Vrednost napake')
    plt.xlabel('x')
    plt.yscale('log')
    plt.legend()
    plt.show()
    # Sprejemljivo na Ai:[-28, 17], Bi:[-28, 27.5]
