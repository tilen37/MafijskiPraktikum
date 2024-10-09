from asimptotic_p import Ai_asimptotic_p, Bi_asimptotic_p
from asimptotic_n import Ai_asimptotic_n, Bi_asimptotic_n
from taylor import Ai_taylor, Bi_taylor
from mpmath import mp, airyai, airybi
from matplotlib import pyplot as plt
import numpy as np
import time

mp.dps = 100


def Ai(x):
    if x > 10:
        return Ai_asimptotic_p(x)
    if x < -28:
        return Ai_asimptotic_n(x)
    return Ai_taylor(x)


def Bi(x):
    if x > 26:
        return Bi_asimptotic_p(x)
    if x < -28:
        return Bi_asimptotic_n(x)
    return Bi_taylor(x)


line = np.linspace(-40, 5, 1000)
line = [mp.mpf(x) for x in line]

startAi = time.time()
mineAi = [Ai(x) for x in line]
endAi = time.time()
print('Ai', endAi - startAi)
libAi = [airyai(x) for x in line]
startBi = time.time()
mineBi = [Bi(x) for x in line]
endBi = time.time()
print('Bi', endBi - startBi)
libBi = [airybi(x) for x in line]

aerrAi = [abs(mineAi[i] - libAi[i]) for i in range(len(line))]
rerrAi = [aerrAi[i] / abs(libAi[i]) for i in range(len(line))]

aerrBi = [abs(mineBi[i] - libBi[i]) for i in range(len(line))]
rerrBi = [aerrBi[i] / abs(libBi[i]) for i in range(len(line))]

# plt.plot(line, mineAi, c='r', linestyle='dashed')
# plt.plot(line, libAi, c='g')
# # plt.plot(line, mineBi, c='r', linestyle='dashed')
# # plt.plot(line, libBi, c='g')
# plt.legend(['Mine', 'Library'])
# plt.show()


plt.plot(line, mineAi, label='Ai')
plt.plot(line, mineBi, label='Bi')
# plt.scatter(line, aerrAi, marker='x', c='C0',
#             label='Absolutna napaka Ai', s=15)
# plt.scatter(line, rerrAi, marker='o', c='C2',
#             label='Relativna napaka Ai', s=15)
# plt.scatter(line, aerrBi, marker='s', c='C3',
#             label='Absolutna napaka Bi', s=15)
# plt.scatter(line, rerrBi, marker='v', c='C5',
#             label='Relativna napaka Bi', s=15)
# plt.plot(line, [1e-10 for _ in line], c='gray',
#          linestyle='dashed', label='Mejna vrednost')
plt.title('Airyevi funkciji')
plt.ylabel('f(x)')
plt.xlabel('x')
# plt.yscale('log')
plt.ylim([-0.7, 2])
plt.legend()
plt.show()

# Time
# 500dps: Ai: 2.98, Bi: 2.98
# 200dps: Ai: 2.50, Bi: 2.38
# 100dps: Ai
