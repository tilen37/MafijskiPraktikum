import numpy as np
import time
from numpy import linspace
from memory_profiler import profile
import matplotlib.pyplot as plt
from numpy import fft
from functions import DFT

timesFFT = []
timesDFT = []

def timeComplexity(h, N, M):
    t = linspace(-1, 1, N)
    ti = linspace(-1, 1, M)

    x = h(t)
    x = x.reshape(-1, 1)

    time0 = time.time()
    X = fft.fft(x, axis=0)
    time1 = time.time()
    X = DFT(x)
    time2 = time.time()

    return time1 - time0, time2 - time1

@profile
def main():
    for i in range(1, 1000, 10):
        print(i // 10)
        num_freq = np.random.randint(1, 50)

        def h(x): return sum([A * np.sin(omega * x) for omega, A in zip(
            [i for i in range(num_freq)], np.random.uniform(0.1, 1, num_freq))])

        timeFFT, timeDFT = timeComplexity(h, i, 2 * i)
        timesFFT.append(timeFFT)
        timesDFT.append(timeDFT)


if __name__ == "__main__":
    main()