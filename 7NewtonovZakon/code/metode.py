from typing import Callable, Any
import numpy as np


def euler(f: Callable[[np.ndarray, float, Any], np.ndarray], y0: np.ndarray, t: np.ndarray, *args) -> np.ndarray:
    '''Metoda Euler za reševanje diferencialnih enačb.

    :param f: Funkcija, ki opisuje diferencialno enačbo.
    :param y0: Začetni pogoji.
    :param t: Časovni vektor.
    :return: Rešitev diferencialne enačbe.
    '''

    y = np.zeros((len(t), len(y0)))
    y[0] = y0

    for i in range(1, len(t)):
        y[i] = y[i-1] + f(y[i-1], t[i-1], *args) * (t[i] - t[i-1])
    return y


def mid(f: Callable[[np.ndarray, float, Any], np.ndarray], y0: np.ndarray, t: np.ndarray, *args) -> np.ndarray:
    '''Metoda sredinske točke za reševanje diferencialnih enačb.

    :param f: Funkcija, ki opisuje diferencialno enačbo.
    :param y0: Začetni pogoji.
    :param t: Časovni vektor.
    :return: Rešitev diferencialne enačbe.
    '''

    y = np.zeros((len(t), len(y0)))
    y[0] = y0

    for i in range(1, len(t)):
        y[i] = y[i-1] + f(y[i-1] + f(y[i-1], t[i-1], *args) * (t[i] - t[i-1]) /
                          2, t[i-1] + (t[i] - t[i-1]) / 2, *args) * (t[i] - t[i-1])
    return y


def rk4(f: Callable[[np.ndarray, float, Any], np.ndarray], y0: np.ndarray, t: np.ndarray, *args) -> np.ndarray:
    '''Runge-Kutta 4. reda za reševanje diferencialnih enačb.

    :param f: Funkcija, ki opisuje diferencialno enačbo.
    :param y0: Začetni pogoji.
    :param t: Časovni vektor.
    :return: Rešitev diferencialne enačbe.
    '''

    y = np.zeros((len(t), len(y0)))
    y[0] = y0

    for i in range(1, len(t)):
        h = t[i] - t[i-1]
        k1 = f(y[i-1], t[i-1], *args)
        k2 = f(y[i-1] + k1 * h / 2, t[i-1] + h / 2, *args)
        k3 = f(y[i-1] + k2 * h / 2, t[i-1] + h / 2, *args)
        k4 = f(y[i-1] + k3 * h, t[i-1] + h, *args)
        y[i] = y[i-1] + (k1 + 2*k2 + 2*k3 + k4) * h / 6
    return y

# ----------------------------------------------------------------------------------------------------------------
# Methods from instructions


import numpy as np
from typing import Callable, Any

def verlet(a_func: Callable[[float, float, float, Any], float], y0: np.ndarray, t: np.ndarray, *args) -> np.ndarray:
    '''
    Modified Verlet integration method for second-order ODEs with velocity-dependent forces.

    :param a_func: Function that returns acceleration given position x, velocity v, time t, and additional arguments.
                   It should have the signature: a_func(x: float, v: float, t: float, *args) -> float
    :param y0: Initial conditions [x0, v0], a NumPy array with shape (2,)
    :param t: Time array
    :param args: Additional arguments to pass to a_func
    :return: NumPy array with shape (len(t), 2), containing positions and velocities at each time step
    '''
    n = len(t)
    dt = t[1] - t[0]  # Assuming uniform time steps

    x = np.zeros(n)
    v = np.zeros(n)
    x[0], v[0] = y0

    # Compute initial acceleration
    a0 = a_func(y0, t[0], *args)[1]

    for i in range(n - 1):
        # Update position
        x[i + 1] = x[i] + v[i] * dt + 0.5 * a0 * dt ** 2

        # Estimate velocity at half-step
        v_half = v[i] + 0.5 * a0 * dt

        # Compute acceleration at new position and estimated half-step velocity
        a1 = a_func(np.array([x[i + 1], v_half]), t[i + 1], *args)[1]

        # Update velocity
        v[i + 1] = v_half + 0.5 * a1 * dt

        # Prepare for next iteration
        a0 = a1

    return np.column_stack((x, v))

# def verlet(f: Callable[[np.ndarray, float, Any], np.ndarray], y0: np.ndarray, t: np.ndarray, *args) -> np.ndarray:
#     """Verlet's 2nd order symplectic method

#     USAGE:
#         (x,v) = varlet(f, y0, t)

#     INPUT:
#         f     - function of x and t equal to d^2x/dt^2.  x may be multivalued,
#                 in which case it should a list or a NumPy array.  In this
#                 case f must return a NumPy array with the same dimension
#                 as x.
#         y0    - the initial condition(s) of x and v=dx/dt. Specifies the value of x and v when
#                 t = t[0]. It should be a 1D or 2D NumPy array.
#         t     - list or NumPy array of t values to compute solution at.
#                 t[0] is the the initial condition point, and the difference
#                 h=t[i+1]-t[i] determines the step size h.

#     OUTPUT:
#         x     - NumPy array containing solution values for x corresponding to each
#                 entry in t array.  If a system is being solved, x will be
#                 an array of arrays.
#         v     - NumPy array containing solution values for v=dx/dt corresponding to each
#                 entry in t array.  If a system is being solved, x will be
#                 an array of arrays.

#     NOTES:
#         This function used the Varlet/Stoermer/Encke (symplectic) method
#         method to solve the initial value problem

#             dx^2
#             -- = f(x),     x(t(1)) = x0  v(t(1)) = v0
#             dt^2

#         at the t values stored in the t array (so the interval of solution is
#         [t[0], t[N-1]].  The 3rd-order Taylor is used to generate
#         the first values of the solution.

#     """
#     n = len(t)
#     x = np.array([y0[0]] * n)
#     v = np.array([y0[1]] * n)
#     for i in range(n - 1):
#         h = t[i+1] - t[i]
#         x[i+1] = x[i] + h * v[i] + (h*h/2) * f(x[i], t[i], *args)
#         v[i+1] = v[i] + (h/2) * (f(x[i], *args)+f(x[i+1], t[i], *args))

#     return np.array([x, v])


def pefrl(f: Callable[[np.ndarray, float, Any], np.ndarray], y0: np.ndarray, t: np.ndarray, *args) -> np.ndarray:
    """Position Extended Forest-Ruth Like algorithm for 2nd order ODEs
    where acceleration can depend on both position and velocity"""
    
    # Coefficients from the paper
    xsi = 0.1786178958448091
    lam = -0.2123418310626054
    chi = -0.6626458266981849e-1

    # Initialize arrays
    n = len(t)
    y = np.zeros((n, len(y0)))
    y[0] = y0

    for i in range(1, n):
        dt = t[i] - t[i-1]
        x, v = y[i-1]  # Current position and velocity
        
        # First substep
        x += xsi * dt * v
        a = f(np.array([x, v]), t[i-1], *args)[1]
        v += (1-2*lam) * (dt/2) * a
        
        # Second substep
        x += chi * dt * v
        a = f(np.array([x, v]), t[i-1], *args)[1]
        v += lam * dt * a
        
        # Third substep
        x += (1-2*(chi+xsi)) * dt * v
        a = f(np.array([x, v]), t[i-1], *args)[1]
        v += lam * dt * a
        
        # Fourth substep
        x += chi * dt * v
        a = f(np.array([x, v]), t[i-1], *args)[1]
        v += (1-2*lam) * (dt/2) * a
        
        # Final substep
        x += xsi * dt * v
        
        y[i] = np.array([x, v])

    return y
