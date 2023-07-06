import numpy as np
from struct import unpack

def real_time_series():#Выбор первых 1500 отсчетов пульсового сигнала
    file = open('d1000','br')
    signal=np.flip(np.array(unpack('10000H',file.read())))
    return signal[:1500]

def lorenz():#Выбор первой коориданты системы
    def solve_lorenz(x, y, z, sigma=10, rho=28, beta=8.0/3.0, dt=0.01):
        x_dot = sigma * (y - x)
        y_dot = x * (rho - z) - y
        z_dot = x * y - beta * z
        x += x_dot * dt
        y += y_dot * dt
        z += z_dot * dt
        return x, y, z
    xs, ys, zs = np.empty((3, 1500 + 1))
    xs[0], ys[0], zs[0] = (0., 1., 1.05)
    for i in range(1500):
        xs[i+1], ys[i+1], zs[i+1] = solve_lorenz(xs[i], ys[i], zs[i])
    return xs

def ressler():#Выбор первой коориданты системы
    def solve_ressler(x, y, z, a=0.2, b=0.2, c=5.7, dt=0.08):
        x_dot = -y - z
        y_dot = x + a*y
        z_dot = b + z*(x-c)
        x += x_dot * dt
        y += y_dot * dt
        z += z_dot * dt
        return x, y, z
    xs, ys, zs = np.empty((3, 1500 + 1))
    xs[0], ys[0], zs[0] = (0.5, 0.5, 0.5)
    for i in range(1500):
        xs[i+1], ys[i+1], zs[i+1] = solve_ressler(xs[i], ys[i], zs[i])
    return xs
