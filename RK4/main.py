import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# определение правых частей системы уравнений
def f(t, y, z):
    return y / (y + z), z / (y + z)

# определенре правой части для встроенного метода
def f_for_embedded(t, yz):
    y, z = yz
    return [y / (y + z), z / (y + z)]

# метод РК4 (Рунге - Кутты) четвертого порядка для решения системы
def RK4(f, x0, y0, z0, h, n):
    x = [x0]
    y = [y0]
    z = [z0]

    for i in range(n):
        xi = x[i]
        yi = y[i]
        zi = z[i]

        k1y, k1z = f(xi, yi, zi)
        k2y, k2z = f(xi + h/2, yi + h/2*k1y, zi + h/2*k1z)
        k3y, k3z = f(xi + h/2, yi + h/2*k2y, zi + h/2*k2z)
        k4y, k4z = f(xi + h, yi + h*k3y, zi + h*k3z)

        y_next = yi + h/6 * (k1y + 2*k2y + 2*k3y + k4y)
        z_next = zi + h/6 * (k1z + 2*k2z + 2*k3z + k4z)

        x.append(xi + h)
        y.append(y_next)
        z.append(z_next)

    return x, y, z

# Исходные данные
x0 = 0
y0 = 2
z0 = 4
xk = 8.0
delta_x = 0.4

# Вычисление шага
n = int((xk - x0) / delta_x)
h = (xk - x0) / n

# Численное интегрирование
x, y, z = RK4(f, x0, y0, z0, h, n)

# решаем с использованием solve_ivp и RK45
sol = solve_ivp(f_for_embedded, (x0, xk), [y0, z0], method='RK45', max_step=delta_x)

x_ivp = sol.t
y_ivp, z_ivp = sol.y

# Вывод результатов
table = np.column_stack((x, y, z))
print("Таблица решений:")
print("x\t\ty\t\tz")
print("------------------------")
for row in table:
    print("\t".join(str(val) for val in row))

print(len(table))

# график y(x)
def graph1():
    plt.figure()
    plt.plot(x, y, label='y(x) - RK4')
    plt.plot(x_ivp, y_ivp, label='y(x) - RK45', linestyle='dashed')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('График y(x)')
    plt.legend()
    plt.grid(True)
    plt.show()

# график z(x)
def graph2():
    plt.figure()
    plt.plot(x, z, label='z(x) - RK4')
    plt.plot(x_ivp, z_ivp, label='z(x) - RK45', linestyle='dashed')
    plt.xlabel('x')
    plt.ylabel('z')
    plt.title('График z(x)')
    plt.legend()
    plt.grid(True)
    plt.show()

# график y(x) и z(x)
def graph3():
    plt.figure()
    plt.plot(x, y, label='y(x) - RK4')
    plt.plot(x, z, label='z(x) - RK4')
    plt.plot(x_ivp, y_ivp, label='y(x) - RK45', linestyle='dashed')
    plt.plot(x_ivp, z_ivp, label='z(x) - RK45', linestyle='dashed')
    plt.xlabel('x')
    plt.ylabel('y, z')
    plt.title('Решение системы уравнений')
    plt.legend()
    plt.grid(True)
    plt.show()

# график y(x) в полярных координатах
def polar_graph1():
    plt.figure()
    plt.polar(x, y, label='y(x) - RK4')
    plt.polar(x_ivp, y_ivp, label='y(x) - RK45', linestyle='dashed')
    plt.title('График y(x) в полярных координатах')
    plt.legend()
    plt.grid(True)
    plt.show()

# график z(x) в полярных координатах
def polar_graph2():
    plt.figure()
    plt.polar(x, z, label='z(x) - RK4')
    plt.polar(x_ivp, z_ivp, label='z(x) - RK45', linestyle='dashed')
    plt.title('График z(x) в полярных координатах')
    plt.legend()
    plt.grid(True)
    plt.show()

# график y(x) и z(x) в полярных координатах
def polar_graph3():
    plt.figure()
    plt.polar(x, y, label='y(x) - RK4')
    plt.polar(x, z, label='z(x) - RK4')
    plt.polar(x_ivp, y_ivp, label='y(x) - RK45', linestyle='dashed')
    plt.polar(x_ivp, z_ivp, label='z(x) - RK45', linestyle='dashed')
    plt.title('Решение системы уравнений в полярных координатах')
    plt.legend()
    plt.grid(True)
    plt.show()

# Вызов функций графиков
graph1()
graph2()
graph3()
polar_graph1()
polar_graph2()
polar_graph3()