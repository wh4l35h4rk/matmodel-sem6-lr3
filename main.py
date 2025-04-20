import matplotlib.pyplot as plt
import numpy as np


def dydt(x, y, alpha):
    return - alpha * y * (1 - x)

def dxdt(x, y):
    return x * (1 - y)

def dydx(x, y, alpha):
    print(dydt(x, y, alpha), dxdt(x, y))
    return dydt(x, y, alpha) / dxdt(x, y)


def func(t, func_values, alpha):
    x = func_values[0]
    y = func_values[1]
    return np.array([dxdt(x, y),
                     dydt(x, y, alpha)])


def rungeKutta(x, start_conditions, func_type, param_list):
    h = (x[len(x) - 1] - x[0]) / len(x)
    y = np.zeros((1, len(start_conditions)))
    y[0] = start_conditions

    for i in range(len(x) - 1):
        k1 = func_type(x[i], y[i], *param_list)
        k2 = func_type(x[i] + 0.5 * h, y[i] + 0.5 * h * k1, *param_list)
        k3 = func_type(x[i] + 0.5 * h, y[i] + 0.5 * h * k2, *param_list)
        k4 = func_type(x[i] + h, y[i] + h * k3, *param_list)

        y_new = y[i] + (h/6) * (k1 + 2 * k2 + 2 * k3 + k4)
        y = np.append(y, [y_new], axis=0)
    return y



if __name__ == '__main__':
    x0 = 2
    y0 = 2

    alpha = 1

    a = 0
    b = 30
    N = 300
    t = np.linspace(a, b, N + 1)

    parameters = [alpha]


    system_t = rungeKutta(t, [x0, y0], func, parameters)
    # system_phase = rungeKutta(t, [x0, y0], func_phase, parameters)

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    axs[0].plot(t, system_t[:, 0], label=rf"Жертвы")
    axs[0].plot(t, system_t[:, 1], label=rf"Хищники")
    axs[1].plot(system_t[:, 0], system_t[:, 1])

    axs[0].set_xlabel(r"$t$, с")
    axs[0].set_ylabel(r"Численность популяции")
    axs[0].grid()
    axs[0].legend()
    axs[1].set_xlabel(r"Численность популяции жертв")
    axs[1].set_ylabel(r"Численность популяции хищников")
    axs[1].grid()
    plt.savefig(f"alpha1.png", bbox_inches='tight')
    plt.show()
    plt.clf()
    plt.close()

