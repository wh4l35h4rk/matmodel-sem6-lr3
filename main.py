import matplotlib.pyplot as plt
import numpy as np


def dydt(x, y, alpha):
    return - alpha * y * (1 - x)

def dxdt(x, y):
    return x * (1 - y)


def func(t, func_values, alpha):
    x = func_values[0]
    y = func_values[1]
    return np.array([dxdt(x, y),
                     dydt(x, y, alpha)])


def rungeKutta(x, start_conditions, param_list):
    h = (x[len(x) - 1] - x[0]) / len(x)
    y = np.zeros((1, len(start_conditions)))
    y[0] = start_conditions

    for i in range(len(x) - 1):
        k1 = func(x[i], y[i], *param_list)
        k2 = func(x[i] + 0.5 * h, y[i] + 0.5 * h * k1, *param_list)
        k3 = func(x[i] + 0.5 * h, y[i] + 0.5 * h * k2, *param_list)
        k4 = func(x[i] + h, y[i] + h * k3, *param_list)

        y_new = y[i] + (h/6) * (k1 + 2 * k2 + 2 * k3 + k4)
        y = np.append(y, [y_new], axis=0)
    return y


def plot_starts(t, start_conditions_list, parameters, ris):
    N_plots = len(start_conditions_list)
    ratios = [1, 1] if N_plots == 1 else [6, 3]

    fig, axs = plt.subplots(1, 2, figsize=(12, 5),  width_ratios=ratios)

    for i in range(N_plots):
        start_conditions = start_conditions_list[i]
        x0, y0 = start_conditions[0], start_conditions[1]

        system = rungeKutta(t, start_conditions, parameters)

        style = "-" if i % 2 == 0 else "--"
        if N_plots == 1:
            axs[0].plot(t, system[:, 0], label=rf"Жертвы, $x_0 = {x0}$")
            axs[0].plot(t, system[:, 1], label=rf"Хищники, $y_0 = {y0}$")
            axs[1].plot(system[:, 0], system[:, 1])
        else:
            axs[0].plot(t, system[:, 0], linestyle=style, label=rf"$x_0 = {x0}$")
            axs[0].plot(t, system[:, 1], linestyle=style, label=rf"$y_0 = {y0}$")
            axs[1].plot(system[:, 0], system[:, 1], linestyle=style, label=rf"$x_0 = {x0}$, $y_0 = {y0}$")

        if i == N_plots - 1:
            axs[1].plot([x0], [y0], "o", color="red", label="Нач. точка")
        else:
            axs[1].plot([x0], [y0], "o", color="red")

    axs[0].set_xlabel(r"$t$, сут")
    axs[0].set_ylabel(r"Численность популяции")
    axs[0].grid()
    axs[0].legend(loc="upper right")
    axs[1].set_xlabel(r"Численность популяции жертв")
    axs[1].set_ylabel(r"Численность популяции хищников")
    axs[1].grid()
    axs[1].legend()
    plt.savefig(f"ris_{ris}.png", bbox_inches='tight')
    plt.show()
    plt.clf()
    plt.close()

def plot_params(t, start_conditions, param_list, ris):
    N_plots = len(param_list)
    ratios = [1, 1] if N_plots == 1 else [6, 3]

    fig, axs = plt.subplots(1, 2, figsize=(12, 5),  width_ratios=ratios)

    for i in range(N_plots):
        x0, y0 = start_conditions[0], start_conditions[1]
        parameters = param_list[i]
        print(parameters)
        alpha = parameters[0]

        system = rungeKutta(t, start_conditions, parameters)

        style = "-" if i % 2 == 0 else "--"
        if N_plots == 1:
            axs[0].plot(t, system[:, 0], label=rf"Жертвы, $\alpha = {alpha}$")
            axs[0].plot(t, system[:, 1], label=rf"Хищники, $\alpha = {alpha}$")
            axs[1].plot(system[:, 0], system[:, 1])
        else:
            axs[0].plot(t, system[:, 0], linestyle=style, label=rf"$x(t), \alpha = {alpha}$")
            axs[0].plot(t, system[:, 1], linestyle=style, label=rf"$y(t), \alpha = {alpha}$")
            axs[1].plot(system[:, 0], system[:, 1], linestyle=style, label=rf"$\alpha = {alpha}$")

        if i == N_plots - 1:
            axs[1].plot([x0], [y0], "o", color="red", label="Нач. точка")
        else:
            axs[1].plot([x0], [y0], "o", color="red")

    axs[0].set_xlabel(r"$t$, сут")
    axs[0].set_ylabel(r"Численность популяции")
    axs[0].grid()
    axs[0].legend(loc="upper right")
    axs[1].set_xlabel(r"Численность популяции жертв")
    axs[1].set_ylabel(r"Численность популяции хищников")
    axs[1].grid()
    axs[1].legend()
    plt.savefig(f"ris_{ris}.png", bbox_inches='tight')
    plt.show()
    plt.clf()
    plt.close()


if __name__ == '__main__':
    x0 = 2
    y0 = 2

    alpha = 1

    a = 0
    b = 20
    N = 1500
    t = np.linspace(a, b, N + 1)

    starts1 = [[x0, y0]]
    parameters = [alpha]

    plot_starts(t, starts1, parameters, 1)

    b = 14
    t = np.linspace(a, b, N + 1)

    starts2 = [[2, 2],
              [4, 3],
              [4, 5]]

    plot_starts(t, starts2, parameters, 2)

    b = 20
    t = np.linspace(a, b, N + 1)

    param_list =  [[0.4], [2]]
    plot_params(t, starts1[0], param_list, 3)
