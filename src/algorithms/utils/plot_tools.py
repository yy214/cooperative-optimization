import matplotlib.pyplot as plt
import numpy as np
from kernel import calc_f

def plot_data(agents, ind, X, y, xi, algorithm_name, max_step = 10000):
    a = len(agents)
    for i in range(a):
        plt.scatter(X[agents[i]], y[agents[i]], label=f"agent {i}")

    nt = 250
    x_linspace = np.linspace(-1, 1, nt)
    step_size = max_step//100
    for s in range(0, max_step+1, step_size):
        pred = [calc_f(X, ind, v, xi[s, 0, :]) for v in x_linspace]
        if s in [step_size, max_step]:
            plt.plot(x_linspace, pred, label=f"step {s}")
        else:
            plt.plot(x_linspace, pred, color='gray', alpha=0.3)

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Prediction for {algorithm_name}')
    plt.legend()
    plt.show()