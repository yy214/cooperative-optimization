import numpy as np
import matplotlib.pyplot as plt
import pickle
import networkx as nx
from utils.centralized_solution import *

from utils.kernel import kernel_matrix, grad_fi, calc_f, estimate_alpha

def solve_mod(x,y,x_selected):
    n = len(x)

    M = Cov2(x, x_selected)
    A = (0.5**2)*Cov(x_selected) + M.T @ M
    b = M.T @ y

    # here the regularization parameter nu is 1.0
    A = A + 1.*np.eye(int(np.sqrt(n)))

    # it is good to compute the max/min eigenvalues of A for later, but only for small-size matrices
    if n<101:
        ei, EI =np.linalg.eig(A)
        vv = [min(ei), max(ei)]
        print('Min and max eigenvalues of A : ', print(vv))

    alpha = np.linalg.solve(A,b)

    return alpha

def fedAVG(dataY, m, Kmm, Knm, agents, B, C, E, T, lr, lr_value):
    
    a = len(agents)
    idx_a = [i for i in range(a)]

    def sgd(dataY, B, E, lr, lr_value, t, x_t, ids_agent):
        x_k = np.copy(x_t)
        for i in range(E):
            lr_val = lr(lr_value, t*E+i)
            batch = np.random.choice(ids_agent, B, replace = False)
            x_k = x_k - lr_val*grad_fi(dataY, Kmm, Knm, x_k, batch)
        return x_k
    
    x_t = np.random.normal(0, 0.01, size=(m))
    sol = np.zeros((T+1,m))
    sol[0] = x_t
    for t in range(T):
        x_i= []
        selected_agents = np.random.choice(idx_a, C, replace = False)
        for agent in selected_agents:
            x_i.append(sgd(dataY, B, E, lr, lr_value, t, sol[t], agents[agent]))
        sol[t+1] = np.mean(np.array(x_i), axis = 0)
    return sol

def scaffold(dataY, m, Kmm, Knm, agents, B, C, E, T, lr, gamma):

    a = len(agents)
    idx_a = [i for i in range(a)]

    def sgd_scaffold(dataY, B, E, lr, x_t, c, ids_agent, c_i, t):
        x_k = np.copy(x_t)
        for i in range(E):
            batch = np.random.choice(ids_agent, B, replace=False)
            grad = grad_fi(dataY, Kmm, Knm, x_k, batch)
            x_k = x_k - lr(t*E+i)* (grad - c_i + c)
        c_i_plus = c_i-c + 1/(E*lr(t*E+i))*(x_t-x_k)
        return x_k, c_i_plus,

    x_0 = np.random.normal(0, 0.01, size=(m))
    c = np.random.normal(0, 0.01, size=(m))
    x = np.zeros((T+1, m))
    x[0] = x_0
    c_i_list = [np.zeros(m) for _ in range(a)]

    for t in range(T):
        delta_x_c = []
        delta_c_i = []
        selected_agents = np.random.choice(idx_a, C, replace=False)
        for agent in selected_agents:
            x_c, c_i_plus = sgd_scaffold(dataY, B, E, lr, x[t], c, agents[agent], c_i_list[agent], t)
            delta_x_c.append(x_c-x[t])
            delta_c_i.append(c_i_plus - c_i_list[agent])
            c_i_list[agent] = c_i_plus

        x[t+1] = x[t]+ gamma*np.mean(np.array(delta_x_c), axis=0)
        c = c + np.sum(np.array(delta_c_i), axis=0)/a

    return x



def constant_lr(lr_value, iter):
    return lr_value

def variant_lr(lr_value, iter):
    return lr_value/(0.001*iter + 1)

def variant_lr2(iter):
    return 0.002/(0.001*iter + 1)

def compute_gap(xi, X,y,x_selected):
    x_star = solve_mod(X,y,x_selected)
    gap = np.sqrt(np.sum((xi - x_star) ** 2, axis=1))
    return gap

def compute_f_gap(xi,Kmm, Knm, X,y,x_selected):

    x_star = solve_mod(X,y, x_selected)
    f_star = estimate_alpha(Kmm, Knm, x_star, y)
    fi = np.zeros(len(xi))
    for i in range(len(xi)):
        fi[i] = estimate_alpha(Kmm, Knm, xi[i], y)
    gap = np.sqrt((fi - f_star) ** 2)
    return gap

def plot_gap_TE(gaps, labels, E):

    plt.figure(figsize=(8, 6))

    for k in range(len(gaps)):
        plt.loglog(E[k]*np.arange(len(gaps[k])), gaps[k], linestyle='-', label = labels[k])
    plt.xlabel('TE')
    plt.ylabel(r'Gap $||F(\alpha_i) - F(\alpha*)||$')
    plt.title('optimality gap')
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.show()

def plot_gap_TC(gaps, labels, C):

    plt.figure(figsize=(8, 6))

    for k in range(len(gaps)):
        plt.loglog(C[k]*np.arange(len(gaps[k])), gaps[k], linestyle='-', label = labels[k])
    plt.xlabel('TC')
    plt.ylabel(r'Gap $||F(\alpha_i) - F(\alpha*)||$')
    plt.title('optimality gap')
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.show()

def kernel_matrix(dataX1, dataX2, ids1, ids2):
    return np.array([[kernel(dataX1[i], dataX2[j]) for j in ids2] for i in ids1])

if __name__=="__main__":

    with open("second_database.pkl", "rb")as f:
        X,y=pickle.load(f)

    liste_aplatie = [arr.flatten() for arr in X]

    # Concaténer les tableaux aplatis
    X = np.concatenate(liste_aplatie)
    liste_aplatie = [y_i for y_agent in y for y_i in y_agent]

    # Concaténer les tableaux aplatis
    y = np.array(liste_aplatie)

    # parameters
    n = 100
    m = 10
    a = 5 # nb of agents
    step_size = 0.002


    # selected indexes for calculations
    sel = list(range(n))
    ind = np.random.choice(sel,m,replace=False)
    x_m_points = np.linspace(-1,1,m) # ids_M
    Kmm = kernel_matrix(x_m_points, x_m_points, range(m), range(m))
    Knm = kernel_matrix(X, x_m_points, sel, range(m))

    # agents
    sel_copy = np.arange(n)
    agents = np.array_split(sel_copy, a)

### Constant step size fedAVG
    labels = []
    gaps = []
    E = [20,1,50]
    C = [5,5,5]
    labels.append("fedAVG B=20, C=5, E = 20")
    alpha = fedAVG(y, m, Kmm, Knm, agents, 20, 5, 20, 500, constant_lr, step_size)
    gaps.append(compute_gap(alpha, X[sel],y[sel], x_m_points))
    labels.append("fedAVG B=20, C=5, E = 1 (=DGD)")
    alpha = fedAVG(y, m, Kmm, Knm, agents, 20, 5, 1, 10000, constant_lr, step_size)
    gaps.append(compute_gap(alpha, X[sel],y[sel], x_m_points))
    labels.append("fedAVG B=20, C=5, E = 50")
    alpha = fedAVG(y, m, Kmm, Knm, agents, 20, 5, 50, 200, constant_lr, step_size)
    gaps.append(compute_gap(alpha, X[sel],y[sel], x_m_points))
    
    plot_gap_TE(gaps, labels, E)
    plot_gap_TC(gaps, labels, C)

    # labels = []
    # gaps = []
    # E = [20,20]
    # C = [3,5]
    # labels.append("fedAVG B=20, C=3, E = 20")
    # alpha = fedAVG(y, m, Kmm, Knm, agents, 20, 3, 20, 500, constant_lr, step_size)
    # gaps.append(compute_gap(alpha, X[sel],y[sel], x_m_points))
    # labels.append("fedAVG B=20, C=5, E = 20")
    # alpha = fedAVG(y, m, Kmm, Knm, agents, 20, 5, 20, 500, constant_lr, step_size)
    # gaps.append(compute_gap(alpha, X[sel],y[sel], x_m_points))

    # plot_gap_TE(gaps, labels, E)
    # plot_gap_TC(gaps, labels, C)

    # labels = []
    # gaps = []
    # E = [20,20,20]
    # C = [5,5,5]
    # labels.append("fedAVG B=10, C=5, E = 20")
    # alpha = fedAVG(y, m, Kmm, Knm, agents, 10, 5, 20, 500, constant_lr, step_size)
    # gaps.append(compute_gap(alpha, X[sel],y[sel], x_m_points))
    # labels.append("fedAVG B=15, C=5, E = 20")
    # alpha = fedAVG(y, m, Kmm, Knm, agents, 15, 5, 20, 500, constant_lr, step_size)
    # gaps.append(compute_gap(alpha, X[sel],y[sel], x_m_points))
    # labels.append("fedAVG B=20, C=5, E = 20")
    # alpha = fedAVG(y, m, Kmm, Knm, agents, 20, 5, 20, 500, constant_lr, step_size)
    # gaps.append(compute_gap(alpha, X[sel],y[sel], x_m_points))
    
    # plot_gap_TE(gaps, labels, E)
    # plot_gap_TC(gaps, labels, C)

### Diminishing stepsize fedAVG

    labels = []
    gaps = []
    E = [20,1,50]
    C = [5,5,5]
    labels.append("fedAVG B=20, C=5, E = 20")
    alpha = fedAVG(y, m, Kmm, Knm, agents, 20, 5, 20, 500, variant_lr, step_size)
    gaps.append(compute_gap(alpha, X[sel],y[sel], x_m_points))
    labels.append("fedAVG B=20, C=5, E = 1 (=DGD)")
    alpha = fedAVG(y, m, Kmm, Knm, agents, 20, 5, 1, 10000, variant_lr, step_size)
    gaps.append(compute_gap(alpha, X[sel],y[sel], x_m_points))
    labels.append("fedAVG B=20, C=5, E = 50")
    alpha = fedAVG(y, m, Kmm, Knm, agents, 20, 5, 50, 200, variant_lr, step_size)
    gaps.append(compute_gap(alpha, X[sel],y[sel], x_m_points))
    
    plot_gap_TE(gaps, labels, E)
    plot_gap_TC(gaps, labels, C)

    # labels = []
    # gaps = []
    # E = [20,20]
    # C = [3,5]
    # labels.append("fedAVG B=20, C=3, E = 20")
    # alpha = fedAVG(y, m, Kmm, Knm, agents, 20, 3, 20, 500, variant_lr, step_size)
    # gaps.append(compute_gap(alpha, X[sel],y[sel], x_m_points))
    # labels.append("fedAVG B=20, C=5, E = 20")
    # alpha = fedAVG(y, m, Kmm, Knm, agents, 20, 5, 20, 500, variant_lr, step_size)
    # gaps.append(compute_gap(alpha, X[sel],y[sel], x_m_points))

    # plot_gap_TE(gaps, labels, E)
    # plot_gap_TC(gaps, labels, C)

    # labels = []
    # gaps = []
    # E = [20,20,20]
    # C = [5,5,5]
    # labels.append("fedAVG B=10, C=5, E = 20")
    # alpha = fedAVG(y, m, Kmm, Knm, agents, 10, 5, 20, 500, variant_lr, step_size)
    # gaps.append(compute_gap(alpha, X[sel],y[sel], x_m_points))
    # labels.append("fedAVG B=15, C=5, E = 20")
    # alpha = fedAVG(y, m, Kmm, Knm, agents, 15, 5, 20, 500, variant_lr, step_size)
    # gaps.append(compute_gap(alpha, X[sel],y[sel], x_m_points))
    # labels.append("fedAVG B=20, C=5, E = 20")
    # alpha = fedAVG(y, m, Kmm, Knm, agents, 20, 5, 20, 500, variant_lr, step_size)
    # gaps.append(compute_gap(alpha, X[sel],y[sel], x_m_points))
    
    # plot_gap_TE(gaps, labels, E)
    # plot_gap_TC(gaps, labels, C)

    labels = []
    gaps = []
    labels.append("fedAVG B=20, C=3, E=50")

    alpha = fedAVG(y, m, Kmm, Knm, agents, 20, 3, 20, 500, variant_lr, step_size)
    gaps.append(compute_f_gap(alpha,Kmm, Knm, X[sel],y[sel],x_m_points))

    labels.append("SCAFFOLD B=20, C=3, E=50")
    alpha = scaffold(y, m, Kmm, Knm, agents, 20,3,20,500, variant_lr2, 1)
    gaps.append(compute_f_gap(alpha,Kmm, Knm, X[sel],y[sel],x_m_points))
    plot_gap_TE(gaps, labels, [50,50])
    
