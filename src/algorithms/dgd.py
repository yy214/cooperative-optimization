import numpy as np
import matplotlib.pyplot as plt
import pickle
import networkx as nx
from utils.centralized_solution import *

from utils.graph_tools import metropolis_weight_matrix, laplacian_weight_matrix
from utils.kernel import kernel_matrix, grad_fi, calc_f

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

def dgd(a,m,agents,Kmm, Knm,W,y, step_count, step_size, constant_step_size = True):
    xi = np.random.normal(0, 0.01, size=(step_count+1, a, m))
    g = np.zeros((a, m))
    # local gradients
    for i, ids_agent in enumerate(agents):
        g[i, :] = grad_fi(y, Kmm, Knm, xi[0,i,:], ids_agent)

    for k in range(step_count):
        if not constant_step_size:
            val_step_size = step_size(k)
        else:
            val_step_size = step_size

        xi[k+1, :, :] = W@xi[k, :, :] - val_step_size*g[:, :] # note that W is symetric so W.T doesn't matter
        for i, ids_agent in enumerate(agents):
            g[i, :] = grad_fi(y, Kmm, Knm, xi[k+1,i, :], ids_agent)
    return xi

def plot(agents, X,y, xi, step_count):
    for i in range(len(agents)):
        plt.scatter(X[agents[i]], y[agents[i]], label="agent %d"%(i))

    nt = 250
    x_linspace = np.linspace(-1, 1, nt)
    for s in range(0, step_count+1, step_count // 5):
        pred = [calc_f(X, ind, v, xi[s, 0, :]) for v in x_linspace]
        plt.plot(x_linspace, pred, label="step %d" % s)

    plt.legend()
    plt.show()

def compute_gap(xi, X,y,x_selected):
    x_k = np.mean(xi, axis=1)
    x_star = solve_mod(X,y, x_selected)

    gap = np.sqrt(np.sum((x_k - x_star) ** 2, axis=1))
    return gap

def plot_gap(gaps):

    plt.figure(figsize=(8, 6))

    for k in range(len(gaps)):
        plt.loglog(np.arange(len(gaps[k])), gaps[k], linestyle='-', label = str(k))
    plt.xlabel('number of steps')
    plt.ylabel(r'Gap $||\alpha_i - \alpha*||$')
    plt.title('optimality gap of DGD')
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.show()

def diminishing_step_size(k):
    return 0.002/(0.001*k+1)

def dgd_dp(a,m,agents,Kmm, Knm,W,y, step_count):
    xi = np.random.normal(0, 0.01, size=(step_count+1, a, m))
    g = np.zeros((a, m))
    # local gradients
    for i, ids_agent in enumerate(agents):
        g[i, :] = grad_fi(y, Kmm, Knm, xi[0,i,:], ids_agent)

    W_hat = W - np.identity(W.shape[0])
    for k in range(step_count):
        val_step_size = dim_step_size(k)
        var = laplace_scale(k)
        gamma = dim_gamma(k)
        laplacian_noise = np.random.laplace(loc = 0.0, scale = var, size = (a,m))
        ksi = xi[k, :, :] + laplacian_noise
        for i in range(a):
            xi[k+1, i, :] = xi[k, i, :] - gamma*np.sum([W_hat[i,j]*(ksi[j,:]-xi[k,i,:]) for j in range(a) if j != i], axis = 0) - val_step_size*g[i, :] # note that W is symetric so W.T doesn't matter
        for i, ids_agent in enumerate(agents):
            g[i, :] = grad_fi(y, Kmm, Knm, xi[k+1,i, :], ids_agent)
    return xi

def laplace_scale(iter):
    return 1+0.001*iter**0.3
def dim_step_size(iter):
    return 0.002/(1+0.001*iter)
def dim_gamma(iter):
    return 0.002/(1+0.001*iter**0.9)


if __name__=="__main__":

    with open("first_database.pkl", "rb")as f:
        X,y=pickle.load(f)

    # parameters
    n = 100
    m = 10
    a = 5 # nb of agents
    step_size = 0.002 
    step_count = 100000

    # selected indexes for calculations
    sel = list(range(n))
    ind = np.random.choice(sel,m,replace=False)
    x_selected = X[ind] # ids_M
    Kmm = kernel_matrix(X, ind, ind)
    Knm = kernel_matrix(X, sel, ind)

    # agents
    sel_copy = np.arange(n)
    np.random.shuffle(sel_copy)
    agents = np.array_split(sel_copy, a)


    G = nx.cycle_graph(a)
    W = laplacian_weight_matrix(G, 0.1) # check the matrices

    # dgd

    xi_cst_step_size = dgd(a,m,agents,Kmm, Knm,W,y, step_count, step_size)
    # xi_dim_step_size = dgd(a,m,agents,Kmm, Knm,W,y, step_count, diminishing_step_size, constant_step_size=False)
    # affichage

    # plot(agents, X,y, xi_dim_step_size, step_count)

    # gaps = [compute_gap(xi_cst_step_size, X[sel],y[sel],x_selected), compute_gap(xi_dim_step_size, X[sel],y[sel],x_selected)]
    # plot_gap(gaps)



    xi_dgd_dp = dgd_dp(a,m,agents,Kmm, Knm,W,y, step_count)
    gaps = [compute_gap(xi_dgd_dp, X[sel],y[sel],x_selected), compute_gap(xi_cst_step_size, X[sel],y[sel],x_selected)]
    plot_gap(gaps)



    
