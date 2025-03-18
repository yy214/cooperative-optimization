import numpy as np
import matplotlib.pyplot as plt
import pickle
import networkx as nx

from utils.graph_tools import metropolis_weight_matrix, laplacian_weight_matrix
from utils.kernel import kernel_matrix, grad_fi, calc_f

def main():
    with open("first_database.pkl", "rb")as f:
        X,y=pickle.load(f)

    # parameters
    n = 100
    m = 10
    a = 5 # nb of agents
    step_size = 0.002 
    step_count = 1000

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
    xi = np.random.normal(0, 0.01, size=(step_count+1, a, m))
    g = np.zeros((step_count+1, a, m))
    # local gradients
    for i, id_agent in enumerate(agents):
        g[0, i, :] = grad_fi(y, Kmm, Knm, xi[0,i,:], id_agent)

    for k in range(step_count):
        xi[k+1, :, :] = W@xi[k, :, :] - step_size*g[k, :, :] # note that W is symetric so W.T doesn't matter
        g[k+1, :, :] = W@g[k,:,:]
        for i, id_agent in enumerate(agents):
            g[k+1, i, :] += \
                + grad_fi(y, Kmm, Knm, xi[k+1,i, :], id_agent) \
                - grad_fi(y, Kmm, Knm, xi[k,i,:], id_agent)

    # affichage
    for i in range(a):
        plt.scatter(X[agents[i]], y[agents[i]], label="agent %d"%(i))

    nt = 250
    x_linspace = np.linspace(-1, 1, nt)
    for s in range(0, step_count+1, step_count // 5):
        pred = [calc_f(X, ind, v, xi[s, 0, :]) for v in x_linspace]
        plt.plot(x_linspace, pred, label="step %d" % s)

    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()