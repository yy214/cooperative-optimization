import networkx as nx # requires scipy
import numpy as np


def laplacian_weight_matrix(G, epsilon):
    assert epsilon < min(1/val for (node, val) in G.degree() if val!=0)
    return np.eye(G.number_of_nodes()) - epsilon*nx.laplacian_matrix(G)


def metropolis_weight_matrix(G, lazy=False): 
    raise NotImplementedError("CORRECT THIS")
    a = G.number_of_nodes()
    W = np.zeros((a,a))
    degrees = [val for (node, val) in G.degree()]
    for i in range(a):
        for j in range(i+1, a):
            weight = 1 / (1 + max(degrees[i], degrees[j]))
            if lazy:
                weight /= 2
            W[i,j] = W[j,i] = weight

    W[np.diag_indices_from(W)] = 1 - np.sum(W, axis = 1)
    return W