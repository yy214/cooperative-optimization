import numpy as np
import pickle

# parameters
sigma = 0.5
nu = 1.0


def kernel(x1, x2):
    return np.exp(-(x1-x2)**2)


def kernel_matrix(dataX, ids1, ids2):
    return np.array([[kernel(dataX[i], dataX[j]) for j in ids2] for i in ids1])

# Kmm = kernel_matrix(dataX, ind, ind)
# Knm = kernel_matrix(dataX, sel, ind)

def calc_f(dataX, ind, val, alpha):
    r"""
    :param dataX: X coord of data
    :param ind: the m chosen indexes over which we calculate the kernels
    :param val: the "x_i" in the paper
    :param alpha: the "alpha_i" in the paper
    """
    return sum(alpha[i]*kernel(val, dataX[ind[i]]) for i in range(len(alpha)))


def grad_fi(dataY, Kmm, Knm, alpha, ids_agent):
    r"""
    :param dataY: y coord of data
    :param Kmm: Kmm as in the paper
    :param Knm: Knm as in the paper
    :param alpha: the "alpha_i" in the paper
    :param ids_agent: array of ids of the data points that are selected by the agent
    """
    Kim = Knm[ids_agent, :]
    return sigma**2/5*Kmm@alpha + Kim.T@(Kim@alpha - dataY[ids_agent]) + nu/5*alpha 


def estimate_alpha(Kmm, Knm, alpha, y):
    # alpha_opt is the argmin of this
    return sigma**2 / 2 * alpha.T@Kmm@alpha + 1/2 * np.sum((y - Knm@alpha)**2)  + nu/2*np.sum(alpha**2)

def calc_alpha_opt(Kmm, Knm, y):
    return np.linalg.solve(sigma**2*Kmm + Knm.T@Knm + nu*np.eye(y.shape[0]), Knm.T@y)

def calc_optimality_gap(Kmm, Knm, y, alpha):
    alpha_opt = calc_alpha_opt(Kmm, Knm, y)
    return np.sqrt(np.sum((alpha_opt-alpha)**2))
