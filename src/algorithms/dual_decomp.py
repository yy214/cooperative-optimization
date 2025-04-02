import numpy as np
from algorithms.utils.kernel import nu, sigma

def dual_decomp(a, m, agents, Kmm, Knm, G, dataY, step_count, step_size=0.02):
    x_i = np.zeros((step_count+1, a, m))
    lambda_ij = np.zeros((step_count+1, a, a, m))

    Kim = [Knm[ids_agent,:] for ids_agent in agents]

    for step in range(step_count):
        for agentId in range(a):
            A = sigma**2/5*Kmm + Kim[agentId].T@Kim[agentId] + nu/5*np.eye(m)
            b = Kim[agentId].T @ dataY[agents[agentId]]
            for (_, j) in G.edges(agentId):
                b -= lambda_ij[step, agentId, j, :] * (2*(agentId > j) - 1)
            x_i[step+1, agentId, :] = np.linalg.solve(A, b)
        for (i,j) in G.edges():
            k, l = min(i,j), max(i,j)
            lambda_ij[step+1, k, l, :] = \
            lambda_ij[step+1, l, k, :] = lambda_ij[step, l, k, :] + step_size*(x_i[step+1, l, :] - x_i[step+1, k, :])
    
    return x_i
 