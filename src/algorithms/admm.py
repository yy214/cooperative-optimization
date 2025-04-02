import numpy as np
from algorithms.utils.kernel import nu, sigma

def admm(a, m, agents, Kmm, Knm, G, dataY, step_count, beta=0.1):
    x_i = np.zeros((step_count+1, a, m))
    y_ij = np.zeros((step_count+1, a, a, m))
    lambda_ij = np.zeros((step_count+1, a, a, m))

    Kim = [Knm[ids_agent,:] for ids_agent in agents]

    for step in range(step_count):
        for agentId in range(a):
            A = sigma**2/a*Kmm + Kim[agentId].T@Kim[agentId] + (nu/a + beta*G.degree[agentId])*np.eye(m)
            b = Kim[agentId].T @ dataY[agents[agentId]]
            for (_, j) in G.edges(agentId):
                b += beta*y_ij[step, agentId, j, :] - lambda_ij[step, agentId, j, :]
            x_i[step+1, agentId, :] = np.linalg.solve(A, b)
        for (i,j) in G.edges():
            y_ij[step+1, i, j, :] = \
            y_ij[step+1, j, i, :] = (x_i[step+1, i, :] + x_i[step+1, j, :])/2
            lambda_ij[step+1, i, j, :] = lambda_ij[step, i, j, :] + beta*(x_i[step+1, i, :] - y_ij[step+1, i, j, :]) 
            lambda_ij[step+1, j, i, :] = lambda_ij[step, j, i, :] + beta*(x_i[step+1, j, :] - y_ij[step+1, j, i, :])

    return x_i
 