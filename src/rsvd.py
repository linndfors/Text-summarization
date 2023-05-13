import numpy as np
from src.reduced_svd import reduced_svd


def r_svd(M, r_reducing_size, q_iteration, p_over_sampling):
    col_space = M.shape[1]
    P_proj = np.random.randn(col_space, r_reducing_size + p_over_sampling)
    Z = M @ P_proj
    for i in range(q_iteration):
        Z = M @ (M.T @ Z)

    Q, R = np.linalg.qr(Z, mode='reduced')

    Y_proj = Q.T @ M
    UY, S, VT = reduced_svd(Y_proj)
    U = Q @ UY

    return U, S, VT

