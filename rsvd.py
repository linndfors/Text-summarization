import numpy as np




def r_svd(M, r_reducing_size, q_iteration, p_over_sampling):
    col_space = M.shape[1]
    # print(col_space)
    P_proj = np.random.randn(col_space, r_reducing_size + p_over_sampling)
    Z = M @ P_proj
    for i in range(q_iteration):
        Z = M @ (M.T @ Z)

    Q, R = np.linalg.qr(Z, mode='reduced')

    Y_proj = Q.T @ M
    UY, S, VT = np.linalg.svd(Y_proj, full_matrices=0)
    U = Q @ UY

    return U, S, VT

