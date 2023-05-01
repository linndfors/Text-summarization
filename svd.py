import numpy as np
from numpy.linalg import eigh, norm




A = np.array([[3, 2, 2],
            [2, 3, -2]])

print(A)  

# Sl = A@A.T
# Sr = A.T@A

# eigen_values, V = eigh(Sr)

# u0 = A@V[:, 0] / norm(A@V[:,0])
# u1 = A@V[:, 1] / norm(A@V[:,1])
# # u2 = A@V[:, 2] / norm(A@V[:,2])

# U = np.array([u0, u1]).T

# D = np.round(U.T@A@V, decimals=5)

# print(U@D@V.T)


    # Sr = A.T@A
    # _, V = eigh(Sr)
    # U = np.zeros((A.shape[1], A.shape[1]))
    # for i in range(V.shape[0]):
    #     temp = A@V[:,i]
    #     # U = np.append(U, temp / norm(temp))
    #     U[i] = temp / norm(temp)
    # U = U.T
    # D = np.round(U.T@A@V, decimals=5)


def normalise(matrix):
    norms = norm(matrix, axis=0)
    matrix /= norms
    return matrix


def compute_svd(A):
    """
    Returns a list of arrays
    coresponding to the SVD:
    U, D, VT
    """

    n, m = A.shape

    Sr = A @ A.T
    Sl = A.T @ A

    e_val_V, V = eigh(Sl)

    V = normalise(V)

    #Sort eigenvectors
    idx = np.argsort(-e_val_V)
    V = V[:, idx]

    e_val_U, U = eigh(Sr)

    U = normalise(U)

    #Sort eigenvectors
    idx = np.argsort(-e_val_U)
    U = U[:, idx]

    # Change later
    V = np.negative(V)
    V = np.round(V, decimals=5)
    #Find singular values optimisation needed

    inter = set(e_val_V).intersection(set(e_val_U))
    temp = np.array(list(inter))
    temp = np.sort(temp)[::-1]
    singular_vals = np.sqrt(temp)

    D = np.zeros((n,m))
    np.fill_diagonal(D, singular_vals)

    matr = normalise(A@V) @ D

    return U, D, V, matr

U, D, V, matr = compute_svd(A)

# print(V.T@D@U)
print(U)
print(matr)
print(D)
print(V.T)

# print(U@D@V.T)
