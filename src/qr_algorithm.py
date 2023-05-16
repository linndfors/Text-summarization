import numpy as np

def qr_algorithm(matrix, max_iter=1000, tolerance=1e-7):
    n = matrix.shape[0]
    eigenvalues = np.zeros(n)
    eigenvectors = np.eye(n)

    for _ in range(max_iter):
        Q, R = np.linalg.qr(matrix)
        matrix = R@Q
        eigenvectors = eigenvectors@Q
        if np.max(np.abs(np.triu(matrix, k=1))) < tolerance and np.max(np.abs(np.tril(matrix, k=1))):
            break

    for i in range(n):
        eigenvalues[i] = matrix[i, i]

    return eigenvalues, eigenvectors