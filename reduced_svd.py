import numpy as np

def reduced_svd(A):
  eigenvalues, V = np.linalg.eig(A.T@A)
  singular_values = np.sqrt(eigenvalues)
  indices = np.argsort(-singular_values)
  singular_values = singular_values[indices]
  U = A@V[:, indices]
  U /= singular_values
  r = np.count_nonzero(singular_values)
  reduced_U = U[:, :r]
  reduced_singular_values = singular_values[:r]
  reduced_V = V[:, indices[:r]]
  return reduced_U, reduced_singular_values, reduced_V.T