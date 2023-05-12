import numpy as np
from math import sqrt
from reduced_svd import reduced_svd

def count_score(V, S):
    num_cols = len(V[0])
    num_rows = len(V[1])
    scores = list()
    for i in range(num_rows):
        score = 0
        for j in range(num_cols):
            # print("S: ", S[j])
            # print("V: ", V[i, j])
            # print("S**2: ", S[j]**2)
            # print("V**2: ", V[i, j]**2)
            # print("S*V: ", (V[i, j]**2)*(S[j]**2))
            # print("Score: ", score)
            score += (V[i, j]**2)*(S[j]**2)
        scores.append(sqrt(score))
        # print("-----------------------------------------")
    return scores

def test():
    A = np.array([[1, 2, 3, 4], [2, 7, 4, 5], [3, 4, 3, 8], [4, 5, 8, 3]])
    U, S, VT = reduced_svd(A)
    print(U)
    print(S)
    print(VT)
    print(U@S@VT)
    print(count_score(VT, S))

test()