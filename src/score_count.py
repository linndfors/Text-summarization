from cmath import sqrt
import numpy as np

def count_score(Vt: np.array, S: np.array) -> list:
    """
    Count scores for every sentence in text
    """
    V = Vt.T
    num_rows = V.shape[0]
    num_cols = V.shape[1]
    scores = list()
    for i in range(num_rows):
        score = 0
        for j in range(num_cols):
            score += (V[i, j]**2)*(S[j]**2)
        scores.append(sqrt(score))
    return scores