from cmath import sqrt
import numpy as np

def count_score(V: np.array, S: np.array) -> list:
    """
    Count scores for every sentence in text
    """
    num_cols = len(V[1])
    num_rows = len(V[0])
    scores = list()
    for i in range(num_rows):
        score = 0
        for j in range(num_cols):
            score += (V[i, j]**2)*(S[j]**2)
        scores.append(sqrt(score))
    return scores