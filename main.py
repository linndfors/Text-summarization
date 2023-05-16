import numpy as np
from src.reduced_svd import reduced_svd
from src.parser import parse
from src.score_count import count_score
import sys
import time

def find_sentence(path: str, n: int, app_flag = 0):
    """
    Find the most important n sentences
    """
    data_frame, sentence_list = parse(path, app_flag)
    sentence_list = np.array(sentence_list)
    matrix_of_words = data_frame.values
    U, S, Vt = reduced_svd(matrix_of_words.T)
    scores_for_sentences = np.array(count_score(Vt, S))
    indices_of_scores = np.argsort(-scores_for_sentences)
    best_sentences = sentence_list[indices_of_scores][: n]
    best_sentences.sort()
    for sentence in best_sentences:
        print(sentence)
    return best_sentences


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Please provide two arguments.")
        sys.exit(1)

    path = sys.argv[1]
    cutter_idx = int(sys.argv[2])
    find_sentence(path, cutter_idx)