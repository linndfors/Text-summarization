import re
from os import listdir

def get_score(file_path_target, file_path_test):
    with open(file_path_test, "r", encoding="utf-8") as file:
        data_test = file.read()
    with open(file_path_target, "r", encoding="utf-8") as file:
        data_target = file.read()
    
    data_test = re.sub(r'[\.\?\!\,\:\;\"\*\“\_\-\”\'\’\[\]\(\)\@\%\/]', '', data_test)
    data_test = data_test.lower()
    data_test = data_test.split()

    data_target = re.sub(r'[\.\?\!\,\:\;\"\*\“\_\-\”\'\’\[\]\(\)\@\%\/]', '', data_target)
    data_target = data_target.lower()
    data_target = data_target.split()

    word_counts1 = {}
    word_counts2 = {}
    for word in data_test:
        word_counts1[word] = word_counts1.get(word, 0) + 1
    for word in data_target:
        word_counts2[word] = word_counts2.get(word, 0) + 1

    shared = {word: min(word_counts1[word], word_counts2[word]) for word in word_counts1 if word in word_counts2}
    multiplicity = sum(shared.values())

    precision = multiplicity / len(data_test)

    recall = multiplicity / len(data_target)

    score = 2 * (precision * recall) / (precision + recall)

    print(precision)
    print(recall)
    print(score)

    return score


def evaluate(folder_path_target, folder_path_test):
    score = 0
    lst_target = listdir(folder_path_target)
    lst_test = listdir(folder_path_test)
    assert len(lst_target) == len(lst_test), "The number of files doesn't match"
    count = len(lst_target)
    for idx in range(count):
        score += get_score(folder_path_target+"\\"+lst_target[idx], folder_path_test+"\\"+lst_test[idx])
    score /= count
    return score

print(evaluate(r"test_rouge-n\f1", r"test_rouge-n\f2"))


# from rouge import Rouge
# model_out = "he began by starting a five person war cabinet and included chamberlain as lord president of the council"

# reference = "he began his premiership by forming a five-man war cabinet which included chamberlain as lord president of the council"
# rouge = Rouge()
# rouge.get_scores(model_out, reference)