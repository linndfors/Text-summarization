import re
from os import listdir
from rouge import Rouge
from nltk.corpus import stopwords

def clean_sentence(text: str) -> str:
    """
    Returns a cleaned string
    """
    text = re.sub(r'[\.\?\!\,\:\;\"*\“\_\-\”\'\’\[\]\(\)\@\%\/\+\°]', '', text)
    pattern = r'[\d\.]+'
    text = re.sub(pattern, '', text)

    text = text.lower()
    text = [word for word in text.split() if word not in stopwords.words('english')]
    return " ".join(text)

def generate_ngrams(words_list, n):
    ngrams = []
    for i in range(len(words_list) - n + 1):
        ngram = ' '.join(words_list[i:i+n])
        ngrams.append(ngram)
    return ngrams


def get_score(file_path_target, file_path_test, n) -> float:
    with open(file_path_test, "r", encoding="utf-8") as file:
        data_test = file.read()
    with open(file_path_target, "r", encoding="utf-8") as file:
        data_target = file.read()

    data_test = clean_sentence(data_test)

    data_target = clean_sentence(data_target)

    data_test = generate_ngrams(data_test, n)
    data_target = generate_ngrams(data_target, n)

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

    return score

def evaluate(folder_path_target, folder_path_test, n):
    score = 0
    lst_target = listdir(folder_path_target)
    lst_test = listdir(folder_path_test)
    assert len(lst_target) == len(lst_test), "The number of files doesn't match"
    count = len(lst_target)
    for idx in range(count):
        score += get_score(folder_path_target+"/"+lst_target[idx], folder_path_test+"/"+lst_test[idx], n)
    score /= count
    return score

def rouge_N(parent_path: str, n_lst: list) -> dict:
    k_scores = dict()
    score_lst = []
    temp_lst = list(listdir(parent_path))
    value_lst_lst = []
    for path in temp_lst:
        value_lst = []
        target_path, test_path = listdir(parent_path+"/"+path)
        score = 0
        target_lst, test_lst = listdir(parent_path+"/"+path+"/"+target_path), listdir(parent_path+"/"+path+"/"+test_path)
            
        for n in n_lst:
            value = evaluate(parent_path+"/"+path+"/"+target_path, parent_path+"/"+path+"/"+test_path, n)
            value_lst.append(value) 
        if score == 0:
            score = value_lst
        else:
            score = tuple(a + b for a, b in zip(score, value_lst))
        k_scores[path] = score
    return k_scores

# print(rouge_N(r"test_rouge-n", [1,2,3,4,5]))


import matplotlib.pyplot as plt

data = {
    'k_6_5': [0.8426986193398691, 0.6134944634677812, 0.3074008074744985, 0.16035129267811327, 0.09837154826793999],
    'k_7_10': [0.9102626896912611, 0.7055760046047077, 0.4663283175691023, 0.32691395503272935, 0.25149545165739434],
    'k_8_20': [0.9318631405705435, 0.7896501858981069, 0.5624684234189068, 0.4059943240878484, 0.32256738987508216],
    'k_9_50': [0.9591719383168928, 0.8780216818841384, 0.7435152177117814, 0.6222661736344328, 0.542219496816125]
}
temp = [5,10,20,50]
idx = 0
for key, values in data.items():
    rounded_values = [round(value, 2) for value in values]
    x_ticks = [1, 2, 3, 4, 5]
    
    plt.bar(x_ticks, rounded_values)
    plt.xticks(x_ticks)
    plt.xlabel("Rouge-n")
    plt.ylabel("Score")
    plt.title(f"Histogram - Summary for the lengh {temp[idx]}")
    plt.show()
    idx += 1
