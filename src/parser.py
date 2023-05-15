import re
from tf_idf import *
from nltk.corpus import stopwords
from itertools import chain
# Run this once to download stopwords for text cleaning
# nltk.download('stopwords')

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

def process_file(file_path: str) -> (dict, list):
    """
    Returns a cleaned word dict and tokens list from the txt file
    """
    path = file_path

    with open(path, "r", encoding="utf-8") as file:
        data = file.read()
    sentences = nltk.sent_tokenize(data)

    tokens = []
    for sentence in sentences:
        sentence = clean_sentence(sentence)
        tokens.append(nltk.word_tokenize(sentence))

    uniqueWords = (set(chain(*tokens)))

    word_dict = []
    for token in tokens:
        tokenBag = dict.fromkeys(uniqueWords, 0)
        for word in token:
            tokenBag[word] += 1
        word_dict.append(tokenBag)

    return word_dict, tokens


def parse(target_file: str) -> pd.DataFrame:
    """
    Returns a dataframe of vectorised text
    """
    word_dict, tokens = process_file(target_file)

    tfList = []
    for n in range(len(tokens)):
        if len(tokens[n]) == 0:
            continue
        tfList.append(computeTF(word_dict[n], tokens[n]))

    idfList = computeIDF(word_dict)

    tfidfList = []
    for n in range(len(tfList)):
        tfidfList.append(computeTFIDF(tfList[n], idfList))
    df = pd.DataFrame.from_dict(tfidfList)
    df.to_csv("test_tfidf.csv")
    return df

# print(parse(r"test_datasets\test1.txt"))
