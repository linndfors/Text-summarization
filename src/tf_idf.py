import math
from sklearn.feature_extraction.text import TfidfVectorizer
# import nltk
# nltk.download('punkt')

def computeTF(word_dict, bag_of_words):
    tf_dict = {}
    bag_of_words_count = len(bag_of_words)
    for word, count in word_dict.items():
        tf_dict[word] = count / float(bag_of_words_count)
    return tf_dict

def computeIDF(documents):
    n = len(documents)
    idf_dict = dict.fromkeys(documents[0].keys(), 0)
    for document in documents:
        for word, val in document.items():
            if val > 0:
                idf_dict[word] += 1
    for word, val in idf_dict.items():
        idf_dict[word] = math.log(n / float(val))
    return idf_dict

def computeTFIDF(tf_bag_of_words, idfs):
    tfidf = {}
    for word, val in tf_bag_of_words.items():
        tfidf[word] = val * idfs[word]
    return tfidf
