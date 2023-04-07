import nltk
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# nltk.download('punkt')

def computeTF(wordDict, bagOfWords):
    tfDict = {}
    bagOfWordsCount = len(bagOfWords)
    for word, count in wordDict.items():
        tfDict[word] = count / float(bagOfWordsCount)
    return tfDict

def computeIDF(documents):
    import math
    N = len(documents)
    
    idfDict = dict.fromkeys(documents[0].keys(), 0)
    for document in documents:
        for word, val in document.items():
            if val > 0:
                idfDict[word] += 1
    
    for word, val in idfDict.items():
        idfDict[word] = math.log(N / float(val))
    return idfDict

def computeTFIDF(tfBagOfWords, idfs):
    tfidf = {}
    for word, val in tfBagOfWords.items():
        tfidf[word] = val * idfs[word]
    return tfidf


if __name__ == "__main__":

    documentA = 'the man went out for a walk'
    documentB = 'the children sat around the fire'

    #custom tf-idf implementation
    tokens1 = nltk.word_tokenize(documentA)
    tokens2 = nltk.word_tokenize(documentB)

    uniqueWords = set(tokens1).union(set(tokens2))

    numOfWordsA = dict.fromkeys(uniqueWords, 0)
    for word in tokens1:
        numOfWordsA[word] += 1
    numOfWordsB = dict.fromkeys(uniqueWords, 0)
    for word in tokens2:
        numOfWordsB[word] += 1

    tfA = computeTF(numOfWordsA, tokens1)
    tfB = computeTF(numOfWordsB, tokens2)

    idfs = computeIDF([numOfWordsA, numOfWordsB])

    tfidfA = computeTFIDF(tfA, idfs)
    tfidfB = computeTFIDF(tfB, idfs)
    df = pd.DataFrame([tfidfA, tfidfB])
    print(df)

    #built-in tf-ifd function
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([documentA, documentB])
    feature_names = vectorizer.get_feature_names_out()
    dense = vectors.todense()
    denselist = dense.tolist()
    df1 = pd.DataFrame(denselist, columns=feature_names)
    print(df1)
