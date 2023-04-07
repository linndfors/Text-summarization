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

    with open("test_article.txt", "r") as article:
        data = article.read()

    sentences = nltk.sent_tokenize(data)

    tokens = []
    for sentence in sentences:
        tokens.append(nltk.word_tokenize(sentence))

    uniqueWords = set(nltk.word_tokenize(data))

    bagOfWords = []
    for token in tokens:
        tokenBag = dict.fromkeys(uniqueWords, 0)
        for word in token:
            tokenBag[word] += 1
        bagOfWords.append(tokenBag)

    tfList = []
    for n in range(len(tokens)):
        tfList.append(computeTF(bagOfWords[n], tokens[n]))

    idfList = computeIDF(bagOfWords)


    tfidfList = []
    for n in range(len(tfList)):
        tfidfList.append(computeTFIDF(tfList[n], idfList))
    df = pd.DataFrame.from_dict(tfidfList)
    df.to_csv("test_tfidf.csv")
    print(df)

    # #built-in tf-ifd function
    # vectorizer = TfidfVectorizer()
    # vectors = vectorizer.fit_transform([documentA, documentB])
    # feature_names = vectorizer.get_feature_names_out()
    # dense = vectors.todense()
    # denselist = dense.tolist()
    # df1 = pd.DataFrame(denselist, columns=feature_names)
    # print(df1)
