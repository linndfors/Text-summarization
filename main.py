from tf_idf import *
import numpy as np
from rsvd import r_svd

with open("test_datasets/test1.txt", "r") as article:
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
# print(df)
matrix = df.values

r = 400
q = 1
p = 5

U, S, VT = r_svd(matrix, r, q, p)
print(U, S, VT)