from textrazor import TextRazor
from collections import defaultdict, Counter, OrderedDict
import json
import re
import operator
import string
import numpy as np
import math
from sklearn.cluster import KMeans
import nltk.data
import pandas as pd
from tabulate import tabulate


#Using text razor to get the topic type
# Currently used - entities
def createKeywordsFile(fileName):
    outPutFileName = fileName+"-entities.txt"
    client = TextRazor("eb2a5954577a28b1ba8a97005281289493eee48179d2ac1883b685e6",
                       extractors=["entities"])
    outputFile = open(outPutFileName, "w")
    outPutFileName = "entityStore.txt"
    file = open(fileName, 'r')
    for line in file:
        sentence = line
        response = client.analyze(sentence)
        entityList = []
        for entity in response.entities():
            entityList.append(entity.matched_text)
        entityList = list(set(entityList))
        sentenceModified = ' '.join(entityList)
        outputFile.write(sentenceModified)
        outputFile.write("\n")
    outputFile.close()
    return outPutFileName


# Method to load data from the file
def loadData(fileName, stopWords, train=True):
    ctf_data = defaultdict(int)
    df_data = defaultdict(int)
    f = open(fileName)
    data = []
    cnt = 0

    tokensFile = open("tokens.dat", "w")

    # Calling method that yield every 1000 lines of file
    for l in printer(f, 10000, "%d lines"):

        text = l

        # Calls the tokenize method
        terms = Counter(tokenize(text))

        # Removes stop words
        for term in stopWords:
            if (terms.get(term)):
                terms.pop(term)

        # Create the ctf and df dictionaries
        for key, value in terms.items():
            df_data[key] += 1
            ctf_data[key] += value

        # Create the term list
        data.append(terms)

        cnt = cnt + 1
        if cnt % 100000 == 0:
            # Write to token file
            for d in data:
                json.dump(d, tokensFile)
                tokensFile.write("\n")

            data = []

    if cnt % 100000 != 0:
        for d in data:
            json.dump(d, tokensFile)
            tokensFile.write("\n")

    tokensFile.close()



    # Sort ctf and df dictionaries by values
    ctf_data = OrderedDict(sorted(ctf_data.items(), key=operator.itemgetter(1), reverse=True))

    df_data = OrderedDict(sorted(df_data.items(), key=operator.itemgetter(1), reverse=True))

    # Store values in file
    with open("ctf.dat", "w") as ctf_file:
        json.dump(ctf_data, ctf_file)

    with open("df.dat", "w") as df_file:
        json.dump(df_data, df_file)

    return ctf_data, df_data


def tokenize(text):
    regex = re.compile("\S*\d\S*")
    for c in string.punctuation:
        text = text.replace(c, "")

    text = regex.sub('', text)
    text = text.lower().split()

    return text

#Method to print the number of features traversed
def printer(X, n, stmt):
    count = 0
    for  x in X:
        count += 1
        if count % n == 0:
            print (stmt % count)
        yield x


#Method to design features
def featureDesign(N):

    fd = open("ctf.dat")
    fdf = open("df.dat")


    file = fd.read()
    dic = json.loads(file)
    fd.close()

    file = fdf.read()
    dic_df = json.loads(file)
    fdf.close()

    topk = 2000
    #Modify dictionary to sort it
    dic_mod = OrderedDict(sorted(dic.items(), key=operator.itemgetter(1), reverse=True))
    dic_mod = dict(dic_mod)
    #Fethc top 2000 terms
    top_terms = list(dic_mod.keys())[:topk]

    f = open("tokens.dat")

    data = np.zeros((N, topk))
    term_index = list(enumerate(top_terms))
    term_index = OrderedDict(reverse(term_index))
    top_terms_set = set(top_terms)


    for index, line in enumerate(f):
        terms = json.loads(line)
        #Find the top 2000 terms
        intersection_terms = top_terms_set.intersection(set(terms.keys()))
        for term in intersection_terms:
            # implementation of tf-idf
            a = terms[term] + 1
            b = (N+1-dic_df[term])
            c = (dic_df[term]+1)
            d = b/float(c)
            d = d+2
            e = math.log(d)
            f = a * e
            data[index][term_index[term]] = a * e

    return data

#Returns the dictionary as value,key
def reverse(dic):
    for k,v in dic:
        yield v,k

#Kmeans clustering method
def kMeans(X):
    kmeans = KMeans(n_clusters=10, random_state=0).fit(X)
    labels = kmeans.labels_
    indices = {}
    print([i for i, x in enumerate(labels) if x == 10])
    for l in range(0,11):
        indices[l] = [i for i, x in enumerate(labels) if x == l]
    for l in range(0,11):
        print(str(l)+":"+ str(len(indices[l])))
    print(indices)


if __name__=="__main__":
    #outFile = createKeywordsFile("annotationFile")
    outFile = "annotationFile-entities.txt"
    stopWords = []
    N = 407 #Change this
    ctf, df = loadData(outFile, stopWords, True)
    features = featureDesign(N)
    kMeans(features)