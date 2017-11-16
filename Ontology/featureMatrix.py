import numpy as np
import affinityPropogation as af
from textrazor import TextRazor
import math
import csv
from textrazor import TextRazorAnalysisException
from urllib.error import HTTPError
    
def splitTopics(topicsli):
    split = [words for segments in topicsli for words in segments.split()]
    split = list(set(split))
    print(split)
    return split

def createFeatureMatrix(split,topicsli):
    matrix = np.zeros((len(topicsli),len(split)))
    i = 0
    for topic in topicsli:
        words = topic.split()
        for l in words:
            j = split.index(l)
            matrix[i][j] = 1
        i+= 1

    return matrix

def get_top_relevant_topics(response, thresh_score):
    top_relevant_topics = []
    topics = []
    topics = response.topics()
    top_relevant_topics = [topic for topic in topics if topic.score > thresh_score]

    return top_relevant_topics


def getTopics(fileName):
    rashkey = "cc9e748f39591c90171653b1ded93c6bd2b8a3da262b3a116aa6524d"
    outPutFileName = fileName
    client = TextRazor( "fdd39e9dde7d84983bbe964c514035f3c483d7df5996f62541ef76bc",
                       extractors=["topics"])
    with open(fileName, 'r') as content_file:
        content = content_file.read()
    response = client.analyze(content)
    topics = []
    topics = get_top_relevant_topics(response, 0.5)
    topics = [topic.label for topic in topics]
    return topics

def getOuput(txtFile):
    topicsli = getTopics(txtFile)
    split = splitTopics(topicsli)
    matrix = createFeatureMatrix(split, topicsli)
    clusterCenterIndex, labels, noOfCluster = af.affinityProp(matrix)
    print(noOfCluster)
    indices = {}
    topicind = {}
    for l in range(noOfCluster):
        indices[l] = [i for i, x in enumerate(labels) if x == l]
        topicind[l] = [topicsli[i] for i, x in enumerate(labels) if x == l]
    print("************************")
    clusterdict = {}
    for ind in clusterCenterIndex:
        print(topicsli[ind] + " : " + str(topicind[labels[ind]]))
        clusterdict[topicsli[ind]] = topicind[labels[ind]]
    return  clusterdict


def buildArray(txtFile, clusterdict,windowsize):
    ####################Done once##############################################################################
    inFile = open(txtFile, encoding='utf8').read()
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    sentences = tokenizer.tokenize(inFile)
    num_columns = math.ceil(len(sentences)/windowsize) + 1
    df = pd.DataFrame(index=phraseDict, columns=range(1,num_columns))
    df = df.fillna(0)
    sentence_num=1
    no_cluster_dict = {}
    textrazor_key = "eb2a5954577a28b1ba8a97005281289493eee48179d2ac1883b685e6"
    ###########################################################################################################
    
    for i in range(0,len(sentences),windowsize):
        #while i in range(0,len(sentences),windowsize):
        try:
            print("i = ", i)
            client = TextRazor(textrazor_key, extractors=["topics"])
            row=0
            sentence = str(sentences[i:i+windowsize])
            print(i , sentence)
            topicDict = {}
            response = client.analyze(sentence)
            topics = get_top_relevant_topics(response, 0.5)
            cluster_val = 0
            for topic in topics:
                print(" ", topic.label, topic.score)
                topicDict.update({topic.label: topic.score})
            for key in clusterdict:
                print('Phrase Number =',row)
                phrase = clusterdict.get(key)
                print(clusterdict.get(key))
                val = 0

                for topicKey in topicDict:
                    #print('Topic Number = ',j)
                    #print(topicKey, topicDict.get(topicKey))
                    if topicKey in phrase:
                        print(topicKey, topicDict.get(topicKey))
                        val += topicDict.get(topicKey)
                print('val =', val)
                print('row =', row, 'sentence_num =', sentence_num)
                df[sentence_num].iloc[row] = val
                cluster_val += val
                row+= 1
            if cluster_val == 0:
                no_cluster_dict.update({sentence_num: topicDict})
            sentence_num += 1

        except (HTTPError, TextRazorAnalysisException) as e:
            print("Error on " ,textrazor_key)
            textrazor_key = "fdd39e9dde7d84983bbe964c514035f3c483d7df5996f62541ef76bc"#"324337f35ea1876aae377a6f1191ab5a5b9aaced32f9a4e3634e74c5"
            print("i = ", i)
            #i = i - windowsize
            #print("i = ", i)
            print(e)
            continue
    df_csv_file = 'output_ws'+str(windowsize)+ '.csv'
    no_cluster_csv_file = 'no_cluster_dict_ws'+str(windowsize)+ '.csv'
    print("Output files:", df_csv_file, no_cluster_csv_file)
    df.to_csv(df_csv_file)
    with open(no_cluster_csv_file, 'w') as csv_file:
        writer_dict = csv.writer(csv_file)
        for key, value in no_cluster_dict.items():
            writer_dict.writerow([key, value])
    return df
    
def hardMax(df):
    df_max = pd.DataFrame(0, columns=df.columns, index=df.index)

    for column in df_max:
        index = df[column].idxmax()
        if df.loc[index,column]!=0:
            df_max.loc[index,column] = 1
    return df_max
        

if __name__=="__main__":
    #topicsli = ['American Television Series','Television Drama Series','Romantic Television Series','Sustainable Energy','Renewable Energy']
    txtFile = "textFile"
    clusterdict = getOuput(txtFile)
    print(clusterdict)
    my_df = buildArray(txtFile, clusterdict, 3)
    max_df = hardMax(my_df)
    