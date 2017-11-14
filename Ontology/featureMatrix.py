import numpy as np
import affinityPropogation as af
from textrazor import TextRazor

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

if __name__=="__main__":
    #topicsli = ['American Television Series','Television Drama Series','Romantic Television Series','Sustainable Energy','Renewable Energy']
    txtFile = "textFile"
    clusterdict = getOuput(txtFile)
    print(clusterdict)