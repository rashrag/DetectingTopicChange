from textrazor import TextRazor
import re
import nltk.data
import pandas as pd
import pickle
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import sys
import os

# Pickle data. We use this method to store list of topics to avoid unnecessarily calling textrazor
## Input: - data: Object to be pickled. In our case, list of topics tuples
##        - filename: name of file to store data
def saveData(data, filename):
    file = open(filename, 'wb')
    pickle.dump(data, file)
    file.close()

# Load pickled data. We use this method to load the list of topics that we previously pickled
## Input: - filename: name of file from which we load pickled data
## Output: unpickled data
def loadData(annotation_file):
    file = open(annotation_file+"_topics.dat", 'rb')
    data = pickle.load(file)

    end_of_file = False
    while not end_of_file:
        try:
            # Unpickle the next object.
            data = pickle.load(file)

        except EOFError:
            # Set the flag to indicate the end
            # of the file has been reached.
            end_of_file = True

    file.close()
    return data

# Read csv annotation file into a pandas dataframe
## Input: csv filename
## Output: dataframe with two columns: 'sentence', 'topic'
def readAnnotationFile(filename):
    file = open(filename, 'r')
    text_lines = []
    
    for line in file:
        text_lines.append(line)
        
    #return text_lines
    df = pd.read_csv(filename, header=0)
    return df

# Saves a matrix/array to a text file
## Input: - array
##        - filename to store the array
def saveArray(array, filename):
    file = open(filename, 'w')
    np.savetxt(filename, array, fmt='%2.3f')

# Merge sentences in a list of sentences based on window size.
## Input: - sentences: list of individual sentences
##        - win_size: window size
## Output: list of merged sentences
def mergeSentences(sentences, win_size):
    merged_sentences = []
    for i in range(0, len(sentences), win_size):
        merged_sentences.append(' '.join(sentences[i:i + win_size]))
    return merged_sentences

# Generate topics from list of sentences after merging them based on given window size,
# then store topics in a file named [annotation_filename]+'_topics.dar'
# topics are represented by a tuple of three elements: (sentence index, topic label, topic confidence score)
## Input: - text: list of individual sentences
##        - win_size: window size for merging sentences
def createTopicsFile(annotation_file, sentences, win_size, print=False):
    #TODO: write a method to iterate over API keys whenever an excepton is thrown from extRazor
    client = TextRazor("190e8d56b61fb29f96c350d20a2947dc96218eba2bae86f7f17b212d", extractors=["topics"])
    topics_list = []
    sentence_index = 0
    for sentence in sentences:
        #if print:
        #    print(sentence)
        response = client.analyze(sentence)

        topics_per_sent = []
        for topic in response.topics():
            topic_tuple = (sentence_index, topic.label, topic.score)
            topics_per_sent.append(topic_tuple)
        
        topics_list.append(topics_per_sent)
        sentence_index += 1
        
    saveData(topics_list, annotation_file+"_topics.dat")


# Creates the features matrix from list of topics. Currently, only topic confidence score is used.
## Input: - topic_list: list of topic tuples (sentence index, topic label, topic confidence score)
##        - num_instances: number of sentences (model instances)
## Output: nxd feature matrix, where n=number of sentences, d=number of unique topics
def createFeatures(topics_list, num_instances, annotation_filename):
    # Flatten the list of list of tuples (each sentence has a list of tuples)
    flat_list = [item for sublist in topics_list for item in sublist]
    # Extract topic labels from the flat list. So now we have a list of string topics
    topics = [topic[1] for topic in flat_list]
    print(flat_list[1:10])
    # Get unique topics
    unique_topics = list(set(topics))
    print("unique topics:")
    print(unique_topics)
    
    num_topics = len(unique_topics)
    feature_mat = np.zeros((num_instances, num_topics))

    # For every sentence (sentence_index) get each topic confidence score and update the correspoding cell
    # in the features matrix
    for sentence_index, topic, score in flat_list:
        feature_mat[sentence_index][unique_topics.index(topic)] = score

    #features_df = pd.DataFrame(np.asarray(feature_mat), columns = unique_topics)
    #print(features_df.head())
    #print('First sentence:')
    #print(features_df.loc[0,].sort_values(ascending=False))
    print('Feature matrix dimensions:', len(feature_mat), 'x', len(feature_mat[0]))
    # Save features matrix to a text file for easy display
    saveArray(feature_mat, annotation_filename+'_topicfeatures.txt')
    return feature_mat

# Utility method to print each list item in a new line
## Input: list to be printed
def printList(liist):
    for item in liist:
        print(item)

# Kmeans clustering method. K = 15
## Input: - X: features matrix
## Output: cluster labels
def kMeans(X):
    kmeans = KMeans(n_clusters=15, random_state=0).fit(X)
    labels = kmeans.labels_
    return labels
    indices = {}
    for l in range(0,11):
        indices[l] = [i for i, x in enumerate(labels) if x == l]
    for l in range(0,11):
        print(str(l)+":"+ str(len(indices[l])))
    print(indices)

    return indices

# Calculate evaluation metrics for the model. Currently, accuracy is used
## Input: - y_true: true labels
##        - y_pred: predicted labels
## Output: accuracy score
def evaluateModel(y_true, y_pred):
    accuracy = metrics.accuracy_score(y_true, y_pred)
    return accuracy

# Utility method to print message to stderr
def err_print(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

if __name__=="__main__":
    if len(sys.argv) != 2:
        err_print('Incorrect number of arguments. Must provide 1 arg: filename.')
        sys.exit(1)
    # csv annotations filename
    annotation_filename = "./annotations/rashmi_annotations(1).csv" #sys.argv[1]

    main_df = readAnnotationFile(annotation_filename)

    # Extract only filename without extension
    base = os.path.basename(annotation_filename)
    annotation_filename, file_extension = os.path.splitext(base)
    # Extract sentences and topic# into two separate vectors
    text = main_df['sentence']
    annotations = main_df['topic']
    #print(text[0:10])
    #print(annotations[0:10])
    #text = text[0:10]

    win_size = 4
    sentences = mergeSentences(text, win_size)
    #createTopicsFile(annotation_filename, sentences, text, win_size)
    num_sentences = len(sentences)

    # Load pickled list of topics
    topics = loadData(annotation_filename)
    training_data = createFeatures(topics, num_sentences, annotation_filename)
    clusters = kMeans(training_data)
    #accuracy = evaluateModel(annotations, clusters)
    #print('Model accuracy =', format(accuracy, '.3f'))

    fig, ax = plt.subplots()
    # Scatter-plot visualization
    i = range(0,num_sentences)
    frame = pd.DataFrame(clusters, index=i, columns=['cluster'])
    #print(frame)
    plt.scatter(x=frame.index, y=frame['cluster'], c=frame['cluster'], s=60)
    plt.xlabel('Time')
    plt.ylabel('Topic Cluster #')
    plt.title('15-means Clustering Output')

    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                    ax.get_xticklabels() + ax.get_yticklabels()):
       item.set_fontsize(24)
    plt.show()
