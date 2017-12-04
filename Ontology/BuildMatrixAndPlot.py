
# coding: utf-8

# Imports
from textrazor import TextRazor
import re
import nltk
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from itertools import islice

#Imports for plots
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.style as style
from ggplot import *

## Functions Definitions ##

def get_top_relevant_topics(response, thresh_score):
    top_relevant_topics = []
    topics = []
    topics = response.topics()
    top_relevant_topics = [topic for topic in topics if topic.score > thresh_score]
    return top_relevant_topics


#Splitting input file by number of words
def splitter(n, s,overrlapSize=0):
    pieces = s.split()
    # Use this line for sliding window. Here overlap size is 5
    #str = (" ".join(pieces[i-5:i+n-5]) for i in range(5, len(pieces), n-5))
    str = (" ".join(pieces[i:i+n]) for i in range(0, len(pieces), n))
    return list(str)

def mergeSentences(sentences, win_size):
    merged_sentences = []
    for i in range(0, len(sentences), win_size):
        merged_sentences.append(' '.join(sentences[i:i + win_size]))
    return merged_sentences

def createOverlaps(listWindows, overlap):

    i = 0
    if overlap == 0:
        return listWindows
    newWindows = []
    for window in listWindows:
        words=[]
        #print("Window:")
        #print(window)
        words = window.split()

        if(i==0):
            lastwords = words[len(words) - overlap:]
        else:
            words = lastwords + words
            lastwords = words[len(words) - overlap:]
            words = words[:len(words)-overlap]
        #print('len(words): ', len(words))
        #print('overlap: ', overlap)
        #print("words: "+str(words))

        #print("last words: " + str(lastwords))
        newWindows.append(' '.join(words))
        i += 1
    newWindows.append(' '.join(lastwords))
    #print(newWindows)
    return newWindows

#Build average score matrix
#Inputs: Textfile, phrase dictionary (exemplars), windowsize
#Use tokenizer to split data for clean text, and number of words for noisy text
def buildArray(txtFile, clusterdict,windowsize,overlapSize):
    ####################Done once##############################################################################
    import winsound
    duration = 2000  # millisecond
    freq = 440  # Hz
    import math
    import csv
    from textrazor import TextRazorAnalysisException
    from urllib.error import HTTPError
    inFile = open(txtFile, encoding='utf8').read()
    #For clean data
    ##############################################################
    #tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    #sentences = tokenizer.tokenize(inFile)
    ##############################################################
    ##############################################################

    #For noisy data to split by newline
    ##############################################################
    #sentences = inFile.split('\n')
    ##############################################################
    
    #For splitting by number of words
    ##############################################################
    sentences = splitter(num_words, inFile,overlapSize)
    mergedSentences = mergeSentences(sentences, windowsize)
    overlapWindows = createOverlaps(mergedSentences, overlapSize)
    sentences = overlapWindows

    ##############################################################
    #print(mergedSentences)

    num_columns = math.ceil(len(sentences)/windowsize) + 1
    df = pd.DataFrame(index=clusterdict, columns=range(1,num_columns))
    df = df.fillna(0)
    
    
    sentence_num=1
    no_cluster_dict = {}
    
    # Change textRazor keys when it runs out - here and in the catch exception section
    # Once the key runs out here, it will use the key in the catch exception clause
    
    textrazor_key = "cc9e748f39591c90171653b1ded93c6bd2b8a3da262b3a116aa6524d"
    #"190e8d56b61fb29f96c350d20a2947dc96218eba2bae86f7f17b212d"
    #"eb2a5954577a28b1ba8a97005281289493eee48179d2ac1883b685e6"
    #"5fdb0f64097c21849b8a9345fd8df3df5730ecfca4ab1c4e01a285b0"
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
            topics = get_top_relevant_topics(response, 0)
            cluster_val = 0
            for topic in topics:
                print(" ", topic.label, topic.score)
                topicDict.update({topic.label: topic.score})
            for key in clusterdict:
                print('Phrase Number =',row)
                phrase = clusterdict.get(key)
                print(clusterdict.get(key))
                val = 0
                
                sumProb = 0
                numMatch = 0
                for topicKey in topicDict:
                    sumProb += topicDict.get(topicKey)
                    #print('Topic Number = ',j)
                    #print(topicKey, topicDict.get(topicKey))
                    if topicKey in phrase:
                        print(topicKey, topicDict.get(topicKey))
                        val += topicDict.get(topicKey)
                        numMatch +=1
                print('val =', val)
                print('row =', row, 'sentence_num =', sentence_num)
                if numMatch!=0:
                    df[sentence_num].iloc[row] = val/numMatch
                else:
                    df[sentence_num].iloc[row] = val/1
                cluster_val += val
                row+= 1
            if cluster_val == 0:
                no_cluster_dict.update({sentence_num: topicDict})
            sentence_num += 1

        except (HTTPError, TextRazorAnalysisException) as e:
            
            winsound.MessageBeep(0)
            print("Error on " ,textrazor_key)
            textrazor_key = "49bb9fee0d6262d1f107bf5e9b35078fc0e5327a2d019bbaa908c442"
            print("i = ", i)
            #i = i - windowsize
            #print("i = ", i)
            print(e)
            continue
    df_csv_file = 'nooutput/output_ws'+str(windowsize)+ '.csv'
    no_cluster_csv_file = 'clusterout/no_cluster_dict_ws'+str(windowsize)+ '.csv'
    print("Output files:", df_csv_file, no_cluster_csv_file)
    df.to_csv(df_csv_file)
    with open(no_cluster_csv_file, 'w') as csv_file:
        writer_dict = csv.writer(csv_file)
        for key, value in no_cluster_dict.items():
            writer_dict.writerow([key, value])
    winsound.Beep(freq, duration)
    return df
        

#Build the hardMax matrix
def hardMax(df):
    df_max = pd.DataFrame(0, columns=df.columns, index=df.index)

    for column in df_max:
        index = df[column].idxmax()
        if df.loc[index,column]!=0:
            df_max.loc[index,column] = 1
    return df_max
    

#Parameters to change: 
#1. winsize: Line 168
#2. phraseDictionary: Line 172
#3. Input textfileName: Line 170
#4. Number of words in a sentence: num_words parameter: Line 169
# ## Plot facet plots and hardmax plots for specified windowsize

# Change according to required window size. 
# This will be used for executing buildArray function and for plot titles
winsize = 2
num_words = 20  # This will be used for splitter function in buildArray function
inputFile = 'redbarron_rashmi.txt'
overlapSize = 5
#Pass the exemplars here as phrase Dictionary
import featureMatrix as fm
phraseDict1 = fm.getOuput(inputFile)
'''    
{'Roy Brown (RAF officer)' : ['Roy Brown (RAF officer)'],
'Jagdstaffel' : ['Jagdstaffel', 'Dogfight', 'Fighter aircraft', 'Bloody April', 'Luftstreitkräfte', 'Jagdstaffel 2', 'Warfare', 'Trench warfare', 'Sopwith Camel', 'War', 'World War I', 'Lanoe Hawker', 'Aeronautics', 'Jagdstaffel 11', 'Royal Flying Corps', 'Battle of the Somme', 'Sopwith Aviation Company', 'Aircraft', 'Aerial warfare', 'Combat', 'National security', 'Aerospace engineering', 'Triplane', 'Nazi Germany', 'Violent conflict', 'Conflicts', 'Battle', 'Aircraft pilot', 'Luftwaffe'],
'Jagdgeschwader 1 (World War I)' : ['Jagdgeschwader 1 (World War I)'],
'Military' : ['Manfred von Richthofen', 'Oswald Boelcke', 'Flying ace', 'Synchronization gear', 'Max Immelmann', 'Cedric Popkin', 'Military forces', 'Military science', 'Military aviation', 'Military', 'Aviation', 'Cavalry', 'Airplane', 'Roland Georges Garros', 'International security', 'Battle of Vimy Ridge', 'Propeller (aeronautics)', 'Reconnaissance', 'Military organization', 'Artillery', 'Allies of World War II', 'Military operations', 'Flight', 'Verdun', 'Albatros D.II'],
'Fokker Dr.I' : ['Anthony Fokker', 'Fokker Dr.I', 'Fokker Eindecker fighters'],
'Air supremacy' : ['Air force', 'Royal Air Force', 'Air forces', 'Air supremacy'],
'No. 209 Squadron RAF' : ['No. 209 Squadron RAF']
}
'''
#Change the textfile name
# Execute buildArray function
my_df_array = buildArray(inputFile, phraseDict1, winsize,overlapSize)

# Execute hardmax function
dfmax = hardMax(my_df_array)

#Display and store the Facet plot
#Reshape the dataframe for plotting facets
df_new_Array = my_df_array
df_new_Array["id"] = df_new_Array.index
df_new_Array = pd.melt(df_new_Array, id_vars='id', value_vars=range(1,len(df_new_Array.columns)))
df_new_Array.rename(columns={'id': 'Topic', 'variable': 'Time', 'value': 'Average Score'}, inplace=True)

#Plot the facet graph with ggplot
style.use('fivethirtyeight')
p = ggplot(aes(x='Time', y='Average Score'), data=df_new_Array)  +geom_point(color = "#D55E00")  +facet_wrap('Topic') +ggtitle("Distribution of Topics across Time  (Window size = "+ str(winsize) + ")") +theme_bw() +theme(axis_title_x  = element_text(vjust = -0.02)) +theme(axis_title_y  = element_text(hjust = -0.02))

#Save plot as svg file
p.save('facetplots/facetsvgv' + str(winsize)+'.svg', dpi=1200)
print("Saved ggplot")

#Save hardmax plot
#Reshape dataframe for plotting hardmax
df_new_max = dfmax
df_new_max["id"] = df_new_max.index
df_new_max = pd.melt(df_new_max, id_vars='id', value_vars=range(1,len(df_new_max.columns)))
#Filter out 0 values
df_filter = df_new_max.loc[df_new_max['value'] != 0]
ylabel = list(df_filter.groupby("id").sum().sort_values('value').index.values)
df_filter.rename(columns={'id': 'Topic', 'variable': 'Time', 'value': 'Avg_Score'}, inplace=True)

#Plot the graph
sns.set_style("whitegrid", {'axes.grid' : False})
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.style as style
plt.figure(figsize=(20,10))

#sns.set(font_scale = 2)
strip_plot = sns.stripplot(x='Time' , y = 'Topic', data=df_filter,order=ylabel,size = 10, color = '#D55E00').set_title("Assigned Topics vs. Time (Window size = "+str(winsize)+")",fontsize=35)
strip_plot.axes.set_xlabel("Time",fontsize=30)
strip_plot.axes.set_ylabel("Topics",fontsize=30)
#strip_plot.axes.set_yticklabels(["Air forces","Roy Brown\n(RAF officer)","Imperial German\nArmy Air Service", "Jagdstaffel", "Military"], va ='center' )#,rotation = 50
#strip_plot.axes.set_axes_locator([1,2,3,4,5])
fig = strip_plot.get_figure()

#Save figure in svg format
plt.tight_layout()
fig.savefig('hardmaxplots/HardMaxsvgv'+str(winsize)+'.svg', dpi=1200)
plt.show()
print("Saved hardmaxplot")