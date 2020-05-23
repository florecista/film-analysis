import urllib.request
import os
from bs4 import BeautifulSoup
import re
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from ClassifierBuilder import ClassifierBuilder
from nltk_opennlp.taggers import OpenNLPTagger
import configparser

import nltk
from nltk.corpus import names 
nltk.download('names')

import pickle

class SplitChunk:
    def __init__(self, name, dialog, narrative):
        self.name = name
        self.dialog = dialog
        self.narrative = narrative

class test_urllib():

    def read_html_from_website(self):
        movie_url = 'http://www.imsdb.com/scripts/Platoon.html'

        try:
            request = urllib.request.Request(movie_url)
            webpage_bytes = urllib.request.urlopen(request)
            soup = BeautifulSoup(webpage_bytes, 'lxml')
        except urllib.error.URLError as err:
            print('Catched an URLError while fetching the URL:', err)
            print()
            pass
        except ValueError as err:
            print('Catched a ValueError while fetching the URL:', err)
            print()
            pass
        except:
            print('Catched an unrecognized error')
            raise

        return soup

    def read_html_from_file(self):
        page = open("Platoon.html")
        soup = BeautifulSoup(page.read(), "html.parser")
        page.close()
        return soup

    def stripHTML(self, soup):
        pre = soup.findAll('pre')[1]
        string = str(pre)
        return string

    def isValidName(self, name):
        stoplist = ["omit","dissovle","fade","ext","int","day","...","cut","close","med","-","shot", "it", "up","shoot","get","back","here"]
                
        nameLowerCase = str(name).lower()
        if any(element in nameLowerCase for element in stoplist):
            return False
        if re.match(".*\d.*", nameLowerCase):
            return False

        return True

    def constructGraph(self, dataFrame, isDirected):
        #get names from dataframe
        names = dataFrame["name"]
        namesFromOpenNLP = self.opennlp_test(self.getNamesAsString(names))
        uniqueListOfNamesFromOpenNLP = self.getNamesUnique(namesFromOpenNLP)

        if(isDirected) :
            G = nx.DiGraph()
        else:
            G = nx.Graph()

        dict_ = {}
        previous = ""
        for index,splitchunk in dataFrame.iterrows():
            if not self.isValidName(splitchunk["name"]) or splitchunk["name"] not in uniqueListOfNamesFromOpenNLP:
                previous = ""
                continue
            #get nameq from dataframe row
            current = splitchunk["name"]
            if previous != "" and previous != current:

                sentiment = splitchunk["Sentiment"]

                # not sure if this is great
                if(sentiment == 'pos'):
                    sentimentDouble = 1.0
                else:
                    sentimentDouble = -1.0

                #Create a dictionary to sum the sentimentDouble of the same two persons
                dict_key  = previous + "&" + current
                dict_key_inv = current + "&" + previous
                inv_ = False
                
                if dict_key not in dict_.keys():
                    if dict_key_inv in dict_.keys():
                        inv_ = True
                        dict_[dict_key_inv] = dict_[dict_key_inv] + sentimentDouble
                    else:
                        dict_[dict_key] = sentimentDouble
                    
                else:
                    dict_[dict_key] = dict_[dict_key] + sentimentDouble

                if inv_:
                    sentimentDouble = dict_[dict_key_inv]
                    previous = dict_key_inv.split("&")[1]
                    current = dict_key_inv.split("&")[0]
                else:
                    sentimentDouble = dict_[dict_key]
                    previous = dict_key_inv.split("&")[0]
                    current = dict_key_inv.split("&")[1]

                #Add color to edges
                if sentimentDouble <0:
                    color = 'r'
                elif sentimentDouble >0:
                    color = 'b'
                #Add neutral case
                else:
                    color = 'g'
    
                G.add_edge(previous, current, color=color, weight=sentimentDouble)

            previous = current

        return G

    def getNames(self, dataFrame):
        names = []
        for index,splitchunk in dataFrame.iterrows():
            if not self.isValidName(splitchunk["name"]):
                continue
            names.append(splitchunk["name"])

        return names

    def getNamesUnique(self, list_):
        x = np.array(list_)
        return np.unique(x).tolist()

    def getNamesAsString(self, list_):
        names = ' '.join(list_)
        return names

    def getValidNames(self, sentence):
        list_ =[]
        names_ = []
        name_set = set(names.words()) 
        for word,tag in sentence:
            for h in name_set:
                if word.upper() == h.upper():
                    list_.append((word,tag))
                    break
                
        for el in list_:
            if el[1] in ['NNP','NP']:
                names_.append(el[0])

        return names_
    
    def parse(self, content):
        list_ = []
        content = content.replace("<b>\t...\n</b>","\n")
        chunks = content.split("<b>")
        chunks = chunks[1:]
        
        
        for chunk in chunks:
            #print(chunk)
            list_b = chunk.split("</b>",1)        
            if len(list_b)==1:
                name = list_b[0].replace("\n",'').strip()
                dialog = ""
                narrative = ""
            elif len(list_b)==2:
                name = list_b[0].replace("\n",'').strip()
                if (len(list_b[1].split("\n\n",1))==1):
                    dialog = list_b[1].replace("\n",'').strip()
                    narrative = ""
                    
                else:
                    list_b_n = list_b[1].split("\n\n",1)
                    dialog = list_b_n[0].replace("\n",'').strip()
                    narrative = list_b_n[1].replace("\n",'').strip()
                    
            # ugly but needed
            if " " in name and "(" in name:
                tokens = name.split()
                name = tokens[0]
            
            new_splitchunk = SplitChunk(name, dialog, narrative)
            list_.append(new_splitchunk)
        return list_

    def getDataFrame(self, list_):
        data = []
        classifier = ClassifierBuilder()
        for splitchunk in list_:
            new_row = [splitchunk.name,splitchunk.dialog,splitchunk.narrative,classifier.getSentiment(splitchunk.dialog)]
            data.append(new_row)
        dataframe = pd.DataFrame(data, columns = ['name', 'dialog','narrative','Sentiment'])
        return dataframe

    def opennlp_test(self, content):
        config = configparser.ConfigParser()
        config.read('settings.ini')
        
        opennlp_dir = config['options']['opennlp_dir']
        models_dir = config['options']['models_dir']

        language = 'en'
        tt = OpenNLPTagger(language=language,
                           path_to_bin=os.path.join(opennlp_dir, 'bin'),
                           path_to_model=os.path.join(models_dir, 'en-pos-maxent.bin'))
        phrase = str(content)
        sentence = tt.tag(phrase)
        return self.getValidNames(sentence)

    def drawGraph(self, G, pos, measures, measure_name, showLabels, colorEdges):
        if(showLabels):
            labels = {}
            for k in G.nodes():
                labels[k] = str(k)
            nx.draw_networkx_labels(G, pos=pos, font_color='green', labels=labels)

        pos_attrs = {}
        for node, coords in pos.items():
            pos_attrs[node] = (
            coords[0] + 0.1 * (-1) * np.sign(coords[0]), coords[1] + 0.1 * (-1) * np.sign(coords[1]))

        nodes = nx.draw_networkx_nodes(G, pos_attrs, node_size=250, cmap=plt.cm.plasma, node_color=list(measures.values()),
                                             nodelist=measures.keys())
        nodes.set_norm(mcolors.SymLogNorm(linthresh=0.01, linscale=1))

        if(colorEdges):
            colorEdges = G.edges()
            colors = [G[u][v]['color'] for u, v in colorEdges]
            edges = nx.draw_networkx_edges(G, pos_attrs, edges=colorEdges, edge_color=colors)
        else:
            edges = nx.draw_networkx_edges(G, pos_attrs)

        plt.title(measure_name)
        plt.colorbar(nodes)
        plt.axis('off')
        plt.show()

    def buildDirectedGraph(self, dataFrame):
        G = self.constructGraph(dataFrame, True)
        pos = nx.spring_layout(G)
        measures = nx.pagerank(G, alpha=0.85)
        showLabels = True
        colorEdges = True
        self.drawGraph(G,pos, measures, 'DiGraph PageRank', showLabels, colorEdges)

    def buildUndirectedGraph(self, dataFrame):
        G = self.constructGraph(dataFrame, False)

        pos = nx.circular_layout(G)
        plt.figure(3, figsize=(12, 8))

        measures = nx.degree_centrality(G)
        showLabels = True # Only show labels for Circuler Layout
        colorEdges = False
        self.drawGraph(G,pos, measures, 'Degree Centrality', showLabels, colorEdges)
    #
    # def loadClassifier(self):
    #     try:
    #         f = open('classifier.pickle', 'rb')
    #     except FileNotFoundError:
    #         print('Classifier not found')
    #         exit()
    #
    #     classifier = pickle.load(f)
    #     f.close()

def main():
    app = test_urllib()

    # 1. Get HTML
    #soup = app.read_html_from_website()
    soup = app.read_html_from_file()

    # 2. Strip out unnecessary tags
    strippedcontent = app.stripHTML(soup)

    # 3. Parse HTML for list of Chunks
    list_ = app.parse(strippedcontent)

    # 4. Get DataFrame
    dataFrame = app.getDataFrame(list_,)

    # 5. Add names to graph for visualization
    #app.buildUndirectedGraphWithNodeLabels(dataFrame)
    #app.buildUndirectedGraph(dataFrame)
    app.buildUndirectedGraph(dataFrame)

if __name__ == "__main__":
    main()
