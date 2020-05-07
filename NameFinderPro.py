
import urllib.request
import os
from bs4 import BeautifulSoup
import re
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from nltk_opennlp.chunkers import OpenNERChunker
from nltk_opennlp.taggers import OpenNLPTagger
import configparser

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
        soup = BeautifulSoup(open("Platoon.html"), "lxml")
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

    def constructGraph(self, list_):
        G = nx.Graph()

        previous = ""
        for splitchunk in list_:
            if not self.isValidName(splitchunk.name):
                previous = ""
                continue
            current = splitchunk.name
            if previous != "" and previous != current:

                # TODO - add sentiment analysis
                # analyze relationship between previous and current
                # sentiment = getSentiment(splitchunk.dialog)
                # G.add_edge(previous, current, sentiment)

                G.add_edge(previous, current)
            previous = current

        return G

    def getNames(self, list_):
        names = []
        for splitchunk in list_:
            if not self.isValidName(splitchunk.name):
                continue
            names.append(splitchunk.name)

        return names

    def getNamesUnique(self, list_):
        x = np.array(list_)
        return np.unique(x).tolist()

    def getNamesAsString(self, list_):
        names = ' '.join(list_)
        return names

    def parse(self, content):
        list_ = []
        chunks = content.split("<b>")
        chunks = chunks[1:]
        
        for chunk in chunks:
            list_b = chunk.split("</b>")        
            if len(list_b)==1:
                name = list_b[0].replace("\n",'').strip()
                dialog = ""
                narrative = ""
            elif len(list_b)==2:        
                name = list_b[0].replace("\n",'').strip()
                if (len(list_b[1].split("\n\n"))==2):
                    list_b_n = list_b[1].split("\n\n")
                    dialog = list_b_n[0].replace("\n",'').strip()
                    narrative = list_b_n[1].replace("\n",'').strip()
                else:
                    dialog = list_b[0].replace("\n",'').strip()
                    narrative = ""

            # ugly but needed
            if " " in name and "(" in name:
                tokens = name.split()
                name = tokens[0]

            new_splitchunk = SplitChunk(name, dialog, narrative)
            list_.append(new_splitchunk)
        return list_

    def opennlp_test(self, content):
        names = []

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
        cp = OpenNERChunker(path_to_bin=os.path.join(opennlp_dir, 'bin'),
                            path_to_chunker=os.path.join(models_dir,
                                                         '{}-chunker.bin'.format(language)),
                            path_to_ner_model=os.path.join(models_dir,
                                                           '{}-ner-person.bin'.format(language)))

        tree = cp.parse(sentence)

        for st in tree.subtrees(filter=lambda x: x.label() == "NP" or x.label() == 'NNP'):
            leaves = st.leaves()
            if isinstance(leaves, list):
                for leaf in leaves:
                    if isinstance(leaf, tuple):
                        if 'NNP' in leaf[1]:
                            names.append(leaf[0])

        return names

    def showGraph(self, list_):
        graph = self.constructGraph(list_)
        nx.spring_layout(graph, k=0.15, iterations=20)

        plt.figure(3, figsize=(12, 8))
        nx.draw_networkx(graph)

        plt.show()

def main():
    app = test_urllib()

    # 1. Get HTML
    #soup = app.read_html_from_website()
    soup = app.read_html_from_file()

    # 2. Strip out unnecessary tags
    strippedcontent = app.stripHTML(soup)

    # 3. Parse HTML for list of Chunks
    list_ = app.parse(strippedcontent)

    # 4. Get list of names from Chunks
    print('Text Parsing for names')
    names = app.getNames(list_)
    print('#1 - ' + str(len(names)))
    uniqueListOfNames = app.getNamesUnique(names)
    print('#2 - ' + str(len(uniqueListOfNames)))

    # 5. Get list of names from OpenNLP, based off previous parsed list of names
    print('OpenNLP Parsing for names')
    namesFromOpenNLP = app.opennlp_test(app.getNamesAsString(names))
    print('#3 - ' + str(len(namesFromOpenNLP)))
    uniqueListOfNamesFromOpenNLP = app.getNamesUnique(namesFromOpenNLP)
    print('#4 - ' + str(len(uniqueListOfNamesFromOpenNLP)))

    # 6. Add names to graph for visualization
    app.showGraph(list_)

if __name__ == "__main__":
    main()
