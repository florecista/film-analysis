
import urllib.request
from bs4 import BeautifulSoup
import re
import networkx as nx
import matplotlib.pyplot as plt

class SplitChunk:
    def __init__(self, name, dialog, narrative):
        self.name = name
        self.dialog = dialog
        self.narrative = narrative

class test_urllib():

    def read_url(self):
        movie_url = 'https://www.imsdb.com/scripts/Platoon.html'

        try:
            request = urllib.request.Request(movie_url)
            webpage_bytes = urllib.request.urlopen(request)
            soup = BeautifulSoup(webpage_bytes, 'lxml')
            is_webpage_fetched = True
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

    def stripHTML(self, soup):
        pre = soup.findAll('pre')[1]
        string = str(pre)
        return string

    def isValidName(self, name):
        stoplist = ["omit","dissovle","fade","ext","int","day","...","cut","close","med","-","shot"]
        nameLowerCase = str(name).lower()
        if any(element in nameLowerCase for element in stoplist):
            return False
        if re.match(".*\d.*", nameLowerCase):
            return False

        return True

    def constructGraph(self, list):
        G = nx.Graph()

        previous = ""
        for splitchunk in list:
            if not self.isValidName(splitchunk.name):
                previous = ""
                continue
            current = splitchunk.name
            if previous != "" and previous != current:
                relationship = 0.0
                G.add_edge(previous, current)
            previous = current

        return G

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

            new_splitchunk = SplitChunk(name, dialog, narrative)
            list_.append(new_splitchunk)
        return list_

def main():
    app = test_urllib()
    soup = app.read_url()
    strippedcontent = app.stripHTML(soup)
    list = app.parse(strippedcontent)
    graph = app.constructGraph(list)
    nx.draw_networkx(graph)
    plt.show()

if __name__ == "__main__":
    main()
