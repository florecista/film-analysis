
from bs4 import BeautifulSoup
from abc import ABC, abstractmethod
import re
import os
import numpy as np

from nltk_opennlp.taggers import OpenNLPTagger
import configparser
import nltk
from nltk.corpus import names
nltk.download('names')

class SplitChunk:
    def __init__(self, name, dialog, narrative):
        self.name = name
        self.dialog = dialog
        self.narrative = narrative

class Utils:

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

    def parse(self, content):
        list_ = []
        chunks = content.split("<b>")
        chunks = chunks[1:]

        for chunk in chunks:
            list_b = chunk.split("</b>")
            if len(list_b) == 1:
                name = list_b[0].replace("\n", '').strip()
                dialog = ""
                narrative = ""
            elif len(list_b) == 2:
                name = list_b[0].replace("\n", '').strip()
                if (len(list_b[1].split("\n\n")) == 2):
                    list_b_n = list_b[1].split("\n\n")
                    dialog = list_b_n[0].replace("\n", '').strip()
                    narrative = list_b_n[1].replace("\n", '').strip()
                else:
                    dialog = list_b[0].replace("\n", '').strip()
                    narrative = ""

            # ugly but needed
            if " " in name and "(" in name:
                tokens = name.split()
                name = tokens[0]

            new_splitchunk = SplitChunk(name, dialog, narrative)
            list_.append(new_splitchunk)
        return list_

class NameFinder(ABC):

    @abstractmethod
    def factory_method(self):
        pass

    def get_names(self, list_) -> list:

        name_finder = self.factory_method()

        # Now, use the name_finder.
        result = "Creator: The same creator's code has just worked with {name_finder.get_names()}"

        return result


class Names(ABC):

    @abstractmethod
    def get_names(self, list_) -> list:
        pass

class TextNameFinderImpl(Names):
    def get_names(self, list_) -> list:
        names = []
        utils = Utils()
        for splitchunk in list_:
            if not utils.isValidName(splitchunk.name):
                continue
            names.append(splitchunk.name)

        return names

class OpenNLPNameFinderImpl(Names):
    def getValidNames(self, sentence):
        list_ = []
        names_ = []
        name_set = set(names.words())
        for word, tag in sentence:
            for h in name_set:
                if word.upper() == h.upper():
                    list_.append((word, tag))
                    break

        for el in list_:
            if el[1] in ['NNP', 'NP']:
                names_.append(el[0])

        return names_

    def get_names(self, list_) -> list:
        names = []

        utils = Utils()
        stringList = []
        for splitchunk in list_:
            if not utils.isValidName(splitchunk.name):
                continue
            stringList.append(splitchunk.name)

        content = ' '.join(stringList)

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

class TextNameFinder(NameFinder):

    def factory_method(self) -> TextNameFinderImpl:
        return TextNameFinderImpl()

class OpenNLPNameFinder(NameFinder):
    def factory_method(self) -> OpenNLPNameFinderImpl:
        return OpenNLPNameFinderImpl()

def client_code(creator: NameFinder) -> None:
    utils = Utils()
    soup = utils.read_html_from_file()
    strippedcontent = utils.stripHTML(soup)
    list_ = utils.parse(strippedcontent)

    names = creator.get_names(list_)

    print('#1 - ' + str(len(names)))
    x = np.array(names)
    uniqueListOfNames = np.unique(x).tolist()
    print('#2 - ' + str(len(uniqueListOfNames)))

if __name__ == "__main__":

    print("App: Launched with the TextNameFinderImpl.")
    client_code(TextNameFinderImpl())

    print("App: Launched with the OpenNLPNameFinderImpl.")
    client_code(OpenNLPNameFinderImpl())