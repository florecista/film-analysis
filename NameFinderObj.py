
from bs4 import BeautifulSoup
from abc import ABC, abstractmethod
import re
import os
import numpy as np

from nltk_opennlp.chunkers import OpenNERChunker
from nltk_opennlp.taggers import OpenNLPTagger
import configparser

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
    """
    The Creator class declares the factory method that is supposed to return an
    object of a Product class. The Creator's subclasses usually provide the
    implementation of this method.
    """

    @abstractmethod
    def factory_method(self):
        """
        Note that the Creator may also provide some default implementation of
        the factory method.
        """
        pass

    def get_names(self, list_) -> list:
        """
        Also note that, despite its name, the Creator's primary responsibility
        is not creating products. Usually, it contains some core business logic
        that relies on Product objects, returned by the factory method.
        Subclasses can indirectly change that business logic by overriding the
        factory method and returning a different type of product from it.
        """

        # Call the factory method to create a Product object.
        product = self.factory_method()

        # Now, use the product.
        result = "Creator: The same creator's code has just worked with {product.get_names()}"

        return result


class Names(ABC):
    """
    The Product interface declares the operations that all concrete products
    must implement.
    """

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


class TextNameFinder(NameFinder):
    """
    Note that the signature of the method still uses the abstract product type,
    even though the concrete product is actually returned from the method. This
    way the Creator can stay independent of concrete product classes.
    """

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

    print("Client: I'm not aware of the creator's class, but it still works.\n"
          "{creator.get_names(list_)}", end="")


if __name__ == "__main__":


    print("App: Launched with the TextNameFinderImpl.")
    client_code(TextNameFinderImpl())
    print("\n")

    print("App: Launched with the OpenNLPNameFinderImpl.")
    client_code(OpenNLPNameFinderImpl())