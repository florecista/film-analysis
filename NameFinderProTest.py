import unittest

from NameFinderPro import test_urllib

class NameFinderProTest(unittest.TestCase):

    def test_get_scripts_chunks(self):
        app = test_urllib()
        soup = app.read_html_from_file()
        strippedcontent = app.stripHTML(soup)
        list = app.parse(strippedcontent)

        self.assertEqual(len(list), 908, "Should be 908")

    def test_get_names_from_scripts_chunks(self):
        app = test_urllib()
        soup = app.read_html_from_file()
        strippedcontent = app.stripHTML(soup)
        list = app.parse(strippedcontent)
        names = app.getNames(list)
        uniqueListOfNames = app.getNamesUnique(names)

        self.assertEqual(len(uniqueListOfNames), 73, "Should be 73")

    def test_get_names_from_open_nlp(self):
        app = test_urllib()
        soup = app.read_html_from_file()
        strippedcontent = app.stripHTML(soup)
        list = app.parse(strippedcontent)
        names = app.getNames(list)
        namesFromOpenNLP = app.opennlp_test(app.getNamesAsString(names))
        uniqueListOfNamesFromOpenNLP = app.getNamesUnique(namesFromOpenNLP)

        self.assertEqual(len(uniqueListOfNamesFromOpenNLP), 20, "Should be 20")


    if __name__ == '__main__':
        unittest.main()