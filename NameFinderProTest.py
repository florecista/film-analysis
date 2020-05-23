import unittest

from NameFinderPro import test_urllib

class NameFinderProTest(unittest.TestCase):

    def test_get_scripts_chunks(self):
        app = test_urllib()
        soup = app.read_html_from_file()
        strippedcontent = app.stripHTML(soup)
        list = app.parse(strippedcontent)

        self.assertEqual(len(list), 904, "Should be 904")

    def test_get_dataFrame_from_scripts_chunks(self):
        app = test_urllib()
        soup = app.read_html_from_file()
        strippedcontent = app.stripHTML(soup)
        list = app.parse(strippedcontent)
        dataFrame = app.getDataFrame(list, )

        self.assertEqual(len(dataFrame), 904, "Should be 904 rows")

    def test_get_main_characters_from_dataFrame(self):
        app = test_urllib()
        soup = app.read_html_from_file()
        strippedcontent = app.stripHTML(soup)
        list = app.parse(strippedcontent)
        dataFrame = app.getDataFrame(list, )
        chris = dataFrame.loc[dataFrame['name'] == 'CHRIS']
        barnes = dataFrame.loc[dataFrame['name'] == 'BARNES']
        elias = dataFrame.loc[dataFrame['name'] == 'ELIAS']

        self.assertEqual(len(chris), 117, "Should be 117 instances of Chris")
        self.assertEqual(len(barnes), 69, "Should be 69 instances of Barnes")
        self.assertEqual(len(elias), 66, "Should be 904 instances of Elias")

    def test_get_overall_sentiment_chris_from_dataFrame(self):
        app = test_urllib()
        soup = app.read_html_from_file()
        strippedcontent = app.stripHTML(soup)
        list = app.parse(strippedcontent)
        dataFrame = app.getDataFrame(list, )
        pos = dataFrame.loc[(dataFrame['name'] == 'CHRIS') & (dataFrame['Sentiment'] == 'pos')]
        neg = dataFrame.loc[(dataFrame['name'] == 'CHRIS') & (dataFrame['Sentiment'] == 'neg')]

        self.assertEqual(len(pos), 38, "Positive sentiment of Chris should be 38")
        self.assertEqual(len(neg), 79, "Negative sentiment of Chris should be 79")

    def test_get_names_from_scripts_chunks(self):
        app = test_urllib()
        soup = app.read_html_from_file()
        strippedcontent = app.stripHTML(soup)
        list = app.parse(strippedcontent)
        dataFrame = app.getDataFrame(list, )
        names = app.getNames(dataFrame)
        uniqueListOfNames = app.getNamesUnique(names)

        self.assertEqual(len(uniqueListOfNames), 73, "Should be 73")

    def test_get_names_from_open_nlp(self):
        app = test_urllib()
        soup = app.read_html_from_file()
        strippedcontent = app.stripHTML(soup)
        list = app.parse(strippedcontent)
        dataFrame = app.getDataFrame(list, )
        names = app.getNames(dataFrame)
        namesFromOpenNLP = app.opennlp_test(app.getNamesAsString(names))
        uniqueListOfNamesFromOpenNLP = app.getNamesUnique(namesFromOpenNLP)

        self.assertEqual(len(uniqueListOfNamesFromOpenNLP), 20, "Should be 20")


    if __name__ == '__main__':
        unittest.main()