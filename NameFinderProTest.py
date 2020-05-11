import unittest

from NameFinderPro import test_urllib

class NameFinderProTest(unittest.TestCase):

    def test_get_scripts_chunks(self):
        app = test_urllib()
        soup = app.read_html_from_file()
        strippedcontent = app.stripHTML(soup)
        list = app.parse(strippedcontent)

        self.assertEqual(len(list), 908, "Should be 908")


    if __name__ == '__main__':
        unittest.main()