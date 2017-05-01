"""Unit test for GravitySpy
"""

__author__ = 'Scott Coughlin <scott.coughlin@ligo.org>'

import os
import ML.make_pickle_for_linux as make_pickle
import ML.labelling_test_glitches as label_glitches
TEST_IMAGES_PATH = os.path.join(os.path.split(__file__)[0], 'data',
'images')
print(TEST_IMAGES_PATH)

Score = 0.964064

class GravitSpyTests(unittest2.TestCase):
    """`TestCase` for the GravitySpy
    """
    def test_pickle(self):
        make_pickle.main(TEST_IMAGES_PATH + '/folder1/folder2/',
                         TEST_IMAGES_PATH + '/folder1/folder2/pickleddata/', 1, 0)
        scores, MLlabel = label_glitches.main(TEST_IMAGES_PATH + '/folder1/folder2/pickleddata/',
                                              '../src/ML/trained_model/'.format(pathToModel),
                                              TEST_IMAGES_PATH + '/folder1/folder2/labeled/',
                                              0)
        self.assertEqual(scores[MLlabel], Score)


if __name__ == '__main__':
    unittest2.main()
