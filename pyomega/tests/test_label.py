"""Unit test for GravitySpy
"""

__author__ = 'Scott Coughlin <scott.coughlin@ligo.org>'

import os
import unittest2

import pyomega.ML.make_pickle_for_linux as make_pickle
import pyomega.ML.labelling_test_glitches as label_glitches
import pyomega.ML.train_classifier as train_classifier

TEST_IMAGES_PATH = os.path.join(os.path.split(__file__)[0], 'data',
'images')
MODEL_PATH = os.path.join(os.path.split(__file__)[0], '..', 'ML', 'trained_model')

Score = 0.96406441927 

class GravitSpyTests(unittest2.TestCase):
    """`TestCase` for the GravitySpy
    """
    def test_pickle(self):
        make_pickle.main(TEST_IMAGES_PATH + '/folder1/',
                         TEST_IMAGES_PATH + '/folder1/folder2/pickleddata/', 1, 1)
        scores, MLlabel = label_glitches.main(TEST_IMAGES_PATH + '/folder1/folder2/pickleddata/',
                                              MODEL_PATH,
                                              TEST_IMAGES_PATH + '/folder1/folder2/labeled/',
                                              0)
        self.assertEqual(float(scores[MLlabel + 1]), Score)

    def test_model(self):
        make_pickle.main(TEST_IMAGES_PATH + '/TrainingSet/',
                         TEST_IMAGES_PATH + '/pickleddata/', 1, 1)
        train_classifier.main(10, 10, 1, TEST_IMAGES_PATH + '/pickleddata/', TEST_IMAGES_PATH + '/model/', 2, 1)

if __name__ == '__main__':
    unittest2.main()
