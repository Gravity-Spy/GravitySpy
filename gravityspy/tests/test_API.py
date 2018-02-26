"""Unit test for GravitySpy
"""

__author__ = 'Scott Coughlin <scott.coughlin@ligo.org>'

import os
import unittest2

class GravitySpyTests(unittest2.TestCase):
    """`TestCase` for the GravitySpy
    """
    def test_label(self):

        list_of_images = []
        for ifile in os.listdir(TEST_IMAGES_PATH):
            if 'spectrogram' in ifile:
                list_of_images.append(ifile)

        image_dataDF = pd.DataFrame()
        for idx, image in enumerate(list_of_images):
            image_data = make_pickle.main(os.path.join(
                                                       TEST_IMAGES_PATH,
                                                       image),
                                          resolution=0.1)

            image_dataDF[image] = [image_data]

        # Now label the image
        scores, MLlabel = label_glitches.label_glitches(
                                                        image_dataDF,
                                                        '{0}'.format(
                                                              MODEL_PATH),
                                                        [47, 57],
                                                        False)

        confidence = float(scores[0][MLlabel])
        self.assertEqual(confidence, SCORE)
