# -*- coding: utf-8 -*-
# Copyright (C) Scott Coughlin (2017-)
#
# This file is part of gravityspy.
#
# gravityspy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# gravityspy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with gravityspy.  If not, see <http://www.gnu.org/licenses/>.

from gwpy.timeseries import TimeSeries
from gwpy.table import GravitySpyTable
from gwpy.segments import Segment

from ..plot.plot import plot_qtransform
from ..api.project import GravitySpyProject
from ..utils import log
from ..utils import utils

import ..ml.read_image as read_image
import ..ml.labelling_test_glitches as label_glitches

import os
import numpy
import pandas

def classify(event_time, channel_name,
             project_info_pickle, path_to_cnn,
             verbose=True, **kwargs):
    """classify an excess noise event

    Parameters:

        specsgrams (list):
            A list of `gwpy.spectrogram.Spectrogram` objects

        plot_normalized_energy_range (array):
            The min and max of the colorbar for the plots

        plot_time_ranges (array):
            The duration assosciated with each plot to be made

        detector_name (str):
            What detetor where these spectrograms from

        start_time (float):
            What was the start time of the data used for these spectrograms
            this effects what the plot title is (ER10 O1 O2 etc)

    Returns:

        ind_fig_all
            A list of individual spectrogram plots
        super_fig
            A single `plot` object contianing all spectrograms
    """

    if not os.path.isfile(path_to_cnn):
        raise ValueError('The provided cnn model does not '
                         'exist.')

    logger = log.Logger('Gravity Spy: Classifying Event')

    # Extract Detector Name From Channel Name
    detector_name = channel_name.split(':')[0]

    # Parse Keyword Arguments
    config = kwargs.pop('config', utils.GravitySpyConfigFile())
    plot_directory = kwargs.pop('plot_directory', 'plots')
    timeseries = kwargs.pop('timeseries', None)
    source = kwargs.pop('source', None)
    id_string = kwargs.pop('ID', '{0:.9f}'.format(event_time))

    # Parse Ini File
    plot_time_ranges = config.plot_time_ranges
    plot_normalized_energy_range = config.plot_normalized_energy_range

    # Cropping the results before interpolation to save on time and memory
    # perform the q-transform
    specsgrams, q_value = utils.make_q_scans(event_time, config=config,
                                             **kwargs)

    utils.save_q_scans(plot_directory, specsgrams,
                       plot_normalized_energy_range, plot_time_ranges,
                       detector_name, event_time, id_string)

    # Since we created the images in a
    # special temporary directory we can run os.listdir to get there full
    # names so we can convert the images into ML readable format.
    scores_table = utils.label_q_scans(plot_directory, path_to_cnn,
                                       project_info_pickle)

    return scores_table
