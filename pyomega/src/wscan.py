#!/usr/bin/env python

# ---- Import standard modules to the python path.

from __future__ import division

import sys
import os
import random
import string
import shutil
import ConfigParser
import optparse
import json
import rlcompleter
import pdb
import operator

from panoptes_client import *

import pandas as pd
import numpy as np

from scipy import signal
from scipy.interpolate import InterpolatedUnivariateSpline

from sqlalchemy.engine import create_engine

from matplotlib import use
use('agg')
from matplotlib import (pyplot as plt, cm)
from matplotlib.ticker import ScalarFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from gwpy.plotter import rcParams
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)

from glue import datafind

from gwpy.timeseries import TimeSeries
from gwpy.plotter import Plot

import ML.make_pickle_for_linux as make_pickle
import ML.labelling_test_glitches as label_glitches
import API.projectStructure as Structure

pdb.Pdb.complete = rlcompleter.Completer(locals()).complete

###############################################################################
##########################                             ########################
##########################   Func: parse_commandline   ########################
##########################                             ########################
###############################################################################
# Definite Command line arguments here

def parse_commandline():
    """Parse the options given on the command-line.
    """
    parser = optparse.OptionParser()
    parser.add_option("--inifile", help="Name of ini file of params")
    parser.add_option("--eventTime", type=float, help="Trigger time of the glitch")
    parser.add_option("--uniqueID", action="store_true", default=True,
                      help="Is this image being generated for the GravitySpy project,\
                      if so you must assign a uniqueID string to label the images instead of \
                      GPS time")
    parser.add_option("--ID", help="The uniqueID string to be supplied with --uniqueID")
    parser.add_option("--outDir", help="Outdir for images")
    parser.add_option("--NSDF", action="store_true", default=False,
                      help="If not on LIGO custers you will want to use NSDF server")
    parser.add_option("--pathToModel", default='./ML/trained_model/',
                      help="Path to folder containing trained model")
    parser.add_option("--runML", action="store_true", default=False,
                      help="Run the ML classifer on the omega scans")
    parser.add_option("--verbose", action="store_true", default=False, help="Run in Verbose Mode")
    parser.add_option("--HDF5", action="store_true", default=False,
                      help="Store triggers in local HDF5 table format")
    parser.add_option("--PostgreSQL", action="store_true", default=False,
                      help="Store triggers in a remote PostgreSQL DB")
    opts, args = parser.parse_args()


    return opts

###############################################################################
##########################                     ################################
##########################      MAIN CODE      ################################
##########################                     ################################
###############################################################################

def main(inifile,eventTime,ID,outDir,pathToModel='./ML/trained_model/',uniqueID=True,NSDF=False,runML=False,HDF5=False,PostgreSQL=False,verbose=False):

    ###########################################################################
    #                                   Parse Ini File                        #
    ###########################################################################

    # ---- Create configuration-file-parser object and read parameters file.
    cp = ConfigParser.ConfigParser()
    cp.read(inifile)

    # ---- Read needed variables from [parameters] and [channels] sections.
    alwaysPlotFlag = cp.getint('parameters', 'alwaysPlotFlag')
    sampleFrequency = cp.getint('parameters', 'sampleFrequency')
    blockTime = cp.getint('parameters', 'blockTime')
    searchFrequencyRange = json.loads(cp.get('parameters', 'searchFrequencyRange'))
    searchQRange = json.loads(cp.get('parameters', 'searchQRange'))
    searchMaximumEnergyLoss = cp.getfloat('parameters', 'searchMaximumEnergyLoss')
    searchWindowDuration = cp.getfloat('parameters', 'searchWindowDuration')
    whiteNoiseFalseRate = cp.getfloat('parameters', 'whiteNoiseFalseRate')
    plotTimeRanges = json.loads(cp.get('parameters', 'plotTimeRanges'))
    plotFrequencyRange = json.loads(cp.get('parameters', 'plotFrequencyRange'))
    plotNormalizedERange = json.loads(cp.get('parameters', 'plotNormalizedERange'))
    frameCacheFile = cp.get('channels', 'frameCacheFile')
    frameType = cp.get('channels', 'frameType')
    channelName = cp.get('channels', 'channelName')
    detectorName = channelName.split(':')[0]
    det = detectorName.split('1')[0]

    ###########################################################################
    #                           create output directory                       #
    ###########################################################################

    # if outputDirectory not specified, make one based on center time
    if outDir is None:
        outDirtmp = './scans'
    else:
        outDirtmp = outDir + '/' + ID + '/' + ID
    outDirtmp += '/'

    # report status
    if not os.path.isdir(outDirtmp):
        if verbose:
            print 'creating event directory'
        os.makedirs(outDirtmp)
    if verbose:
        print 'outputDirectory:  {0}'.format(outDirtmp)

    ########################################################################
    #     Determine if this is a normal omega scan or a Gravityspy         #
    #    omega scan with unique ID. If Gravity spy then additional         #
    #    files and what not must be generated                              #
    ########################################################################

    if uniqueID:
        IDstring = ID
    else:
        IDstring = "{0:.2f}".format(eventTime)

    ###########################################################################
    #               Process Channel Data                                      #
    ###########################################################################

    # find closest sample time to event time
    centerTime = np.floor(eventTime) + \
               np.round((eventTime - np.floor(eventTime)) * \
                     sampleFrequency) / sampleFrequency

    # determine segment start and stop times
    startTime = round(centerTime - blockTime / 2)
    stopTime = startTime + blockTime

    # Read in the data
    if NSDF:
        data = TimeSeries.fetch(channelName, startTime, stopTime)
    else:
        connection = datafind.GWDataFindHTTPConnection()
        cache = connection.find_frame_urls(det, frameType, startTime, stopTime, urltype='file')
        data = TimeSeries.read(cache, channelName, format='gwf', start=startTime, end=stopTime)

    # resample data
    if data.sample_rate.decompose().value != sampleFrequency:
        data = data.resample(sampleFrequency)

    qScan = data.q_transform(qrange=(4, 64), frange=(10, 2048),
                             gps=centerTime, search=0.5, tres=0.001,
                             fres=0.5, outseg=None, whiten=True)

    # Crop out the different time duration windows
    specsgrams = []
    for iTimeWindow in plotTimeRanges:
        specsgrams.append(qScan.crop(centerTime-iTimeWindow/2, centerTime+iTimeWindow/2))

    # Set some plotting params
    myfontsize = 15
    mylabelfontsize = 20
    myColor = 'k'
    if detectorName == 'H1':
	title = "Hanford"
    elif detectorName == 'L1':
	title = "Livingston"
    else:
	title = "VIRGO"

    if 1161907217 < startTime < 1164499217:
	title = title + ' - ER10'
    elif startTime > 1164499217:
	title = title + ' - O2a'
    elif 1126400000 < startTime < 1137250000:
	title = title + ' - O1'
    else:
	ValueError("Time outside science or engineering run\
		   or more likely code not updated to reflect\
		   new science run")

    # Create one image containing all spectogram grams
    superFig = Plot(figsize=(27,6))
    for i, spec in enumerate(specsgrams):
        superFig.add_spectrogram(spec,newax=True)
        plot = spec.plot(figsize=[8, 6])
        ax = plot.gca()
        ax.set_position([0.125, 0.1, 0.775, 0.8])
        ax.set_epoch(centerTime)
        ax.set_yscale('log', basey=2)
        ax.set_xlabel('Time (s)', fontsize=mylabelfontsize, color=myColor)
        ax.set_ylabel('Frequency (Hz)', fontsize=mylabelfontsize, color=myColor)
        ax.set_title(title, fontsize=mylabelfontsize, color=myColor)
        ax.set_ylim(10, 2048)
        ax.set_xlim(centerTime-plotTimeRanges[i]/2, centerTime+plotTimeRanges[i]/2)
        for axis in [ax.xaxis, ax.yaxis]:
            axis.set_major_formatter(ScalarFormatter())
        plt.tick_params(axis='both', which='major', labelsize=myfontsize)
        ax.grid(False)
        plot.add_colorbar(cmap='viridis', label='Normalized energy',
                          clim=plotNormalizedERange, pad="3%", width="5%")
        dur = float(plotTimeRanges[i])
        plot.save(outDirtmp + detectorName + '_' + IDstring + '_spectrogram_' + str(dur) +'.png')

    testFig = Plot(figsize=(27,6))
    for i, iAx in enumerate(superFig.axes):
        ax = testFig.add_subplot(1, len(specsgrams), i+1)
        iAx.set_position(ax.get_position())
        iAx.set_epoch(centerTime)
        iAx.set_yscale('log', basey=2)
        iAx.set_xlabel('Time (s)', fontsize=mylabelfontsize, color=myColor)
        iAx.set_ylabel('Frequency (Hz)', fontsize=mylabelfontsize, color=myColor)
        iAx.set_ylim(plotFrequencyRange[0],plotFrequencyRange[1])
        iAx.set_xlim(centerTime-plotTimeRanges[i]/2, centerTime+plotTimeRanges[i]/2)
        for axis in [iAx.xaxis, iAx.yaxis]:
            axis.set_major_formatter(ScalarFormatter())

    superFig.add_colorbar(cmap='viridis', label='Normalized energy',
                          clim=plotNormalizedERange, pad="3%", width="5%")
    superFig.suptitle(title,fontsize=mylabelfontsize, color=myColor,x=0.51)
    superFig.save(outDirtmp + IDstring + '.png',bbox_inches='tight')

    if runML:
        # Create directory called "Classified" were images that were successfully classified go.
        workFlow = 'Classified'
        finalPath = outDir + '/' + workFlow

        if not os.path.isdir(finalPath):
            os.makedirs(finalPath)

        # We must determine the columns that will be saved for this image.
        # First and foremost we want to determine the possible classes the image could be
        # get all the info about the workflows
        workflowDictSubjectSets = Structure.main('1104','O2')

        # Must determine classes from dict
        classes = sorted(workflowDictSubjectSets[2117].keys())

        # Add on columns that are Gravity Spy specific
        classes.extend(["uniqueID","Label","workflow","subjectset","Filename1","Filename2","Filename3","Filename4","UploadFlag"])

        # First label the image
        lastPath = (outDirtmp).split('/')[-2]
        make_pickle.main(outDirtmp.replace(lastPath, "", 1),
                         outDirtmp + '/pickleddata/', 1, verbose)
        scores, MLlabel = label_glitches.main(outDirtmp + '/pickleddata/',
                                              '{0}'.format(pathToModel),
                                              outDirtmp + '/labeled/',
                                              verbose)
        # Determine label
        Label = classes[MLlabel]

        # determine confidence values from ML
        scores = scores.tolist()
        scores = scores[1::]
        scores = [float(iScore) for iScore in scores]
        # Append uniqueID to list so when we update sql we will know which entry to update
        scores.append(ID)
        # Append label
        scores.append(Label)

        # Determine subject set and workflow this should go to.
        for iWorkflow in workflowDictSubjectSets.keys(): 
            if Label in workflowDictSubjectSets[iWorkflow].keys():
                 if workflowDictSubjectSets[iWorkflow][Label][2][1] <= scores[MLlabel]  <= workflowDictSubjectSets[iWorkflow][Label][2][0]:
                     workflowNum = workflowDictSubjectSets[iWorkflow][Label][0]
                     subjectSetNum = workflowDictSubjectSets[iWorkflow][Label][1]
                     break

        subject1 = '{0}/{1}_{2}_spectrogram_0.5.png'.format(finalPath,detectorName,ID)
        subject2 = '{0}/{1}_{2}_spectrogram_1.0.png'.format(finalPath,detectorName,ID)
        subject3 = '{0}/{1}_{2}_spectrogram_2.0.png'.format(finalPath,detectorName,ID)
        subject4 = '{0}/{1}_{2}_spectrogram_4.0.png'.format(finalPath,detectorName,ID)

        scores.append(workflowNum)
        scores.append(subjectSetNum)
        scores.append(subject1)
        scores.append(subject2)
        scores.append(subject3)
        scores.append(subject4)
        # Upload flag (defaults to 0 because image gets uploaded later)
        scores.append(0)

        scoresTable = pd.DataFrame([scores],columns=classes)

        if PostgreSQL:
            engine = create_engine(
                                   'postgresql://{0}:{1}'\
                                   .format(os.environ['QUEST_SQL_USER'],os.environ['QUEST_SQL_PASSWORD'])\
                                   + '@gravityspy.ciera.northwestern.edu:5432/gravityspy')
            columnDict = scoresTable.to_dict(orient='records')[0]
            SQLCommand = 'UPDATE glitches SET '
            for Column in columnDict:
                if isinstance(columnDict[Column],basestring):
                    SQLCommand = SQLCommand + '''\"{0}\" = \'{1}\', '''.format(Column,columnDict[Column])
                else:
                    SQLCommand = SQLCommand + '''\"{0}\" = {1}, '''.format(Column,columnDict[Column])
            SQLCommand = SQLCommand[:-2] + ' WHERE \"uniqueID\" = \'' + scoresTable.uniqueID.iloc[0] + "'"
            engine.execute(SQLCommand)
        elif HDF5:
            scoresTable.to_hdf('{0}/ML_GSpy_{1}.h5'.format(outDir,ID),'gspy_ML_classification')

        system_call = "mv {0}*.png {1}".format(outDirtmp,finalPath)
        os.system(system_call)
        shutil.rmtree(outDirtmp.replace(lastPath, "", 1))

if __name__ == '__main__':
    opts = parse_commandline()
    main(opts.inifile,opts.eventTime,opts.ID,opts.outDir,opts.pathToModel,opts.uniqueID,opts.NSDF,opts.runML,opts.HDF5,opts.PostgreSQL,opts.verbose)
