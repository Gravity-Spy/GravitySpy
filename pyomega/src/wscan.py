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

import ML.make_pickle_for_linux as make_pickle
import ML.labelling_test_glitches as label_glitches

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
    parser.add_option("--eventTime", type=float,help="Trigger time of the glitch")
    parser.add_option("--uniqueID", action="store_true", default=False,help="Is this image being generated for the GravitySpy project, is so we will create a uniqueID strong to use for labeling images instead of GPS time")
    parser.add_option("--ID", default='',help="Already supplying an ID? If not then ignore this flag. Only to be used in conjunction with --uniqueID")
    parser.add_option("--outDir", help="Outdir of omega scan and omega scan webpage (i.e. your html directory)")
    parser.add_option("--NSDF", action="store_true", default=False,help="No framecache file available want to use NSDF server")
    parser.add_option("--condor", action="store_true", default=False,help="Want to run as condor job?")
    parser.add_option("--pathToModel",default='./ML/trained_model/', help="Path to trained model")
    parser.add_option("--plot-whitened-timeseries", action="store_true", default=False,help="Plot whitened timeseries")
    parser.add_option("--plot-highpassfiltered-timeseries", action="store_true", default=False,help="Plot high pass filtered timeseries")
    parser.add_option("--plot-raw-timeseries", action="store_true", default=False,help="Plot raw timeseries")
    parser.add_option("--plot-eventgram", action="store_true", default=False,help="Plot eventgram")
    parser.add_option("--runML", action="store_true", default=False,help="Run the ML classifer on the omega scans")
    parser.add_option("--verbose", action="store_true", default=False,help="Run in Verbose Mode")
    parser.add_option("--HDF5", action="store_true", default=False,help="Store triggers in local HDF5 table format")
    parser.add_option("--PostgreSQL", action="store_true", default=False,help="Store triggers in local PostgreSQL format")
    opts, args = parser.parse_args()


    return opts

###############################################################################
##########################                     ################################
##########################      MAIN CODE      ################################
##########################                     ################################
###############################################################################

def main():
    # Parse commandline arguments

    import time
    start = time.time()
    opts = parse_commandline()

    ###########################################################################
    #                                   Parse Ini File                        #
    ###########################################################################

    # ---- Create configuration-file-parser object and read parameters file.
    cp = ConfigParser.ConfigParser()
    cp.read(opts.inifile)

    # ---- Read needed variables from [parameters] and [channels] sections.
    alwaysPlotFlag           = cp.getint('parameters','alwaysPlotFlag')
    sampleFrequency          = cp.getint('parameters','sampleFrequency')
    blockTime                = cp.getint('parameters','blockTime')
    searchFrequencyRange     = json.loads(cp.get('parameters','searchFrequencyRange'))
    searchQRange             = json.loads( cp.get('parameters','searchQRange'))
    searchMaximumEnergyLoss  = cp.getfloat('parameters','searchMaximumEnergyLoss')
    searchWindowDuration     = cp.getfloat('parameters','searchWindowDuration')
    whiteNoiseFalseRate      = cp.getfloat('parameters','whiteNoiseFalseRate')
    plotTimeRanges           = json.loads(cp.get('parameters','plotTimeRanges'))
    plotFrequencyRange       = json.loads(cp.get('parameters','plotFrequencyRange'))
    plotNormalizedERange     = json.loads(cp.get('parameters','plotNormalizedERange'))
    frameCacheFile           = cp.get('channels','frameCacheFile')
    frameType                = cp.get('channels','frameType')
    channelName              = cp.get('channels','channelName')
    detectorName             = channelName.split(':')[0]
    det                      = detectorName.split('1')[0]
    ###########################################################################
    #                            hard coded parameters                        #
    ###########################################################################

    '''describe these...'''
    # search parameters
    transientFactor = 2
    outlierFactor = 2.0
    durationInflation = 1.0
    bandwidthInflation = 1.0

    # display parameters
    plotHorizontalResolution = 512
    plotDurationInflation = 0.5
    plotBandwidthInflation = 0.5

    # limits on number of significant tiles
    maximumSignificants = 1e4
    maximumMosaics = 1e4

    ###########################################################################
    #                           create output directory                       #
    ###########################################################################

    # if outputDirectory not specified, make one based on center time
    if opts.outDir is None:
        outDir = './scans'
    else:
        outDir = opts.outDir + '/' + opts.ID + '/' + opts.ID
    outDir += '/'

    # report status
    if not os.path.isdir(outDir):
        if opts.verbose:
            print('creating event directory')
        os.makedirs(outDir)
    if opts.verbose:
        print('outputDirectory:  {0}'.format(outDir))

    ########################################################################
    #     Determine if this is a normal omega scan or a Gravityspy         #
    #    omega scan with unique ID. If Gravity spy then additional         #
    #    files and what not must be generated                              #
    ########################################################################

    if opts.uniqueID:
        IDstring = opts.ID
    else:
        IDstring = "{0:.2f}".format(opts.eventTime)

    ###########################################################################
    #               Process Channel Data                                      #
    ###########################################################################

    # find closest sample time to event time
    centerTime = np.floor(opts.eventTime) + \
               np.round((opts.eventTime - np.floor(opts.eventTime)) * \
                     sampleFrequency) / sampleFrequency

    # determine segment start and stop times
    startTime = round(centerTime - blockTime / 2)
    stopTime = startTime + blockTime

    # Read in the data
    if opts.NSDF:
        data = TimeSeries.fetch(channelName,startTime,stopTime)
    else:
        connection = datafind.GWDataFindHTTPConnection()
        cache = connection.find_frame_urls(det, frameType, startTime, stopTime, urltype='file')
        data = TimeSeries.read(cache,channelName, format='gwf',start=startTime,end=stopTime)

    # resample data
    if data.sample_rate.decompose().value != sampleFrequency:
        data = data.resample(sampleFrequency)

    qScan = data.q_transform(qrange=(4, 64), frange=(10, 2048), gps=centerTime, search=0.5, tres=0.001, fres=0.5, outseg=None, whiten=True)

    for iTimeWindow in plotTimeRanges:
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
            ValueError("Time outside science or engineering run or more likely code not updated to reflect new science run")

        plot = qScan.crop(centerTime-iTimeWindow/2, centerTime+iTimeWindow/2).plot(figsize=[8, 6])
        ax = plot.gca()
        ax.set_position([0.125, 0.1, 0.775, 0.8])
        ax.set_epoch(centerTime)
        ax.set_yscale('log',basey=2)
        ax.set_xlabel('Time (s)',fontsize=mylabelfontsize, color=myColor)
        ax.set_ylabel('Frequency (Hz)',fontsize=mylabelfontsize, color=myColor)
        ax.set_title(title,fontsize=mylabelfontsize, color=myColor)
        ax.set_ylim(10, 2048)
        ax.set_xlim(centerTime-iTimeWindow/2, centerTime+iTimeWindow/2)
        for axis in [ax.xaxis, ax.yaxis]:
            axis.set_major_formatter(ScalarFormatter())
        plt.tick_params(axis='both', which='major', labelsize=myfontsize)
        ax.grid(False)
        plot.add_colorbar(cmap='viridis', label='Normalized energy',clim=plotNormalizedERange,pad="3%",width="5%")
        dur = float(iTimeWindow)
        plot.save(outDir + detectorName + '_' + IDstring + '_spectrogram_' + str(dur) +'.png')

    if opts.runML:
        B1 = 1610
        B2 = 1934
        B3 = 1935
        A  = 2360
        M  = 2117
        if detectorName == 'H1':
            workflow_subject_set_dict_app = {
                "1400Ripples":((A,8190,[1,0.5]),(M,8192,[0.5,0])),
                "1080Lines":((A,8193,[1,0.5]),(M,8194,[0.5,0])),
                "Air_Compressor":((A,6714,[1,0.6]),(M,6715,[0.6,0])),
                "Blip":((B1,6717,[1,.998]),(A,6718,[.998,.85]),(M,6719,[.85,0])),
                "Chirp":((B3,6721,[1,.7]),(A,6722,[.7,.5]),(M,6723,[.50,0])),
                "Extremely_Loud":((A,6725,[1,.815]),(M,6726,[.815,0])),
                "Helix":((A,6728,[1,.50]),(M,6729,[.50,0])),
                "Koi_Fish":((B2,6731,[1,.98]),(A,6732,[.98,.621]),(M,6733,[.621,0])),
                "Light_Modulation":((A,6752,[1,0.9]),(M,6753,[0.9,0])),
                "Low_Frequency_Burst":((B3,6755,[1,.99995]),(A,7068,[.99995,.93]),(M,7070,[.93,0])),
                "Low_Frequency_Lines":((A,6759,[1,.65]),(M,6760,[.65,0])),
                "No_Glitch":((B3,6762,[1,.9901]),(A,6763,[.9901,.85]),(M,6764,[.85,0])),
                "None_of_the_Above":((A,6766,[1,.50]),(M,6767,[.50,0])),
                "Paired_Doves":((A,6769,[1,.5]),(M,6770,[.50,0])),
                "Power_Line":((B2,6772,[1,.998]),(A,6773,[.998,.86]),(M,6774,[.86,0])),
                "Repeating_Blips":((A,6776,[1,.686]),(M,6777,[.686,0])),
                "Scattered_Light":((B3,6779,[1,.99965]),(A,6780,[.99965,.96]),(M,6781,[.96,0])),
                "Scratchy":((A,6783,[1,.913]),(M,6784,[.913,0])),
                "Tomte":((A,6786,[1,.7]),(M,6787,[.7,0])),
                "Violin_Mode":((B2,6902,[1,.99]),(A,6789,[.99,.5]),(M,6790,[.50,0])),
                "Wandering_Line":((A,6792,[1,.97]),(M,6793,[.97,0])),
                "Whistle":((B1,6795,[1,.99]),(A,6796,[.99,.6]),(M,6797,[.6,0])),
              }
        elif detectorName == 'L1':
            workflow_subject_set_dict_app = {
                "1400Ripples":((A,8190,[1,0.5]),(M,8192,[0.5,0])),
                "1080Lines":((A,8193,[1,0.5]),(M,8194,[0.5,0])),
                "Air_Compressor":((A,6714,[1,0.6]),(M,6715,[0.6,0])),
                "Blip":((B1,6717,[1,.998]),(A,6718,[.998,.85]),(M,6719,[.85,0])),
                "Chirp":((B3,6721,[1,.7]),(A,6722,[.7,.5]),(M,6723,[.50,0])),
                "Extremely_Loud":((A,6725,[1,.815]),(M,6726,[.815,0])),
                "Helix":((A,6728,[1,.50]),(M,6729,[.50,0])),
                "Koi_Fish":((B2,6731,[1,.98]),(A,6732,[.98,.621]),(M,6733,[.621,0])),
                "Light_Modulation":((A,6752,[1,0.9]),(M,6753,[0.9,0])),
                "Low_Frequency_Burst":((B3,6755,[1,.99995]),(A,7068,[.99995,.93]),(M,7070,[.93,0])),
                "Low_Frequency_Lines":((A,6759,[1,.65]),(M,6760,[.65,0])),
                "No_Glitch":((B3,6762,[1,.9901]),(A,6763,[.9901,.85]),(M,6764,[.85,0])),
                "None_of_the_Above":((A,6766,[1,.50]),(M,6767,[.50,0])),
                "Paired_Doves":((A,6769,[1,.5]),(M,6770,[.50,0])),
                "Power_Line":((B2,6772,[1,.998]),(A,6773,[.998,.86]),(M,6774,[.86,0])),
                "Repeating_Blips":((A,6776,[1,.686]),(M,6777,[.686,0])),
                "Scattered_Light":((B3,6779,[1,.99965]),(A,6780,[.99965,.96]),(M,6781,[.96,0])),
                "Scratchy":((A,6783,[1,.913]),(M,6784,[.913,0])),
                "Tomte":((A,6786,[1,.7]),(M,6787,[.7,0])),
                "Violin_Mode":((B2,6902,[1,.99]),(A,6789,[.99,.5]),(M,6790,[.50,0])),
                "Wandering_Line":((A,6792,[1,.97]),(M,6793,[.97,0])),
                "Whistle":((B1,6795,[1,.99]),(A,6796,[.99,.6]),(M,6797,[.6,0])),
              }

        lastPath = (outDir).split('/')[-2]
        make_pickle.main(outDir.replace(lastPath,"",1),outDir + '/pickleddata/',1,opts.verbose)

        scores,MLlabel = label_glitches.main(outDir + '/pickleddata/','{0}'.format(opts.pathToModel),outDir + '/labeled/',opts.verbose)

        scores = scores.tolist()
        scores = scores[1::]
        scores = [float(iScore) for iScore in scores]
        scores.append(opts.ID)
        classes = ["1080Lines","1400Ripples","Air_Compressor","Blip","Chirp","Extremely_Loud","Helix","Koi_Fish","Light_Modulation","Low_Frequency_Burst","Low_Frequency_Lines","No_Glitch","None_of_the_Above","Paired_Doves","Power_Line","Repeating_Blips","Scattered_Light","Scratchy","Tomte","Violin_Mode","Wandering_Line","Whistle","uniqueID","Label","workflow","subjectset","Filename1","Filename2","Filename3","Filename4","UploadFlag"]
        scores.append(classes[MLlabel])

        workFlow = 'Classified'
        finalPath = opts.outDir + '/' + workFlow

        if not os.path.isdir(finalPath):
            os.makedirs(finalPath)

        for iWorkflow in range(len(workflow_subject_set_dict_app[classes[MLlabel]])):
            if max(workflow_subject_set_dict_app[classes[MLlabel]][iWorkflow][2]) >=scores[MLlabel] >=min(workflow_subject_set_dict_app[classes[MLlabel]][iWorkflow][2]):
                 workflowNum = workflow_subject_set_dict_app[classes[MLlabel]][iWorkflow][0]
                 subjectSetNum = workflow_subject_set_dict_app[classes[MLlabel]][iWorkflow][1]
                 break

        subject1 = '{0}/{1}_{2}_spectrogram_0.5.png'.format(finalPath,detectorName,opts.ID)
        subject2 = '{0}/{1}_{2}_spectrogram_1.0.png'.format(finalPath,detectorName,opts.ID)
        subject3 = '{0}/{1}_{2}_spectrogram_2.0.png'.format(finalPath,detectorName,opts.ID)
        subject4 = '{0}/{1}_{2}_spectrogram_4.0.png'.format(finalPath,detectorName,opts.ID)

        scores.append(workflowNum)
        scores.append(subjectSetNum)
        scores.append(subject1)
        scores.append(subject2)
        scores.append(subject3)
        scores.append(subject4)
        scores.append(0)

        scoresTable = pd.DataFrame([scores],columns=classes)
        if opts.PostgreSQL:
            engine = create_engine('postgresql://scoughlin@localhost:5432/gravityspy')
            columnDict = scoresTable.to_dict(orient='records')[0]
            SQLCommand = 'UPDATE glitches SET '
            for Column in columnDict:
                if isinstance(columnDict[Column],basestring):
                    SQLCommand = SQLCommand + '''\"{0}\" = \'{1}\', '''.format(Column,columnDict[Column])
                else:
                    SQLCommand = SQLCommand + '''\"{0}\" = {1}, '''.format(Column,columnDict[Column])
            SQLCommand = SQLCommand[:-2] + ' WHERE \"uniqueID\" = \'' + scoresTable.uniqueID.iloc[0] + "'"
            engine.execute(SQLCommand)
        elif opts.HDF5:
            scoresTable.to_hdf('{0}/ML_GSpy_{1}.h5'.format(opts.outDir,opts.ID),'gspy_ML_classification')

        system_call = "mv {0}*.png {1}".format(outDir,finalPath)
        os.system(system_call)
        shutil.rmtree(outDir.replace(lastPath,"",1))
        print('ML finished: {0}'.format(time.time()-start))

if __name__ == '__main__':
    main()
