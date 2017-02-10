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

from matplotlib import use
use('agg')
from matplotlib import (pyplot as plt, cm)
from matplotlib.ticker import ScalarFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

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
    opts, args = parser.parse_args()


    return opts

###############################################################################
##########################                            #########################
##########################   Func: write_subfile     #########################
##########################                            #########################
###############################################################################

# Write submission file for the condor job

def write_subfile():
    for d in ['logs', 'condor']:
        if not os.path.isdir(d):
            os.makedirs(d)
    with open('./condor/gravityspy.sub', 'w') as subfile:
        subfile.write('universe = vanilla\n')
        subfile.write('executable = {0}/wscan.py\n'.format(os.getcwd()))
        subfile.write('\n')
        subfile.write('arguments = "--inifile wini.ini --eventTime $(eventTime) --outDir {0} --uniqueID"\n'.format(opts.outDir))
        subfile.write('getEnv=True\n')
        subfile.write('\n')
        subfile.write('accounting_group_user = scott.coughlin\n')#.format(opts.username))
        subfile.write('accounting_group = ligo.dev.o1.detchar.ch_categorization.glitchzoo\n')
        subfile.write('\n')
        subfile.write('priority = 0\n')
        subfile.write('request_memory = 1000\n')
        subfile.write('\n')
        subfile.write('error = logs/gravityspy-$(jobNumber).err\n')
        subfile.write('output = logs/gravityspy-$(jobNumber).out\n')
        subfile.write('notification = never\n')
        subfile.write('queue 1')

###############################################################################
##########################                            #########################
##########################   Func: write_dagfile     #########################
##########################                            #########################
###############################################################################

# Write dag_file file for the condor job

def write_dagfile():
    with open('gravityspy.dag','a+') as dagfile:
        dagfile.write('JOB {0} ./condor/gravityspy.sub\n'.format(opts.eventTime))
        dagfile.write('RETRY {0} 3\n'.format(opts.eventTime))
        dagfile.write('VARS {0} jobNumber="{0}" eventTime="{0}"'.format(opts.eventTime))
        dagfile.write('\n\n')

###############################################################################
##########################                             ########################
##########################   Func: id_generator        ########################
##########################                             ########################
###############################################################################

def id_generator(size=10, chars=string.ascii_uppercase + string.digits +string.ascii_lowercase):
    return ''.join(random.SystemRandom().choice(chars) for _ in range(size))

###############################################################################
##########################                             ########################
##########################   Func: nextpow2            ########################
##########################                             ########################
###############################################################################

def nextpow2(i):
    n = 1
    while n < i: n *= 2
    return n

###############################################################################
##########################                             ########################
##########################      Func: wtile            ########################
##########################                             ########################
###############################################################################

def wtile(blockTime, searchQRange, searchFrequencyRange, sampleFrequency,
          searchMaximumEnergyLoss, highPassCutoff, lowPassCutoff,
          whiteningDuration, transientFactor):
    # extract minimum and maximum Q from Q range
    minimumQ = searchQRange[0]
    maximumQ = searchQRange[1]

    # extract minimum and maximum frequency from frequency range
    minimumFrequency = searchFrequencyRange[0]
    maximumFrequency = searchFrequencyRange[1]

    ###########################################################################
    #                          compute derived parameters                     #
    ###########################################################################

    # nyquist frequency (half the sampling rate)
    nyquistFrequency = sampleFrequency / 2

    # maximum mismatch between neighboring tiles
    '''why is this the formula for max mismatch'''
    mismatchStep = 2 * np.sqrt(searchMaximumEnergyLoss / 3)

    # maximum possible time resolution
    minimumTimeStep = 1 / sampleFrequency

    # maximum possible frequency resolution
    minimumFrequencyStep = 1 / blockTime

    # conversion factor from Q' to true Q
    '''why sqrt(11)'''
    qPrimeToQ = np.sqrt(11)

    # total number of samples in input data
    numberOfSamples = blockTime * sampleFrequency

    ############################################################################
    #                       determine parameter constraints                    #
    ############################################################################

    # minimum allowable Q' to prevent window aliasing at zero frequency
    minimumAllowableQPrime = 1.0

    # minimum allowable Q to avoid window aliasing at zero frequency
    minimumAllowableQ = minimumAllowableQPrime * qPrimeToQ

    # reasonable number of statistically independent tiles in a frequency row
    minimumAllowableIndependents = 50

    # maximum allowable mismatch parameter for reasonable performance
    maximumAllowableMismatch = 0.5

    ############################################################################
    #                             validate parameters                          #
    ############################################################################

    # check for valid time range
    if blockTime < 0:
        raise ValueError('negative time range')

    # check for valid Q range
    if minimumQ > maximumQ:
        raise ValueError('minimum Q is larger than maximum Q')

    # check for valid frequency range
    if minimumFrequency > maximumFrequency:
        raise ValueError('minimum frequency exceeds maximum frequency')

    # check for valid minimum Q
    if minimumQ < minimumAllowableQ:
        raise ValueError('minimum Q less than {0}'.format(minimumAllowableQ))

    # check for reasonable maximum mismatch parameter
    if searchMaximumEnergyLoss > maximumAllowableMismatch:
        raise ValueError('maximum mismatch exceeds {0}'.format(maximumAllowableMismatch))

    # check for integer power of two data length
    if not np.mod(np.log(blockTime * sampleFrequency) / np.log(2), 1) == 0:
        raise ValueError('data length is not an integer power of two')

    ############################################################################
    #                          determine Q planes                              #
    ############################################################################

    # cumulative mismatch across Q range
    qCumulativeMismatch = np.log(maximumQ / minimumQ) / np.sqrt(2)

    # number of Q planes
    numberOfPlanes = np.ceil(qCumulativeMismatch / mismatchStep)

    # ensure at least one plane
    if numberOfPlanes == 0:
        numberOfPlanes = 1

    # mismatch between neighboring planes
    qMismatchStep = qCumulativeMismatch / numberOfPlanes

    # index of Q planes
    qIndices = np.linspace(0.5,numberOfPlanes - 0.5,numberOfPlanes)

    # vector of Qs
    qs = minimumQ * np.exp(np.sqrt(2) * qIndices * qMismatchStep)


    ############################################################################
    #                         validate frequencies                             #
    ############################################################################

    # minimum allowable frequency to provide sufficient statistics
    minimumAllowableFrequency = minimumAllowableIndependents * max(qs) / \
                            (2 * np.pi * blockTime)

    # maximum allowable frequency to avoid window aliasing
    maximumAllowableFrequency = nyquistFrequency / (1 + qPrimeToQ / min(qs))

    # check for valid minimum frequency
    if (not minimumFrequency == 0) and \
        (minimumFrequency < minimumAllowableFrequency):
            raise ValueError('requested minimum frequency of {0} Hz  \
            less than minimum allowable frequency of {1} Hz').format(\
                                 minimumFrequency, minimumAllowableFrequency)

    # check for valid maximum frequency
    if (not np.isinf(maximumFrequency)) and \
        (maximumFrequency > maximumAllowableFrequency):
            raise ValueError('requested maximum frequency of {0} Hz  \
            less than maximum allowable frequency of {1} Hz').format(\
                                 maximumFrequency, maximumAllowableFrequency)

    tiling = {}
    tiling["generalparams"] = {}
    tiling["generalparams"]["duration"] = blockTime
    tiling["generalparams"]["minimumQ"] = minimumQ
    tiling["generalparams"]["maximumQ"] = maximumQ
    tiling["generalparams"]["minimumFrequency"] = minimumFrequency
    tiling["generalparams"]["maximumFrequency"] = maximumFrequency
    tiling["generalparams"]["sampleFrequency"] = sampleFrequency
    tiling["generalparams"]["searchMaximumEnergyLoss"] = searchMaximumEnergyLoss
    tiling["generalparams"]["qs"] = qs
    tiling["generalparams"]["numberOfPlanes"] = numberOfPlanes
    tiling["generalparams"]["numberOfTiles"] = 0
    tiling["generalparams"]["numberOfIndependents"] = 0
    tiling["generalparams"]["numberOfFlops"] = numberOfSamples * np.log(numberOfSamples)

    for plane in np.arange(0,numberOfPlanes):

        q = qs[int(plane)]

        #######################################################################
        #                 determine plane properties                          #
        #######################################################################

        # find Q' for the plane
        qPrime = q / qPrimeToQ

        # for large Q'
        if qPrime > 10:

            # use asymptotic value of planeNormalization
            planeNormalization = 1
        else:

            # polynomial coefficients for plane normalization factor
            coefficients = [\
                            np.log((qPrime + 1) / (qPrime - 1)), -2,\
                    - 4 * np.log((qPrime + 1) / (qPrime - 1)), 22 / 3,\
                      6 * np.log((qPrime + 1) / (qPrime - 1)), - 146 / 15,\
                    - 4 * np.log((qPrime + 1) / (qPrime - 1)), 186 / 35,\
                          np.log((qPrime + 1) / (qPrime - 1))]
            # Cast as an array
            coefficients = np.asarray(coefficients)

            # plane normalization factor
            planeNormalization = np.sqrt(256 / (315 * qPrime * \
                                     np.polyval(coefficients, qPrime)))

        ###################################################################
        #                   determine frequency rows                      #
        ###################################################################

        # plane specific minimum allowable frequency to provide sufficient statistics
        minimumAllowableFrequency = minimumAllowableIndependents * q / \
                              (2 * np.pi * tiling['generalparams']['duration'])

        # plane specific maximum allowable frequency to avoid window aliasing
        maximumAllowableFrequency = nyquistFrequency / (1 + qPrimeToQ / q)

        # use plane specific minimum allowable frequency if requested
        if tiling['generalparams']['minimumFrequency'] == 0:
            minimumFrequency = minimumAllowableFrequency

        # use plane specific maximum allowable frequency if requested
        if np.isinf(tiling['generalparams']['maximumFrequency']):
            maximumFrequency = maximumAllowableFrequency

        # cumulative mismatch across frequency range
        frequencyCumulativeMismatch = np.log(maximumFrequency / \
            minimumFrequency) * np.sqrt(2 + q**2) / 2

        # number of frequency rows
        numberOfRows = np.ceil(frequencyCumulativeMismatch / mismatchStep)

        # ensure at least one row
        if numberOfRows == 0:
            numberOfRows = 1

        # mismatch between neighboring frequency rows
        frequencyMismatchStep = frequencyCumulativeMismatch / numberOfRows

        # index of frequency rows
        frequencyIndices = np.linspace(0.5,numberOfRows - 0.5,numberOfRows)

        # vector of frequencies
        frequencies = minimumFrequency * np.exp((2 / np.sqrt(2 + q**2)) * \
                                       frequencyIndices * \
                                       frequencyMismatchStep)

        # ratio between successive frequencies
        frequencyRatio = np.exp((2 / np.sqrt(2 + q**2)) * frequencyMismatchStep)

        # project frequency vector onto realizable frequencies
        frequencies = np.round(frequencies / minimumFrequencyStep) * \
                minimumFrequencyStep

        #######################################################################
        #             create Q transform plane structure                      #
        #######################################################################

        planestr = 'plane' +str(plane)
        tiling[planestr] = {}
        tiling["generalparams"]["duration"] = blockTime
        # insert Q of plane into Q plane structure
        tiling[planestr]['q'] = q

        # insert minimum search frequency of plane into Q plane structure
        tiling[planestr]['minimumFrequency'] = minimumFrequency

        # insert maximum search frequency of plane into Q plane structure
        tiling[planestr]['maximumFrequency'] = maximumFrequency

        # insert plane normalization factor into Q plane structure
        tiling[planestr]['normalization'] = planeNormalization

        # insert frequency vector into Q plane structure
        tiling[planestr]['frequencies'] = frequencies

        # insert number of frequency rows into Q plane structure
        tiling[planestr]['numberOfRows'] = numberOfRows

        # initialize cell array of frequency rows into Q plane structure
        for i in np.arange(0,numberOfRows):
            rowstr = 'row' + str(i)
            tiling[planestr][rowstr] = {}

        # initialize number of tiles in plane counter
        tiling[planestr]['numberOfTiles'] = 0

        # initialize number of independent tiles in plane counter
        tiling[planestr]['numberOfIndependents'] = 0

        # initialize number of flops in plane counter
        tiling[planestr]['numberOfFlops'] = 0

        #######################################################################
        #               begin loop over frequency rows                        #
        #######################################################################

        for row in np.arange(0,numberOfRows):

            rowstr = 'row' + str(row)

            # extract frequency of row from frequency vector
            frequency = frequencies[int(row)]

            ####################################################################
            #              determine tile properties                           #
            ####################################################################

            # bandwidth for coincidence testing
            bandwidth = 2 * np.sqrt(np.pi) * frequency / q

            # duration for coincidence testing
            duration = 1 / bandwidth

            # frequency step for integration
            frequencyStep = frequency * (frequencyRatio - 1) / np.sqrt(frequencyRatio)

            ####################################################################
            #                 determine tile times                             #
            ####################################################################

            # cumulative mismatch across time range
            timeCumulativeMismatch = blockTime * 2 * np.pi * frequency / q

            # number of time tiles
            numberOfTiles = nextpow2(timeCumulativeMismatch / mismatchStep)

            # mismatch between neighboring time tiles
            timeMismatchStep = timeCumulativeMismatch / numberOfTiles

            # index of time tiles
            timeIndices = np.arange(0,numberOfTiles)

            # vector of times
            times = q * timeIndices * timeMismatchStep / (2 * np.pi * frequency)

            # time step for integration
            timeStep = q * timeMismatchStep / (2 * np.pi * frequency)

            # number of flops to compute row
            numberOfFlops = numberOfTiles * np.log(numberOfTiles)

            # number of independent tiles in row
            numberOfIndependents = 1 + timeCumulativeMismatch

            ####################################################################
            #                   generate window                                #
            ####################################################################

            # half length of window in samples
            halfWindowLength = np.floor((frequency / qPrime) / minimumFrequencyStep)

            # full length of window in samples
            windowLength = 2 * halfWindowLength + 1

            # sample index vector for window construction
            windowIndices = np.arange(-halfWindowLength,halfWindowLength+1)

            # frequency vector for window construction
            windowFrequencies = windowIndices * minimumFrequencyStep

            # dimensionless frequency vector for window construction
            windowArgument = windowFrequencies * qPrime / frequency

            # bi square window function
            '''what?'''
            window = (1 - windowArgument**2)**2
            # row normalization factor
            rowNormalization = np.sqrt((315 * qPrime) / (128 * frequency))

            # inverse fft normalization factor
            ifftNormalization = numberOfTiles / numberOfSamples

            # normalize window
            window = window * ifftNormalization * rowNormalization

            # number of zeros to append to windowed data
            zeroPadLength = numberOfTiles - windowLength

            # vector of data indices to inverse fourier transform
            dataIndices = np.round(1 + frequency / minimumFrequencyStep + windowIndices)
            ####################################################################
            #           create Q transform row structure                       #
            ####################################################################

            # insert frequency of row into frequency row structure
            tiling[planestr][rowstr]['frequency'] = frequency

            # insert duration into frequency row structure
            tiling[planestr][rowstr]['duration'] = duration

            # insert bandwidth into frequency row structure
            tiling[planestr][rowstr]['bandwidth'] = bandwidth

            # insert time step into frequency row structure
            tiling[planestr][rowstr]['timeStep'] = timeStep

            # insert frequency step into frequency row structure
            tiling[planestr][rowstr]['frequencyStep'] = frequencyStep

            # insert window vector into frequency row structure
            tiling[planestr][rowstr]['window'] = window

            # insert zero pad length into frequency row structure
            tiling[planestr][rowstr]['zeroPadLength'] = zeroPadLength

            # insert data index vector into frequency row structure
            tiling[planestr][rowstr]['dataIndices'] = dataIndices.astype('int')

            # insert number of time tiles into frequency row structure
            tiling[planestr][rowstr]['numberOfTiles'] = numberOfTiles

            # insert number of independent tiles in row into frequency row structure
            tiling[planestr][rowstr]['numberOfIndependents'] = numberOfIndependents

            # insert number of flops to compute row into frequency row structure
            tiling[planestr][rowstr]['numberOfFlops'] = numberOfFlops

            # increment number of tiles in plane counter
            tiling[planestr]['numberOfTiles'] = \
            tiling[planestr]['numberOfTiles'] + numberOfTiles

            # increment number of indepedent tiles in plane counter
            tiling[planestr]['numberOfIndependents'] = \
            tiling[planestr]['numberOfIndependents'] \
            + numberOfIndependents * \
                ((1 + frequencyCumulativeMismatch) / numberOfRows)

            # increment number of flops in plane counter
            tiling[planestr]['numberOfFlops'] = \
            tiling[planestr]['numberOfFlops'] + numberOfFlops

            ##############################################################
            #       end loop over frequency rows                         #
            ##############################################################

        # increment number of tiles in plane counter
        tiling['generalparams']['numberOfTiles'] = \
             tiling['generalparams']['numberOfTiles']  + tiling[planestr]['numberOfTiles']

        # increment number of indepedent tiles in plane counter
        tiling['generalparams']['numberOfIndependents'] = \
            tiling['generalparams']['numberOfIndependents'] +\
                tiling[planestr]['numberOfIndependents'] *\
                        ((1 + qCumulativeMismatch) / numberOfPlanes)

        # increment number of flops in plane counter
        tiling['generalparams']['numberOfFlops'] = \
                tiling['generalparams']['numberOfFlops'] + tiling[planestr]['numberOfFlops']

        ########################################################################
        #                        end loop over Q planes                        #
        ########################################################################

    ########################################################################
    #                 determine filter properties                          #
    ########################################################################

    # default high pass filter cutoff frequency
    defaultHighPassCutoff = float('Inf')
    # default low pass filter cutoff frequency
    defaultLowPassCutoff = 0
    # default whitening filter duration
    defaultWhiteningDuration = 0
    for plane in np.arange(0,numberOfPlanes):
        planestr = 'plane' + str(plane)
        defaultHighPassCutoff = min(defaultHighPassCutoff, \
                              tiling[planestr]['minimumFrequency'])
        defaultLowPassCutoff = max(defaultLowPassCutoff, \
                             tiling[planestr]['maximumFrequency'])
        defaultWhiteningDuration = max(defaultWhiteningDuration, \
                                 tiling[planestr]['q'] / \
                                 (2 * tiling[planestr]['minimumFrequency']))

        # put duration as an integer power of 2 of seconds
        defaultWhiteningDuration = 2**np.round(np.log2(defaultWhiteningDuration))
        # high pass filter cutoff frequency
        if not highPassCutoff:
            tiling['generalparams']['highPassCutoff'] = defaultHighPassCutoff
        else:
            tiling['generalparams']['highPassCutoff'] = highPassCutoff

        # low pass filter cutoff frequency
        if not lowPassCutoff:
            tiling['generalparams']['lowPassCutoff'] = defaultLowPassCutoff
        else:
            tiling['generalparams']['lowPassCutoff'] = lowPassCutoff

        # whitening filter duration
        if not whiteningDuration:
            tiling['generalparams']['whiteningDuration'] = defaultWhiteningDuration
        else:
            tiling['generalparams']['whiteningDuration'] = whiteningDuration

        # estimated duration of filter transients to supress
        tiling['generalparams']['transientDuration'] = transientFactor * tiling['generalparams']['whiteningDuration']

        # test for insufficient data
        if (2 * tiling['generalparams']['transientDuration']) >= \
            tiling['generalparams']['duration']:
                error('duration of filter transients equals or exceeds data duration')

    return tiling

###############################################################################
#####################                                  ########################
#####################           highpassfilter         ########################
#####################                                  ########################
###############################################################################

def highpassfilt(data,tiling):

    # determine required data lengths
    dataLength = tiling['generalparams']['sampleFrequency'] * tiling['generalparams']['duration']
    halfDataLength = dataLength / 2 + 1

    # nyquist frequency
    nyquistFrequency = tiling["generalparams"]["sampleFrequency"] / 2

    # linear predictor error filter order
    lpefOrder = int(np.ceil(tiling["generalparams"]["sampleFrequency"] * \
    tiling['generalparams']['whiteningDuration']))

    if tiling['generalparams']['highPassCutoff'] > 0:
        # high pass filter order
        filterOrder = 12

        # design high pass filter
        [hpfZeros, hpfPoles, hpfGain] = \
        signal.butter(filterOrder, tiling['generalparams']['highPassCutoff'] \
                            / nyquistFrequency, btype='high',output='zpk')

        data = data.zpk(hpfZeros,hpfPoles,hpfGain)

        # End if statement

    # supress high pass filter transients
    data.value[:lpefOrder] = np.zeros(lpefOrder)
    data.value[(dataLength - lpefOrder) :dataLength] = np.zeros(lpefOrder)

    return data, lpefOrder

###############################################################################
#####################                                  ########################
#####################           wtransform             ########################
#####################                                  ########################
###############################################################################

def wtransform(data, tiling, outlierFactor,
               analysisMode, channelNames, coefficients, coordinate):

    # determine number of channels
    numberOfChannels = 1

    # determine required data lengths
    dataLength = tiling['generalparams']['sampleFrequency'] * tiling['generalparams']['duration']
    halfDataLength = dataLength / 2 + 1

    # validate data length and force row vectors
    if len(data) != halfDataLength:
        raise ValueError('data wrong length')

    # determine number of sites
    numberOfSites = 1

    if len(coordinate) != 2:
        raise ValueError('wrong number of coordinates')

    #######################################################################
    #                   Define some variables                             #
    #######################################################################

    outputChannelNames = channelNames
    numberOfPlanes = tiling['generalparams']['numberOfPlanes']

    #######################################################################
    #             initialize Q transform structures                       #
    #######################################################################

    # create empty cell array of Q transform structures
    transforms = {}

    # begin loop over channels
    for channel in np.arange(0,numberOfChannels):
        channelstr = 'channel' + str(channel)
        transforms[channelstr] = {}

    # End loop over channels

    # begin loop over Q planes
    for plane in np.arange(0,numberOfPlanes):

        planestr = 'plane' + str(plane)
        transforms[channelstr][planestr] = {}

        # Begin loop over frequency rows
        for row in np.arange(0,tiling[planestr]['numberOfRows']):
            # create empty cell array of frequency row structures
            rowstr = 'row' +str(row)
            transforms[channelstr][planestr][rowstr] = {}
        # End loop over frequency rows

    # End loop over Q planes

    # Initialize energies
    energies = {}
    # Initialize windowedData
    windowedData = {}
    # Initialize Tile Coefficients
    tileCoefficients = {}
    # Initialize validIndices
    validIndices = {}
    # Initialize Quatiles strcuture arrays
    lowerQuartile = {}
    upperQuartile = {}
    interQuartileRange = {}
    # Initialize outlier threshold
    outlierThreshold = {}
    # Initialize mean energy
    meanEnergy = {}
    # Initialize Normalized energy
    normalizedEnergies = {}


    ############################################################################
    #                       begin loop over Q planes                           #
    ############################################################################

    # begin loop over Q planes
    for plane in np.arange(0,numberOfPlanes):
        planestr = 'plane' + str(plane)

        ########################################################################
        #                begin loop over frequency rows                        #
        ########################################################################

        # begin loop over frequency rows
        for row in np.arange(0,tiling[planestr]['numberOfRows']):

            rowstr = 'row' +str(row)

            ####################################################################
            #          extract and window frequency domain data                #
            ####################################################################

            # number of zeros to pad at negative frequencies
            leftZeroPadLength = int((tiling[planestr][rowstr]['zeroPadLength'] - 1) / 2)

            # number of zeros to pad at positive frequencies
            rightZeroPadLength = int((tiling[planestr][rowstr]['zeroPadLength'] + 1) / 2)

            # begin loop over intermediate channels
            for channel in np.arange(0,numberOfChannels):
                channelstr = 'channel' + str(channel)
                windowedData[channelstr] ={}
                # extract and window in-band data
                windowedData[channelstr] = tiling[planestr][rowstr]['window'] * \
                data[tiling[planestr][rowstr]['dataIndices']]

                # zero pad windowed data
                windowedData[channelstr] = np.pad(windowedData[channelstr],\
                                    [leftZeroPadLength,rightZeroPadLength],'constant',constant_values=(0,0))
                # reorder indices for fast fourier transform
                lastIndex = len(windowedData[channelstr])
                order1 = np.arange(lastIndex / 2,lastIndex)
                order2 = np.arange(0,lastIndex/2)
                order1 = order1.astype('int')
                order2 = order2.astype('int')
                order  = np.concatenate((order1,order2))
                windowedData[channelstr] = windowedData[channelstr][order]

            # end loop over intermediate channels

            ################################################################
            #        inverse fourier transform windowed data               #
            ################################################################

            # begin loop over intermediate channels
            for channel in np.arange(0,numberOfChannels):
                channelstr = 'channel' + str(channel)

                # Initialize tileCoefficients
                tileCoefficients[channelstr] = {}
                # complex valued tile coefficients
                tileCoefficients[channelstr] = np.fft.ifft(windowedData[channelstr])
            # End loop over intermediate channels

            ##################################################################
            #              energies directly or indirectly                   #
            ##################################################################

            # compute energies directly from intermediate data
            for channel in np.arange(0,numberOfChannels):
                channelstr = 'channel' + str(channel)
                energies[channelstr] = \
                    tileCoefficients[channelstr].real**2 + \
                    tileCoefficients[channelstr].imag**2

            # End loop over channels

            ####################################################################
            #        exclude outliers and filter transients from statistics    #
            ####################################################################

            times = np.arange(0,tiling[planestr][rowstr]['numberOfTiles']) * \
                tiling[planestr][rowstr]['timeStep']

            # begin loop over channels
            for channel in np.arange(0,numberOfChannels):
                channelstr = 'channel' + str(channel)

                  # indices of non-transient tiles
                validIndices[channelstr] = \
                np.logical_and(times > \
                    tiling['generalparams']['transientDuration'], \
                        times < \
                    tiling['generalparams']['duration']- tiling['generalparams']['transientDuration'])

                # identify lower and upper quartile energies
                sortedEnergies = \
                    np.sort(energies[channelstr][validIndices[channelstr]])
                lowerQuartile[channelstr] = \
                sortedEnergies[np.round(0.25 * len(energies[channelstr][validIndices[channelstr]])).astype('int')]
                upperQuartile[channelstr] = \
                sortedEnergies[np.round(0.75 * len(energies[channelstr][validIndices[channelstr]])).astype('int')]

                # determine inter quartile range
                interQuartileRange[channelstr] = upperQuartile[channelstr] - \
                                          lowerQuartile[channelstr]

                # energy threshold of outliers
                outlierThreshold[channelstr] = upperQuartile[channelstr] + \
                    outlierFactor * interQuartileRange[channelstr]

                # indices of non-outlier and non-transient tiles
                validIndices[channelstr] = \
                    np.logical_and(np.logical_and(energies[channelstr] < \
                          outlierThreshold[channelstr],\
                          times > \
                          tiling['generalparams']['transientDuration']), \
                          times < \
                          tiling['generalparams']['duration']- tiling['generalparams']['transientDuration'])

            # end loop over channels

            # for reasonable outlier factors,
            if outlierFactor < 100:

                # mean energy correction factor for outlier rejection bias
                meanCorrectionFactor = (4 * 3**outlierFactor - 1) / \
                             ((4 * 3**outlierFactor - 1) - \
                             (outlierFactor * np.log(3) + np.log(4)))

            # otherwise, for large outlier factors
            else:

                # mean energy correction factor for outlier rejection bias
                meanCorrectionFactor = 1

            # End if statement

            ####################################################################
            #       determine tile statistics and normalized energies          #
            ####################################################################

            # begin loop over channels
            for channel in np.arange(0,numberOfChannels):
                channelstr = 'channel' + str(channel)

                # mean of valid tile energies
                meanEnergy[channelstr] = \
                  np.mean(energies[channelstr][validIndices[channelstr]])

                # correct for bias due to outlier rejection
                meanEnergy[channelstr] = meanEnergy[channelstr] * \
                  meanCorrectionFactor

                # normalized tile energies
                normalizedEnergies[channelstr] = energies[channelstr] / \
                        meanEnergy[channelstr]

            # end loop over channels

            ####################################################################
            #              insert results into transform structure             #
            ####################################################################


            # begin loop over channels
            for channel in np.arange(0,numberOfChannels):
                channelstr = 'channel' + str(channel)

                  # insert mean tile energy into frequency row structure
                transforms[channelstr][planestr][rowstr]['meanEnergy'] = \
                    meanEnergy[channelstr]

                # insert normalized tile energies into frequency row structure
                transforms[channelstr][planestr][rowstr]['normalizedEnergies'] = \
                normalizedEnergies[channelstr]

            # end loop over channels

        ########################################################################
        #                 end loop over frequency rows                         #
        ########################################################################


    ############################################################################
    #                        end loop over Q planes                            #
    ############################################################################

    ############################################################################
    #                return discrete Q transform structure                     #
    ############################################################################

    #for channel in np.arange(0,numberOfChannels):
    #    channelstr = 'channel' + str(channel)
    #     transforms[channelstr]['channelName'] = \
    #         outputChannelNames[channelstr]

    return transforms

###############################################################################
##########################                     ################################
##########################      wthreshold     ################################
##########################                     ################################
###############################################################################

def wthreshold(transforms, tiling, startTime, falseEventRate,
             referenceTime, timeRange, frequencyRange, qRange,
             maximumSignificants,analysisMode,falseVetoRate,
             uncertaintyFactor,correlationFactor, PSD):

    """WTHRESHOLD Identify statistically significant tiles in Discrete Q
    transforms

    WTHRESHOLD identifies discrete Q transform coefficients whose magnitudes
    exceed the threshold that approximately yields the specified single channel
    false rate assuming ideal white noise.

    usage: significants = ...
             wthreshold(transforms, tiling, startTime, falseEventRate, \
                       referenceTime, timeRange, frequencyRange, qRange, \
                        maximumSignificants, analysisMode, falseVetoRate, \
                        uncertaintyFactor, correlationFactor, debugLevel)

            transforms           cell array of input Q transform structures
            tiling               discrete Q transform tiling structure from WTILE
            startTime            GPS start time of Q transformed data
            falseEventRate       desired white noise false event rate [Hz]
            referenceTime        reference time for time range to threshold on
            timeRange            vector range of relative times to threshold on
            frequencyRange       vector range of frequencies to threshold on
            qRange               scalar Q or vector range of Qs to threshold on
            maximumSignificants  maximum allowable number of significant tiles
            analysisMode         string name of analysis mode to implement
            falseVetoRate        desired white noise veto rate [Hz]
            uncertaintyFactor    squared calibration uncertainty factor
            correlationFactor    fractional correlated energy threshold
            PSD                  measured PSD as outputted by wcondition

            significants         cell array of Q transform event structures

    WTHRESHOLD returns a cell array of Q transform event structures that
    contain the properties of the identified statistically significant tiles
    for each channel.  The event structure contains the following fields.

            time                 center time of tile [gps seconds]
            frequency            center frequency of tile [Hz]
            q                    quality factor of tile []
            duration             duration of tile [seconds]
            bandwidth            bandwidth of tile [Hz]
            normalizedEnergy     normalized energy of tile []
            amplitude            signal amplitude of tile [Hz^-1/2]
            overflowFlag         boolean overflow flag

    The user can focus on a subset of the times and frequencies available in
    the transform data by specifying a desired range of central times,
    central frequencies, and Qs to threshold on.  Ranges should be specified
    as a two component vector, consisting of a minimum and maximum value.
    Alternatively, if only a single Q is specified, WTHRESHOLD is only
    applied to the time-frequency plane which has the nearest value of Q in a
    logarithmic sense to the requested value.

    To determine the range of central times to threshold on, WTHRESHOLD
    requires the start time of the transformed data in addition to a
    reference time and a relative time range.  Both the start time and
    reference time should be specified as absolute quantities, while the
    range of times to analyze should be specified relative to the requested
    reference time.

    By default, WTHRESHOLD is applied to all available frequencies and Qs,
    and the reference time and relative time range arguments are set to
    exclude data potentially corrupted by filter transients as identified by
    the transient duration field of the tiling structure.  The default value
    can be obtained for any argument by passing the empty matrix [].

    The threshold is set to yield the specified false event rate when applied
    to all available frequencies and Qs, and is not modified to account for
    restricted ranges.  It is also only a rough estimate, and the result
    false event rate may vary significantly depending on the quality of the
    data.

    If provided, the optional analysisMode string is used by WTHRESHOLD to
    determine which channels are signal channels, which channels are null
    channels, and which channels to report results for.  For coherent analysis
    modes, a desired white noise veto rate, squared calibration uncertainy
    factor, and required signal correlation factor must also be specified.

    The optional maximumSignificants argument provides a safety mechanism to
    limit the total number of events returned by WTHRESHOLD.  If this maximum
    number of significants is exceeded, the overflow flag is set, only the
    maximumSignificants most significant tiles are returned, and a warning is
    issued if debugLevel is set to 1 or higher.  By default, maximumSignificants
    is set to infinity and debugLevel is set to unity.

    See also WTILE, WCONDITION, WTRANSFORM, WSELECT, WEXAMPLE, and WSEARCH.

    Authors:
    Shourov K. Chatterji <shourov@ligo.caltech.edu>
    Leo C. Stein <lstein@ligo.mit.edu>
    """

    ############################################################################
    #                    process command line arguments                        #
    ############################################################################

    # apply default arguments
    if referenceTime is None:
        referenceTime = startTime + tiling['generalparams']['duration'] / 2

    if timeRange is None:
        timeRange = 0.5 * (tiling['generalparams']['duration'] - 2 * tiling['generalparams']['transientDuration']) * np.array([-1,1])

    if frequencyRange is None:
        frequencyRange = [float('-Inf'),float('Inf')]

    if qRange is None:
        qRange = [float('-Inf'),float('Inf')]

    if maximumSignificants is None:
        maximumSignificants = float('Inf')

    if analysisMode is None:
        analysisMode = 'independent'

    if falseVetoRate is None:
        falseVetoRate = 0

    if uncertaintyFactor is None:
        uncertaintyFactor = 0

    if correlationFactor is None:
        correlationFactor = 0

    # determine number of channels
    numberOfChannels = 1

    # if only a single Q is requested, find nearest Q plane
    if len(qRange) == 1:
        [ignore, qPlane] = min(abs(np.log(tiling['generalparams']['qs']) / \
                                 qRange))
        qRange = tiling['generalparams']['qs'][qPlane] * [1,1]

    ############################################################################
    #                   validate command line arguments                        #
    ############################################################################

    # Check for two component range vectors
    if timeRange.size != 2:
        raise ValueError('Time range must be two component vector [tmin tmax].')

    if len(frequencyRange) != 2:
        raise ValueError('Frequency range must be two component vector [fmin fmax].')

    if len(qRange) > 2:
        raise ValueError('Q range must be scalar or two component vector [Qmin Qmax].')

    ############################################################################
    #                     normalized energy threshold                          #
    ############################################################################

    # approximate number of statistically independent tiles per second
    independentsRate = tiling["generalparams"]["numberOfIndependents"] / tiling['generalparams']['duration']

    # apply emperically determined correction factor
    independentsRate = independentsRate * 1.5

    # probability associated with desired false event rate
    falseEventProbability = falseEventRate / independentsRate

    # probability associated with desired false veto rate
    falseVetoProbability = falseVetoRate / independentsRate

    # normalized energy threshold for desired false event rate
    eventThreshold = -np.log(falseEventProbability)

    # normalized energy threshold for desired false veto rate
    if falseVetoProbability == 0:
        vetoThreshold = float('Inf')
    else:
        vetoThreshold = -np.log(falseVetoProbability)

    ############################################################################
    #                         apply analysis mode                              #
    ############################################################################

    # create empty cell array of significant event structures
    significants = {}

    # begin loop over channels
    for channel in np.arange(0,numberOfChannels):
        channelstr = 'channel' + str(channel)
        significants[channelstr] = {}

        ########################################################################
        #     initialize statistically significant event structure             #
        ########################################################################

        # initialize result vectors
        significants[channelstr]['time'] = np.array([])
        significants[channelstr]['frequency'] = np.array([])
        significants[channelstr]['q'] = np.array([])
        significants[channelstr]['duration'] = np.array([])
        significants[channelstr]['bandwidth'] = np.array([])
        significants[channelstr]['normalizedEnergy'] = np.array([])
        significants[channelstr]['amplitude']= np.array([])

        # initialize overflow flag
        significants[channelstr]['overflowFlag'] = 0

    # end loop over channels

    ############################################################################
    #                       begin loop over Q planes                           #
    ############################################################################

    # begin loop over Q planes
    numberOfPlanes = tiling['generalparams']['numberOfPlanes']
    for plane in np.arange(0,numberOfPlanes):
        planestr = 'plane' + str(plane)

        ########################################################################
        #                        threshold on Q                                #
        ########################################################################

        # skip Q planes outside of requested Q range
        if ((tiling[planestr]['q'] < min(qRange)) or \
          (tiling[planestr]['q'] > max(qRange))):
            continue

        ########################################################################
        #                begin loop over frequency rows                        #
        ########################################################################

        numberOfRows = tiling[planestr]['numberOfRows']
        # begin loop over frequency rows
        for row in np.arange(0,numberOfRows):
            rowstr = 'row' + str(row)

            ####################################################################
            #             threshold on central frequency                       #
            ####################################################################

            # skip frequency rows outside of requested frequency range
            if ((tiling[planestr][rowstr]['frequency'] < \
                 min(frequencyRange)) or \
                (tiling[planestr][rowstr]['frequency'] > \
                 max(frequencyRange))):
                    continue

            ####################################################################
            #               begin loop over channels                           #
            ####################################################################

            # begin loop over channels
            for channel in np.arange(0,numberOfChannels):
                channelstr = 'channel' + str(channel)

                ################################################################
                #          threshold on significance                           #
                ################################################################

                significantTileIndices = np.argwhere(\
                    transforms[channelstr][planestr][rowstr]\
                        ['normalizedEnergies'] >= eventThreshold)

                ################################################################
                #          threshold on central time                           #
                ################################################################

                times = np.arange(0,tiling[planestr][rowstr]\
                           ['numberOfTiles']) * \
                                     tiling[planestr][rowstr]['timeStep']

                # skip tiles outside requested time range
                keepIndices = \
                     np.logical_and(times[significantTileIndices] >= \
                       (referenceTime - startTime + min(timeRange)), \
                      times[significantTileIndices] <= \
                       (referenceTime - startTime + max(timeRange)))

                # Check to see ifs there are any indices outside the requested time range.
                significantTileIndices = significantTileIndices[keepIndices]

                # Check is there are any significant tile remaining
                if not any(significantTileIndices):
                    continue

                # number of statistically significant tiles in frequency row
                numberOfSignificants = significantTileIndices.size

                ################################################################
                #     append significant tile properties to event structure    #
                ################################################################

                # append center times of significant tiles in row
                significants[channelstr]['time'] = np.concatenate(\
                       (significants[channelstr]['time'],\
                       times[significantTileIndices] + startTime))

                # append center frequencies of significant tiles in row
                significants[channelstr]['frequency']= np.concatenate(\
                       (significants[channelstr]['frequency'],
                       tiling[planestr][rowstr]['frequency'] * \
                       np.ones(numberOfSignificants)))

                # append qs of significant tiles in row
                significants[channelstr]['q'] = np.concatenate(\
                       (significants[channelstr]['q'],
                       tiling[planestr]['q'] * \
                       np.ones(numberOfSignificants)))

                # append durations of significant tiles in row
                significants[channelstr]['duration'] = np.concatenate(\
                       (significants[channelstr]['duration'],
                       tiling[planestr][rowstr]['duration'] * \
                       np.ones(numberOfSignificants)))

                # append bandwidths of significant tiles in row
                significants[channelstr]['bandwidth'] = np.concatenate(\
                       (significants[channelstr]['bandwidth'],
                       tiling[planestr][rowstr]['bandwidth'] * \
                       np.ones(numberOfSignificants)))

                # append normalized energies of significant tiles in row
                significants[channelstr]['normalizedEnergy'] = np.concatenate(\
                       (significants[channelstr]['normalizedEnergy'],
                       transforms[channelstr][planestr][rowstr]\
                       ['normalizedEnergies'][significantTileIndices]))

                if PSD is not None:
                    #FIXME: I did not rewrite this MATLAB code because it seemed a little convulted and I am not sure will even be used in qscans.
                    print('FIXME: I did not rewrite this MATLAB code \
                           because it seemed a little convulted \
                           and I am not sure will even be used in qscans.')
                else:
                    significants[channelstr]['amplitude'] = np.concatenate(\
                        (significants[channelstr]['amplitude'],
                          np.sqrt((transforms[channelstr][planestr][rowstr]\
                          ['normalizedEnergies'][significantTileIndices] - 1) *\
                            transforms[channelstr][planestr][rowstr]['meanEnergy'])))

                ################################################################
                #   prune excessive significants as we accumulate              #
                ################################################################

                # determine number of significant tiles in channel
                numberOfSignificants = len(significants[channelstr]['time'])

                # if maximum allowable number of significant tiles is exceeded
                if numberOfSignificants > maximumSignificants:

                    # set overflow flag
                    significants[channelstr]['overflowFlag'] = 1

                    # find indices of max normalized energy
                    maximumIndices = np.argsort(\
                                 significants[channelstr]['normalizedEnergy'])

                    # find indices of most significant tiles
                    maximumIndices = maximumIndices[::-1]

                    # extract most significant tile properties
                    significants[channelstr]['time'][maximumIndices]
                    significants[channelstr]['frequency'][maximumIndices]
                    significants[channelstr]['q'][maximumIndices]
                    significants[channelstr]['duration'][maximumIndices]
                    significants[channelstr]['bandwidth'][maximumIndices]
                    significants[channelstr]['normalizedEnergy'][maximumIndices]
                    significants[channelstr]['amplitude'][maximumIndices]

                    # otherwise continue

            ####################################################################
            #                end loop over channels                            #
            ####################################################################

            # end loop over channels

        ########################################################################
        #                 end loop over frequency rows                         #
        ########################################################################

        # end loop over frequency rows

     ###########################################################################
     #                       end loop over Q planes                            #
     ###########################################################################

     # end loop over Q planes

     ###########################################################################
     #               return statistically significant tiles                    #
     ###########################################################################

    return significants

def wselect(significants, durationInflation, \
                          bandwidthInflation, maximumEvents):
    """WSELECT Identify statistically significant events in Discrete Q
       transforms.

       WSELECT selects statistically significant events from the set
       of statitically significant Q transform tiles.  Events are
       defined by the properties of their most significant tile and
       are identified by exluding the less significant of any overlapping
       tiles.  The input significant tiles are first sorted by decreasing
       normalized energy.  Starting with the most significant tile, tiles are
       discarded if they overlap with a more significant tile.
       The remaining set of tiles comprises a minimal set of tiles
       that describes an event.

       WSELECT returns a cell array of significant event properties

       usage: events = wselect(significants, durationInflation, \
                         bandwidthInflation, maximumEvents)

       significants         cell array of significant tiles properties
       durationInflation    multiplicative scale factor for duration
       bandwidthInflation   multiplicative scale factor for bandwidth
       maximumEvents        maximum allowable number of events

       events               cell array of significant event properties

       The optional durationInflation and bandwidthInflation arguments are
       multiplicative scale factors that are applied to the duration and
       bandwidth of significant tiles prior to testing for overlap.
       If not specified, these parameters both default to unity such that the
       resulting tiles have unity time-frequency area.  The normalized energy of
       the resulting tiles are scaled by the product of the duration and
       bandwidth inflation factors to avoid over
       counting the total energy of clusters of tiles.  Likewise, the
       amplitude of the resulting tiles is scaled by the square root of the
       product of the duration and bandwidth inflation factors.

       The optional maximumEvents argument provides a safety mechanism to
       limit the total number of events returned by WSELECT.  If this maximum
       number of events is exceeded, an overflow flag is set, only the
       maximumEvents most significant events are returned, and a warning
       is issued if debugLevel is set to unity or higher.
       By default, maximumEvents is set to infinity and debugLevel is set to
       unity.

       WSELECT both expects and returns a cell array of Q transform event
       structures with one cell per channel.  The event structures contain
       the following required fields, which describe the properties of
       statistically significant tiles.  Additional fields such as amplitude,
       phase, or coherent transform properties are optional and are
       retained along with the required fields.

           time                 center time of tile [gps seconds]
           frequency            center frequency of tile [Hz]
           duration             duration of tile [seconds]
           bandwidth            bandwidth of tile [Hz]
           normalizedEnergy     normalized energy of tile []

       The event structures also contain the following flag, which indicates
       if the maximum number of significant tiles or significant events was
       exceeded.

           overflowFlag         boolean overflow flag

       See also WTILE, WCONDITION, WTRANSFORM, WTHRESHOLD, WEXAMPLE, and WSEARCH

       Shourov K. Chatterji <shourov@ligo.caltech.edu>
       Leo C. Stein <lstein@ligo.mit.edu>
       """

    ############################################################################
    #######################   Process Command Line  ############################
    ############################################################################
    # apply default arguments
    if durationInflation is None:
        durationInflation = 1.0

    if bandwidthInflation is None:
        bandwidthInflation = 1.0

    if maximumEvents is None:
        maximumEvents = float('Inf')

    # determine number of channels
    numberOfChannels = 1 # len(significants)

    ############################################################################
    #        initialize statistically significant events structures            #
    ############################################################################
    # create empty array of significant event indices
    eventIndices = {}

    # create empty cell array of significant event structuresi and indices
    events = {}

    for channel in np.arange(0,numberOfChannels):
        channelstr = 'channel' + str(channel)
        eventIndices[channelstr] = {}
        events[channelstr]       = {}

    # begin loop over channels
    for channel in np.arange(0,numberOfChannels):
        channelstr = 'channel' + str(channel)
        # propogate overflow flag
        events[channelstr]['overflowFlag'] = \
            significants[channelstr]['overflowFlag']

    ############################################################################
    #                       begin loop over channels                           #
    ############################################################################

    # begin loop over channels
    for channelNumber in np.arange(0,numberOfChannels):
        channelstr = 'channel' + str(channel)

        ######################################################################
        #           sort by decreasing normalized energy                     #
        ######################################################################

        # sort tile indices by normalized energy
        sortedIndices = \
            np.argsort(significants[channelstr]['normalizedEnergy'])

        # sort by decreasing normalized energy
        sortedIndices = sortedIndices[::-1]

        # reorder significant tile properties by decreasing normalized energy
        significants[channelstr]['time'] = \
            significants[channelstr]['time'][sortedIndices]
        significants[channelstr]['frequency'] = \
            significants[channelstr]['frequency'][sortedIndices]
        significants[channelstr]['q'] = \
            significants[channelstr]['q'][sortedIndices]
        significants[channelstr]['duration'] = \
            significants[channelstr]['duration'][sortedIndices]
        significants[channelstr]['bandwidth'] = \
            significants[channelstr]['bandwidth'][sortedIndices]
        significants[channelstr]['normalizedEnergy'] = \
            significants[channelstr]['normalizedEnergy'][sortedIndices]
        significants[channelstr]['amplitude'] = \
            significants[channelstr]['amplitude'][sortedIndices]

        ######################################################################
        #                   find tile boundaries                             #
        ######################################################################

        # vector of significant tile start times
        minimumTimes = significants[channelstr]['time'] - \
            durationInflation * \
            significants[channelstr]['duration'] / 2

        # vector of significant tile stop times
        maximumTimes = significants[channelstr]['time'] + \
            durationInflation * \
            significants[channelstr]['duration'] / 2

        # vector of significant tile lower frequencies
        minimumFrequencies = significants[channelstr]['frequency'] - \
            bandwidthInflation * \
            significants[channelstr]['bandwidth'] / 2

        # vector of significant tile upper frequencies
        maximumFrequencies = significants[channelstr]['frequency'] + \
            bandwidthInflation * \
            significants[channelstr]['bandwidth'] / 2

        ######################################################################
        #              compress significant tile list                        #
        ######################################################################

        # number of significant tiles in list
        numberOfTiles = len(significants[channelstr]['time'])

        # if input significant tile list is empty,
        if numberOfTiles == 0:

            # set empty event list
            eventIndices[channelstr] = []
            events[channelstr]['time'] = np.array([])
            events[channelstr]['frequency'] = np.array([])
            events[channelstr]['q'] = np.array([])
            events[channelstr]['duration'] = np.array([])
            events[channelstr]['bandwidth'] = np.array([])
            events[channelstr]['normalizedEnergy'] = np.array([])
            events[channelstr]['av_frequency'] = np.array([])
            events[channelstr]['av_bandwidth'] = np.array([])
            events[channelstr]['err_frequency'] =np.array([])
            events[channelstr]['tot_normalizedEnergy'] = np.array([])

            # skip to next channel
            break

        # initialize event list
        eventIndices[channelstr] = []

        # initialize some variables for the significants array.
        significants[channelstr]['av_frequency']  = np.zeros(numberOfTiles)
        significants[channelstr]['av_bandwidth']  = np.zeros(numberOfTiles)
        significants[channelstr]['err_frequency'] = np.zeros(numberOfTiles)
        significants[channelstr]['tot_normalizedEnergy'] = np.zeros(numberOfTiles)

        # begin loop over significant tiles
        for tileIndex in np.arange(0,numberOfTiles):

            # determine if current tile overlaps any events
            overlap = np.logical_and(\
                       np.logical_and(\
                         (minimumTimes[tileIndex] < \
                             maximumTimes[eventIndices[channelstr]]),\
                         (maximumTimes[tileIndex] > \
                             minimumTimes[eventIndices[channelstr]])), \
                       np.logical_and(\
                         (minimumFrequencies[tileIndex] < \
                             maximumFrequencies[eventIndices[channelstr]]),\
                         (maximumFrequencies[tileIndex] > \
                             minimumFrequencies[eventIndices[channelstr]])))

            # if tile does not overlap with any event,
            if not np.any(overlap):
                # append it to the list of events
                eventIndices[channelstr].append(tileIndex)

                # initilaize averages over tiles in the event
                significants[channelstr]['av_frequency'][tileIndex] = \
                    significants[channelstr]['frequency'][tileIndex]

                significants[channelstr]['av_bandwidth'][tileIndex] = \
                    significants[channelstr]['bandwidth'][tileIndex]

                significants[channelstr]['err_frequency'][tileIndex] = \
                    significants[channelstr]['bandwidth'][tileIndex]/\
                        significants[channelstr]['frequency'][tileIndex]

                significants[channelstr]['tot_normalizedEnergy'][tileIndex] = \
                    significants[channelstr]['normalizedEnergy'][tileIndex]

            # otherwise, add to averages over tiles in the event
            else:
                overlap = np.concatenate((overlap,np.zeros(\
                    significants[channelstr]['av_frequency'].size \
                        - overlap.size,dtype=bool)))

                tmp_av_frequency = \
                  significants[channelstr]['av_frequency'][overlap]*\
                  significants[channelstr]['tot_normalizedEnergy'][overlap] +\
                  significants[channelstr]['frequency'][tileIndex]*\
                  significants[channelstr]['normalizedEnergy'][tileIndex]

                tmp_av_bandwidth = \
                  significants[channelstr]['av_bandwidth'][overlap]*\
                  significants[channelstr]['tot_normalizedEnergy'][overlap] +\
                  significants[channelstr]['bandwidth'][tileIndex]*\
                  significants[channelstr]['normalizedEnergy'][tileIndex]

                tmp_2nd_moment_frequency = \
                  significants[channelstr]['tot_normalizedEnergy'][overlap]*\
                  (significants[channelstr]['err_frequency'][overlap]**2 + 1)*\
                  significants[channelstr]['av_frequency'][overlap]**2 +\
                  significants[channelstr]['normalizedEnergy'][tileIndex]*\
                  significants[channelstr]['frequency'][tileIndex]**2

                significants[channelstr]['tot_normalizedEnergy'][overlap] = \
                  significants[channelstr]['tot_normalizedEnergy'][overlap] + \
                  significants[channelstr]['normalizedEnergy'][tileIndex]

                significants[channelstr]['av_frequency'][overlap] = \
                  tmp_av_frequency/ \
                  significants[channelstr]['tot_normalizedEnergy'][overlap]

                significants[channelstr]['av_bandwidth'][overlap] = \
                  tmp_av_bandwidth/ \
                  significants[channelstr]['tot_normalizedEnergy'][overlap]

                significants[channelstr]['err_frequency'][overlap] = \
                  np.sqrt( (tmp_2nd_moment_frequency - \
                  tmp_av_frequency*significants[channelstr]\
                                    ['av_frequency'][overlap])/ \
                  significants[channelstr]['tot_normalizedEnergy'][overlap])/\
                  significants[channelstr]['av_frequency'][overlap]

            # end loop over significant tiles

        # extract events from significant tiles
        events[channelstr]['time'] = significants[channelstr]\
            ['time'][eventIndices[channelstr]]
        events[channelstr]['frequency'] = significants[channelstr]\
            ['frequency'][eventIndices[channelstr]]
        events[channelstr]['q'] = significants[channelstr]\
            ['q'][eventIndices[channelstr]]
        events[channelstr]['duration'] = significants[channelstr]\
            ['duration'][eventIndices[channelstr]]
        events[channelstr]['bandwidth'] = significants[channelstr]\
            ['bandwidth'][eventIndices[channelstr]]
        events[channelstr]['normalizedEnergy'] = significants[channelstr]\
            ['normalizedEnergy'][eventIndices[channelstr]]
        events[channelstr]['av_frequency'] = significants[channelstr]\
            ['av_frequency'][eventIndices[channelstr]]
        events[channelstr]['av_bandwidth'] = significants[channelstr]\
            ['av_bandwidth'][eventIndices[channelstr]]
        events[channelstr]['err_frequency'] = significants[channelstr]\
            ['err_frequency'][eventIndices[channelstr]]
        events[channelstr]['tot_normalizedEnergy'] = significants[channelstr]\
            ['tot_normalizedEnergy'][eventIndices[channelstr]]

        ######################################################################
        #            check for excessive number of events                    #
        ######################################################################

        # determine number of significant tiles in channel
        numberOfEvents = len(events[channelstr]['time'])

        # if maximum allowable number of significant tiles is exceeded
        if numberOfEvents > maximumEvents:

            # issue warning
            print('WARNING: maximum number of events exceeded.\n')

            # set overflow flag
            events[channelstr]['overflowFlag'] = 1

            # indices of most significant tiles
            maximumIndices = np.arange(0,maximumEvents).astype('int')

            # truncate lists of significant event properties
            events[channelstr]['time'] = significants[channelstr]['time'][maximumIndices]
            events[channelstr]['frequency'] = significants[channelstr]['frequency'][maximumIndices]
            events[channelstr]['q'] = significants[channelstr]['q'][maximumIndices]
            events[channelstr]['duration'] = significants[channelstr]['duration'][maximumIndices]
            events[channelstr]['bandwidth'] = significants[channelstr]['bandwidth'][maximumIndices]
            events[channelstr]['normalizedEnergy'] = significants[channelstr]['normalizedEnergy'][maximumIndices]
            events[channelstr]['av_frequency'] = significants[channelstr]['av_frequency'][maximumIndices]
            events[channelstr]['av_bandwidth'] = significants[channelstr]['av_bandwidth'][maximumIndices]
            events[channelstr]['err_frequency'] = significants[channelstr]['err_frequency'][maximumIndices]
            events[channelstr]['tot_normalizedEnergy'] = significants[channelstr]['tot_normalizedEnergy'][maximumIndices]

            # otherwise continue

    ############################################################################
    #                        end loop over channels                            #
    ############################################################################

    # end loop over channels

    ############################################################################
    #               return statistically significant events                    #
    ############################################################################

    return events

###############################################################################
##########################                     ################################
##########################      wmeasure       ################################
##########################                     ################################
###############################################################################

def wmeasure(transforms, tiling, startTime,
             referenceTime, timeRange, frequencyRange,
             qRange):
    """Measure peak and weighted signal properties from Q transforms

    WMEASURE reports the peak and significance weighted mean properties of Q
    transformed signals within the specified time-frequency region.

    usage:

      measurements = wmeasure(transforms, tiling, startTime, referenceTime, \
                              timeRange, frequencyRange, qRange)

      transforms           cell array of input Q transform structures
      tiling               discrete Q transform tiling structure from WTILE
      startTime            GPS start time of Q transformed data
      referenceTime        reference time for time range to search over
      timeRange            vector range of relative times to search over
      frequencyRange       vector range of frequencies to search over
      qRange               scalar Q or vector range of Qs to search over

      measurements         cell array of measured signal properties

    WMEASURE returns a cell array of measured signal properties, with one cell per
    channel.  The measured signal properties are returned as a structure that
    contains the following fields.

      peakTime                 center time of peak tile [gps seconds]
      peakFrequency            center frequency of peak tile [Hz]
      peakQ                    quality factor of peak tile []
      peakDuration             duration of peak tile [seconds]
      peakBandwidth            bandwidth of peak tile [Hz]
      peakNormalizedEnergy     normalized energy of peak tile []
      peakAmplitude            amplitude of peak tile [Hz^-1/2]
      signalTime               weighted central time [gps seconds]
      signalFrequency          weighted central frequency [Hz]
      signalDuration           weighted duration [seconds]
      signalBandwidth          weighted bandwidth [Hz]
      signalNormalizedEnergy   total normalized energy []
      signalAmplitude          total signal amplitude [Hz^-1/2]
      signalArea               measurement time frequency area []

    The user can focus on a subset of the times and frequencies available in
    the transform data by specifying a desired range of central times,
    central frequencies, and Qs to threshold on.  Ranges should be specified
    as a two component vector, consisting of a minimum and maximum value.
    Alternatively, if only a single Q is specified, WMEASURE is only applied to
    the time-frequency plane which has the nearest value of Q in a
    logarithmic sense to the requested value.

    To determine the range of central times to search over, WMEASURE requires
    the start time of the transformed data in addition to a reference time
    and a relative time range.  Both the start time and reference time should
    be specified as absolute quantities, while the range of times to analyze
    should be specified relative to the requested reference time.

    By default, WMEASURE is applied to all available frequencies and Qs, and the
    reference time and relative time range arguments are set to exclude data
    potentially corrupted by filter transients as identified by the transient
    duration field of the tiling structure.  The default value can be
    obtained for any argument by passing the empty matrix [].

    See also WTILE, WCONDITION, WTRANSFORM, WTHRESHOLD, WSELECT, WEXAMPLE, WSCAN,
    and WSEARCH.

    Notes:
    1. Compute absolute or normalized energy weighted signal properties?
    2. Only include tiles with Z>Z0 in integrands?

    Shourov K. Chatterji
    shourov@ligo.caltech.edu
    """

    ###########################################################################
    #                   process command line arguments                        #
    ###########################################################################

    # apply default arguments
    if not referenceTime:
        referenceTime = startTime + tiling['generalparams']['duration'] / 2

#    if not timeRange:
#      timeRange = 0.5 * \
#    (tiling['generalparams']['duration'] - 2 * tiling['generalparams']['transientDuration']) * [-1 +1]

    if not frequencyRange:
        frequencyRange = [float('-Inf'),float('Inf')]

    if not qRange:
        qRange = [float('-Inf'),float('Inf')]

    # determine number of channels
    numberOfChannels = len(transforms)

    # if only a single Q is requested, find nearest Q plane
    if len(qRange) == 1:
        [ignore, qPlane] = min(abs(np.log(tiling['generalparams']['qs']) / \
                                 qRange))
        qRange = tiling['generalparams']['qs'][qPlane] * [1,1]

    ##########################################################################
    #                    validate command line arguments                     #
    ##########################################################################

    # Check for two component range vectors
    if len(timeRange) != 2:
        error('Time range must be two component vector [tmin tmax].')

    if len(frequencyRange) != 2:
        error('Frequency range must be two component vector [fmin fmax].')

    if len(qRange) > 2:
        error('Q range must be scalar or two component vector [Qmin Qmax].')


    ##########################################################################
    #                   initialize measurement structures                    #
    ##########################################################################

    # Initialize measurements cell
    measurements = {}
    peakPlane = {}
    numberOfPlanes = tiling['generalparams']['numberOfPlanes'].astype('int')

    # create empty cell array of measurement structures
    # begin loop over channels
    for channel in np.arange(0,numberOfChannels):
        channelstr = 'channel' + str(channel)
        measurements[channelstr] = {}
        # End loop over channels

    # begin loop over channels
    for channel in np.arange(0,numberOfChannels):
        channelstr = 'channel' + str(channel)
        # initialize peak signal properties
        measurements[channelstr]['peakTime'] = 0
        measurements[channelstr]['peakFrequency'] = 0
        measurements[channelstr]['peakQ'] = 0
        measurements[channelstr]['peakDuration'] = 0
        measurements[channelstr]['peakBandwidth'] = 0
        measurements[channelstr]['peakNormalizedEnergy'] = 0
        measurements[channelstr]['peakAmplitude'] = 0

        # initialize integrated signal properties
        measurements[channelstr]['signalTime'] = \
          np.zeros(numberOfPlanes)
        measurements[channelstr]['signalFrequency'] = \
          np.zeros(numberOfPlanes)
        measurements[channelstr]['signalDuration'] = \
          np.zeros(numberOfPlanes)
        measurements[channelstr]['signalBandwidth'] = \
          np.zeros(numberOfPlanes)
        measurements[channelstr]['signalNormalizedEnergy'] = \
          np.zeros(numberOfPlanes)
        measurements[channelstr]['signalAmplitude'] = \
          np.zeros(numberOfPlanes)
        measurements[channelstr]['signalArea'] = \
          np.zeros(numberOfPlanes)

        # end loop over channels
    numberOfPlanes = numberOfPlanes.astype('float')

    ###########################################################################
    #                      begin loop over Q planes                           #
    ###########################################################################

    # begin loop over Q planes
    for plane in np.arange(0,numberOfPlanes):
        planestr = 'plane' +str(plane)
        plane = plane.astype('int')
        numberOfRows = tiling[planestr]['numberOfRows']

        #######################################################################
        #                       threshold on Q                                #
        #######################################################################

        # skip Q planes outside of requested Q range
        if ((tiling[planestr]['q'] < min(qRange)) or \
          (tiling[planestr]['q'] > max(qRange))):
            continue

        #######################################################################
        #               begin loop over frequency rows                        #
        #######################################################################

        # begin loop over frequency rows
        for row in np.arange(0,numberOfRows):
            rowstr = 'row' + str(row)

            ###################################################################
            #                  calculate times                                #
            ###################################################################

            times = np.arange(0,tiling[planestr][rowstr]['numberOfTiles']) * \
                   tiling[planestr][rowstr]['timeStep']

            ###################################################################
            #           threshold on central frequency                        #
            ###################################################################

            # skip frequency rows outside of requested frequency range
            if ((tiling[planestr][rowstr]['frequency'] < \
                 min(frequencyRange)) or \
                (tiling[planestr][rowstr]['frequency'] > \
                 max(frequencyRange))):
                    continue

            ###################################################################
            #             threshold on central time                           #
            ###################################################################

            # skip tiles outside requested time range
            tileIndices = \
                np.logical_and(times >= \
                  (referenceTime - startTime + min(timeRange)), \
                 times <= \
                  (referenceTime - startTime + max(timeRange)))
            ###################################################################
            #  differential time-frequency area for integration               #
            ###################################################################

            # differential time-frequency area for integration
            differentialArea = tiling[planestr][rowstr]['timeStep'] * \
                           tiling[planestr][rowstr]['frequencyStep']

            ###################################################################
            #              begin loop over channels                           #
            ###################################################################

            # begin loop over channels
            for channel in np.arange(0,numberOfChannels):
                channelstr = 'channel' + str(channel)

                ###############################################################
                #        update peak tile properties                          #
                ###############################################################

                # vector of row tile normalized energies
                normalizedEnergies = transforms[channelstr][planestr][rowstr]\
                        ['normalizedEnergies'][tileIndices]
                # find most significant tile in row
                peakNormalizedEnergy = np.max(normalizedEnergies)
                peakIndex            = np.argmax(normalizedEnergies).astype('int')

                # if peak tile is in this row
                if peakNormalizedEnergy > \
                        measurements[channelstr]['peakNormalizedEnergy']:

                    # update plane index of peak tile
                    peakPlane[channelstr] = plane

                    # update center time of peak tile
                    measurements[channelstr]['peakTime'] = \
                        times[tileIndices][peakIndex] + startTime

                    # update center frequency of peak tile
                    measurements[channelstr]['peakFrequency'] = \
                        tiling[planestr][rowstr]['frequency']

                    # update q of peak tile
                    measurements[channelstr]['peakQ']= \
                        tiling[planestr]['q']

                    # update duration of peak tile
                    measurements[channelstr]['peakDuration'] = \
                                tiling[planestr][rowstr]['duration']

                    # update bandwidth of peak tile
                    measurements[channelstr]['peakBandwidth'] = \
                                tiling[planestr][rowstr]['bandwidth']

                    # update normalized energy of peak tile
                    measurements[channelstr]['peakNormalizedEnergy'] = \
                    peakNormalizedEnergy

                    # udpate amplitude of peak tile
                    measurements[channelstr]['peakAmplitude'] = \
                        np.sqrt((peakNormalizedEnergy - 1) * transforms[channelstr][planestr][rowstr]['meanEnergy'])

                # End If Statement

                ###############################################################
                #                update weighted signal properties            #
                ###############################################################

                # threshold on significance
                normalizedEnergyThreshold = 4.5
                significantIndices = np.where(normalizedEnergies > \
                                                normalizedEnergyThreshold)
                normalizedEnergies = normalizedEnergies[significantIndices]

                # vector of row tile calibrated energies
                calibratedEnergies = (normalizedEnergies - 1) * \
                    transforms[channelstr][planestr][rowstr]['meanEnergy'] * \
                        tiling[planestr]['normalization']

                # sum of normalized tile energies in row
                sumNormalizedEnergies = sum(normalizedEnergies)

                # sum of calibrated tile enregies in row
                sumCalibratedEnergies = sum(calibratedEnergies)

                # update weighted central time integral
                measurements[channelstr]['signalTime'][plane] = \
                    measurements[channelstr]['signalTime'][plane] + \
                        sum(times[tileIndices][significantIndices] * \
                            calibratedEnergies) * \
                            differentialArea

                # update weighted central frequency integral
                measurements[channelstr]['signalFrequency'][plane] = \
                    measurements[channelstr]['signalFrequency'][plane] + \
                        tiling[planestr][rowstr]['frequency'] * \
                            sumCalibratedEnergies * \
                                differentialArea

                # update weighted duration integral
                measurements[channelstr]['signalDuration'][plane] = \
                    measurements[channelstr]['signalDuration'][plane] + \
                            sum(times[tileIndices][significantIndices]**2 * \
                                calibratedEnergies) * \
                                differentialArea

                # update weighted bandwidth integral
                measurements[channelstr]['signalBandwidth'][plane] = \
                    measurements[channelstr]['signalBandwidth'][plane] + \
                        tiling[planestr][rowstr]['frequency']**2 * \
                            sumCalibratedEnergies * \
                                differentialArea

                # update total normalized energy integral
                measurements[channelstr]['signalNormalizedEnergy'][plane] = \
                    measurements[channelstr]['signalNormalizedEnergy'][plane] + \
                        sumNormalizedEnergies * \
                            differentialArea

                # update total calibrated energy integral
                measurements[channelstr]['signalAmplitude'][plane] = \
                    measurements[channelstr]['signalAmplitude'][plane] + \
                        sumCalibratedEnergies * \
                            differentialArea

                # update total signal area integral
                measurements[channelstr]['signalArea'][plane] = \
                    measurements[channelstr]['signalArea'][plane] + \
                        len(normalizedEnergies) * \
                            differentialArea

            ##################################################################
            #              end loop over channels                            #
            ##################################################################


        ######################################################################
        #               end loop over frequency rows                         #
        ######################################################################


        ######################################################################
        #               normalize signal properties                          #
        ######################################################################

        # begin loop over channels
        for channel in np.arange(0,numberOfChannels):
            channelstr = 'channel' + str(channel)
            # normalize weighted signal properties by total normalized energy
            if measurements[channelstr]['signalAmplitude'][plane] != 0:

                measurements[channelstr]['signalTime'][plane] = \
                  measurements[channelstr]['signalTime'][plane] / \
                    measurements[channelstr]['signalAmplitude'][plane]

                measurements[channelstr]['signalFrequency'][plane] = \
                  measurements[channelstr]['signalFrequency'][plane] / \
                    measurements[channelstr]['signalAmplitude'][plane]

                measurements[channelstr]['signalDuration'][plane] = \
                  measurements[channelstr]['signalDuration'][plane] / \
                    measurements[channelstr]['signalAmplitude'][plane]

                measurements[channelstr]['signalBandwidth'][plane] = \
                  measurements[channelstr]['signalBandwidth'][plane] / \
                    measurements[channelstr]['signalAmplitude'][plane]

            # End If Statement

            # duration and bandwidth are second central
            # moments in time and frequency
            '''Not sure what this means...'''
            measurements[channelstr]['signalDuration'][plane] = \
                np.sqrt(measurements[channelstr]['signalDuration'][plane] - \
                     measurements[channelstr]['signalTime'][plane]**2)
            measurements[channelstr]['signalBandwidth'][plane] = \
                np.sqrt(measurements[channelstr]['signalBandwidth'][plane] - \
                     measurements[channelstr]['signalTime'][plane]**2)

            # convert signal energy to signal amplitude
            measurements[channelstr]['signalAmplitude'][plane] = \
                np.sqrt(measurements[channelstr]['signalAmplitude'][plane])

            # add start time to measured central time
            measurements[channelstr]['signalTime'][plane] = \
                measurements[channelstr]['signalTime'][plane] + startTime

        # end loop over channels


    ######################################################################
    #                  end loop over Q planes                            #
    ######################################################################

    ######################################################################
    #   report signal properties from plane with peak significance       #
    ######################################################################

    for channel in np.arange(0,numberOfChannels):
        channelstr = 'channel' + str(channel)

        # weighted central time estimate from plane with peak tile significance
        measurements[channelstr]['signalTime'] = \
            measurements[channelstr]['signalTime'][peakPlane[channelstr]]

        # weighted central frequency estimate from plane with peak tile significance
        measurements[channelstr]['signalFrequency'] = \
            measurements[channelstr]['signalFrequency'][peakPlane[channelstr]]

        # weighted duration estimate from plane with peak tile significance
        measurements[channelstr]['signalDuration'] = \
            measurements[channelstr]['signalDuration'][peakPlane[channelstr]]

        # weighted bandwidth estimate from plane with peak tile significance
        measurements[channelstr]['signalBandwidth'] = \
            measurements[channelstr]['signalBandwidth'][peakPlane[channelstr]]

        # total signal normalized energy estimate from plane with peak tile significance
        measurements[channelstr]['signalNormalizedEnergy'] = \
            measurements[channelstr]['signalNormalizedEnergy'][peakPlane[channelstr]]

        # total signal amplitude estimate from plane with peak tile significance
        measurements[channelstr]['signalAmplitude'] = \
            measurements[channelstr]['signalAmplitude'][peakPlane[channelstr]]

        # measured time frequency area in plane with peak tile significance
        measurements[channelstr]['signalArea'] = \
            measurements[channelstr]['signalArea'][peakPlane[channelstr]]

        # report peak tile properties for very weak signals
        if measurements[channelstr]['signalArea'] < 1:
            measurements[channelstr]['signalTime'] = \
                measurements[channelstr]['peakTime']
            measurements[channelstr]['signalFrequency'] = \
                measurements[channelstr]['peakFrequency']
            measurements[channelstr]['signalDuration'] = \
                measurements[channelstr]['peakDuration']
            measurements[channelstr]['signalBandwidth'] = \
                measurements[channelstr]['peakBandwidth']
            measurements[channelstr]['signalNormalizedEnergy'] = \
                measurements[channelstr]['peakNormalizedEnergy']
            measurements[channelstr]['signalAmplitude'] = \
                measurements[channelstr]['peakAmplitude']
            measurements[channelstr]['signalArea'] = 1

        # End If statment

    # end loop over channels

    ###########################################################################
    #           return most significant tile properties                       #
    ###########################################################################

    return measurements

###############################################################################
##########################                     ################################
##########################      wspectrogram   ################################
##########################                     ################################
###############################################################################

def wspectrogram(transforms, tiling, outDir,IDstring,startTime,
                 referenceTime, timeRange, frequencyRange, qRange,
                 plotNormalizedERange, horizontalResolution,detectorName):

    """Display time-frequency Q transform spectrograms
    """

    ############################################################################
    #                      identify q plane to display                         #
    ############################################################################
    # find plane with Q nearest the requested value
    planeIndices = np.argmin(abs(np.log(tiling['generalparams']['qs'] / qRange)))
    # number of planes to display
    numberOfPlanes = planeIndices.size

    # index of plane in tiling structure
    planeIndex = planeIndices
    planeIndexstr = 'plane' + str(float(planeIndex))

    ########################################################################
    #               identify frequency rows to display                     #
    ########################################################################

    # vector of frequencies in plane
    frequencies = tiling[planeIndexstr]['frequencies']

    # find rows within requested frequency range
    rowIndices = np.logical_and(frequencies >= np.min(frequencyRange),
                  frequencies <= np.max(frequencyRange))
    rowIndices = np.where(rowIndices==True)[0]
    # pad by one row if possible
    #if rowIndices(1) > 1,
    #  rowIndices = [rowIndices(1) - 1 rowIndices]
    #if rowIndices(end) < length(frequencies),
    #  rowIndices = [rowIndices rowIndices(end) + 1]

    # vector of frequencies to display
    frequencies = frequencies[rowIndices]
    # number of rows to display
    numberOfRows = frequencies.size

    ############################################################################
    #                       initialize display matrix                          #
    ############################################################################

    normalizedEnergies = {}
    # initialize matrix of normalized energies for display
    for iN in np.arange(0,len(timeRange)):
        timestr = 'time' + str(iN)
        normalizedEnergies[timestr] = np.zeros((numberOfRows, horizontalResolution))
        timeRange1 = {}
        times      = {}

    for iN in np.arange(0,len(timeRange)):
        timestr = 'time' + str(iN)
        # vector of times to display
        timeRange1[timestr] = timeRange[iN] * np.array([-1,1])*0.5
        times[timestr] = np.linspace(min(timeRange1[timestr]), max(timeRange1[timestr]), horizontalResolution)

    ############################################################################
    #                     begin loop over frequency rows                       #
    ############################################################################
    # loop over rows
    for row in np.arange(0,numberOfRows):

        # index of row in tiling structure
        rowIndex = rowIndices[row]
        rowstr   = 'row' + str(float(rowIndex))

        # vector of times in plane
        rowTimes = \
            np.arange(0,tiling[planeIndexstr][rowstr]['numberOfTiles'])\
            * tiling[planeIndexstr][rowstr]['timeStep'] + \
            (startTime - referenceTime)

        # find tiles within requested time range
        for iN in np.arange(0,len(timeRange)):
            timestr = 'time' + str(iN)
            # vector of times to display
            padTime = 1.5 * tiling[planeIndexstr][rowstr]['timeStep']
            tileIndices = np.logical_and((rowTimes >= \
                                  np.min(timeRange1[timestr]) - padTime),\
                                         (rowTimes <= np.max(timeRange1[timestr]) + padTime))

            # vector of times to display
            rowTimestemp = rowTimes[tileIndices]

            # corresponding tile normalized energies
            rowNormalizedEnergies = transforms['channel0'][planeIndexstr][rowstr]['normalizedEnergies'][tileIndices]
            # interpolate to desired horizontal resolution
            f = InterpolatedUnivariateSpline(rowTimestemp, rowNormalizedEnergies)

            rowNormalizedEnergies = f(times[timestr])

            # insert into display matrix
            normalizedEnergies[timestr][row, :] = rowNormalizedEnergies

    ########################################################################
    #                  end loop over frequency rows                        #
    ########################################################################


    ########################################################################
    #                  Plot spectrograms                                   #
    ########################################################################
    for iN in np.arange(0,len(timeRange)):
        timestr = 'time' + str(iN)
        time = times[timestr]
        freq = frequencies
        Energ = normalizedEnergies[timestr]

        fig, axSpectrogram = plt.subplots()

        # Make Omega Spectrogram

        axSpectrogram.xaxis.set_ticks_position('bottom')

        myfontsize = 15
        myColor = 'k'
        mylabelfontsize = 20

        axSpectrogram.set_xlabel("Time (s)", fontsize=mylabelfontsize, color=myColor)
        axSpectrogram.set_ylabel("Frequency (Hz)", fontsize=mylabelfontsize, color=myColor)
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
        else:
            ValueError("Time outside science or engineering run or more likely code not updated to reflect new science run")

        axSpectrogram.set_title(title,fontsize=mylabelfontsize, color=myColor)

        xmin = min(time)
        xmax = max(time)
        dur = xmax-xmin
        xticks = np.linspace(xmin,xmax,5)
        xticklabels = []
        for i in xticks:
            xticklabels.append(str(i))

        ymin = min(freq)
        ymax = max(freq)

        #myvmax = np.max(Energ)
        myvmax = np.max(plotNormalizedERange)
        cmap = cm.get_cmap(name='viridis')
        myInterp='bicubic'

        cax = axSpectrogram.matshow(Energ, cmap=cmap, \
                           interpolation=myInterp,aspect='auto', origin='lower', \
                           vmin=0.0,extent=[xmin,xmax,ymin,ymax],vmax=myvmax)

        axSpectrogram.set_yscale('log', basey=2, subsy=None)
        ax = plt.gca().yaxis
        ax.set_major_formatter(ScalarFormatter())
        axSpectrogram.ticklabel_format(axis='y', style='plain')

        axSpectrogram.set_xticks(xticks)
        axSpectrogram.set_xticklabels(xticklabels,fontsize=myfontsize)

        # Create Colorbar
        # Make an axis: [left, bottom, width, height], plotting area from 0 to 1
        #cbaxes = fig.add_axes([0.905,0.15,0.04,0.75])
        divider = make_axes_locatable(fig.gca())
        cbaxes = divider.append_axes("right", "5%", pad="3%")
        colorbarticks=range(0,30,5)
        colorbarticklabels=["0","5","10","15","20","25"]
        colorbarlabel = 'Normalized energy'

        cbar   = fig.colorbar(cax, ticks=colorbarticks,cax=cbaxes)
        cbar.set_label(label=colorbarlabel, size=myfontsize, color=myColor)
        cbaxes.set_yticklabels(colorbarticklabels,verticalalignment='center'\
                               , color=myColor)
        axSpectrogram.xaxis.set_ticks_position('bottom')

        fig.savefig(outDir + detectorName + '_' + IDstring + '_spectrogram_' + str(dur) +'.png')

    f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4,sharey=True,figsize=(27,6))

    for iN in np.arange(0,len(timeRange)):
        timestr = 'time' + str(iN)
        time = times[timestr]
        freq = frequencies
        Energ = normalizedEnergies[timestr]

        # Make Omega Spectrogram

        locals()["ax"+str(iN+1)].xaxis.set_ticks_position('bottom')

        myfontsize = 15
        myColor = 'k'
        mylabelfontsize = 20

        if iN == 0:
            locals()["ax"+str(iN+1)].set_ylabel("Frequency (Hz)", fontsize=mylabelfontsize, color=myColor)
        locals()["ax"+str(iN+1)].set_xlabel("Time (s)", fontsize=mylabelfontsize, color=myColor)
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
        else:
            ValueError("Time outside science or engineering run or more likely code not updated to reflect new science run")

        xmin = min(time)
        xmax = max(time)
        dur = xmax-xmin
        xticks = np.linspace(xmin,xmax,5)
        xticklabels = []
        for i in xticks:
            xticklabels.append(str(i))

        ymin = min(freq)
        ymax = max(freq)

        #myvmax = np.max(Energ)
        myvmax = np.max(plotNormalizedERange)
        cmap = cm.get_cmap(name='viridis')
        myInterp='bicubic'

        cax = locals()["ax"+str(iN+1)].matshow(Energ, cmap=cmap, \
                           interpolation=myInterp,aspect='auto', origin='lower', \
                           vmin=0.0,extent=[xmin,xmax,ymin,ymax],vmax=myvmax)

        locals()["ax"+str(iN+1)].set_yscale('log', basey=2, subsy=None)
        locals()["ax"+str(iN+1)].axes.get_yaxis().set_major_formatter(ScalarFormatter())
        locals()["ax"+str(iN+1)].ticklabel_format(axis='y', style='plain')

        locals()["ax"+str(iN+1)].set_xticks(xticks)
        locals()["ax"+str(iN+1)].set_xticklabels(xticklabels,fontsize=myfontsize)

        # Create Colorbar
        locals()["ax"+str(iN+1)].xaxis.set_ticks_position('bottom')

    # Make an axis: [left, bottom, width, height], plotting area from 0 to 1
    cbaxes = f.add_axes([0.925, 0.1, 0.02, 0.8])
    colorbarticks=range(0,30,5)
    colorbarticklabels=["0","5","10","15","20","25"]
    colorbarlabel = 'Normalized energy'

    cbar   = f.colorbar(cax, ticks=colorbarticks,cax=cbaxes)
    cbaxes.set_yticklabels(colorbarticklabels,verticalalignment='center'\
                           , color=myColor)

    f.suptitle(title,fontsize=mylabelfontsize, color=myColor,x=0.51)
    f.savefig(outDir + IDstring + '.png',bbox_inches='tight')


def weventgram(events, tiling, startTime, referenceTime,
                  timeRanges, frequencyRange, durationInflation,
                  bandwidthInflation, plotNormalizedERange,IDstring,detectorName,outDir):

    """WEVENTGRAM Display statistically significant time-frequency events

    WEVENTGRAM Displays filled boxes corresponding to the time-frequency boundary
    of statistically significant events.  WEVENTGRAM takes as input a cell array
    of event matrices, one per channel. A separate figure is produced for each
    channel.  The resulting spectrograms are logarithmic in frequency and linear
    in time, with the tile color denoting normalized energy of tiles or clusters.

    usage:

       handles = weventgram(events, tiling, startTime, referenceTime, ...
                        timeRanges, frequencyRange, ...
                        durationInflation, bandwidthInflation, ...
                        plotNormalizedERange)

         events                  cell array of time-frequency event matrices
         tiling                  q transform tiling structure
         startTime               start time of transformed data
         referenceTime           reference time of plot
         timeRanges               vector range of relative times to plot
         frequencyRange          vector range of frequencies to plot
         durationInflation       multiplicative factor for tile durations
         bandwidthInflation      multiplicative factor for tile bandwidths
         plotNormalizedERange   vector range of normalized energies for colormap

         handles                 vector of axis handles for each eventgram

    WEVENTGRAM expects a cell array of Q transform event structures with one cell
    per channel.  The event structures contain the following fields, which
    describe the properties of statistically significant tiles.

       time                 center time of tile [gps seconds]
       frequency            center frequency of tile [Hz]
       q                    quality factor of tile []
       normalizedEnergy     normalized energy of tile []
       amplitude            signal amplitude of tile [Hz^-1/2]
       phase                phase of tile [radians]

    If the following optional field is present, the color of tiles in the event
    gram correspond to total cluster normalized energy rather than single tile
    normalized energy.

       clusterNormalizedEnergy

    The user can focus on a subset of the times and frequencies available in the
    original transform data by specifying a desired time and frequency range.
    Ranges should be specified as a two component vector, consisting of the
    minimum and maximum value.  By default, the full time and frequency range
    specified in the tiling is displayed.  The default values can be obtained for
     any argument by passing the empty matrix [].

    To determine the range of times to plot, WEVENTGRAM also requires a reference
    time in addition to the specified time range.  This reference time should be
    specified as an absolute quantity, while the range of times to plot should be
    specified relative to the requested reference time.  The specified reference
    time is used as the time origin in the resulting eventgrams and is also
    reported in the title of each plot.  A reference time of zero is assumed by
    default.

    If only one channel is requested, its eventgram is plotted in the current
    figure window.  If more than one eventgram is requested, they are plotted in
    separate figure windows starting with figure 1.

    The optional durationInflation and bandwidthInflation arguments are
    multiplicative scale factors that are applied to the duration and bandwidth of
    displayed events.  If not specified, these parameters both default to unity
    such that the resulting events have unity time-frequency area.

    The optional plotNormalizedERange argument specifies the range of values to
    encode using the colormap.  By default, the lower bound is zero and the upper
    bound is autoscaled to the maximum normalized energy encountered in the
    specified range of time and frequency.

    The optional cell array of channel names are used to label the resulting
    figures.

    WEVENTGRAM returns a vector of axis handles to the eventgram produced for each
    channel.

    See also WTILE, WCONDITION, WTRANSFORM, WTHRESHOLD, WSELECT, WCLUSTER,
    WSPECTROGRAM, and WEXAMPLE.

    Shourov K. Chatterji <shourov@ligo.mit.edu>
    Jameson Rollins <jrollins@phys.columbia.edu>

    $Id: weventgram.m 3415 2012-06-02 17:39:47Z michal.was@LIGO.ORG $
    """
    # determine number of channels
    numberOfChannels = 1
    # begin loop over channels
    for channel in np.arange(0,numberOfChannels):
        channelstr = 'channel' + str(channel)

        ######################################################################
        #                 extract event properties                           #
        ######################################################################

        # bandwidth of events
        bandwidths = 2 * np.sqrt(np.pi) * events[channelstr]['frequency'] / \
              events[channelstr]['q']

        # duration of events
        durations = 1 / bandwidths

        # apply bandwidth inflation factor
        bandwidths = bandwidthInflation * bandwidths

        # apply duration inflation factor
        durations = durationInflation * durations

        # start time of events
        startTimes = events[channelstr]['time'] - referenceTime - durations / 2

        # stop time of events
        stopTimes = events[channelstr]['time'] - referenceTime + durations / 2

        # low frequency of events
        lowFrequencies = events[channelstr]['frequency'] - bandwidths / 2

        # high frequency boundary of events
        highFrequencies = events[channelstr]['frequency'] + bandwidths / 2

        # normalized energy of events or clusters
        #if isfield(events[channelstr]['clusterNormalizedEnergy']):
        #    normalizedEnergies = events[channelstr]['clusterNormalizedEnergy']
        #else:
        #    normalizedEnergies = events[channelstr]['normalizedEnergy']
        normalizedEnergies = events[channelstr]['normalizedEnergy']

        for timeRange in timeRanges:
            ##################################################################
            #            identify events to display                          #
            ##################################################################
            times = np.array([-1,1])*timeRange/2
            # default start time for
            if times[0] == float('-Inf'):
                if not tiling:
                    times[0] = np.floor(np.min(startTimes))
                else:
                    times[0] = startTime - referenceTime

            # default stop time
            if times[1] == float('Inf'):
                if not tiling:
                    times[1] = np.ceil(np.max(stopTimes))
                else:
                    times[1] = startTime - referenceTime + \
                                          tiling['generalparams']['duration']

            # default minimum frequency
            if frequencyRange[0] == float('-Inf'):
                if not tiling:
                    frequencyRange[0] = 2**(np.floor(np.log2(np.min(lowFrequencies))))
                else:
                    frequencyRange[0] = tiling['plane0.0']['minimumFrequency']

            # default maximum frequency
            if frequencyRange[1] == float('Inf'):
                if isempty(tiling):
                    frequencyRange[1] = 2 ** np.ceil(np.log2(np.max(highFrequencies)))
                else:
                    lastPlane = tiling['generalparams']['numberOfPlanes'] -1
                    frequencyRange[1] = tiling['plane' + str(lastPlane)]['maximumFrequency']

            # find events overlapping specified time-frequency ranges
            displayIndices = \
                        (startTimes < np.max(times)) &\
                        (stopTimes > np.min(times)) &\
                        (lowFrequencies < np.max(frequencyRange)) &\
                        (highFrequencies > np.min(frequencyRange))

            # number of events
            numberOfEvents = len(displayIndices)

            ##################################################################
            #            sort by normalized energy                           #
            ##################################################################

            # sort in order of increasing normalized energy
            sortedIndices = np.argsort(normalizedEnergies[displayIndices])

            # sorted properties of events to display
            startTimes = startTimes[displayIndices][sortedIndices]
            stopTimes = stopTimes[displayIndices][sortedIndices]
            lowFrequencies = lowFrequencies[displayIndices][sortedIndices]
            highFrequencies = highFrequencies[displayIndices][sortedIndices]
            normalizedEnergies = normalizedEnergies[displayIndices][sortedIndices]

            ##################################################################
            #             define event boundaries                            #
            ##################################################################
            fig, ax = plt.subplots()

            # Set ticks and labels on eventgram
            myfontsize = 15
            myColor = 'k'
            mylabelfontsize = 20

            ax.set_xlabel("Time (s)", fontsize=mylabelfontsize, color=myColor)
            ax.set_ylabel("Frequency (Hz)", fontsize=mylabelfontsize, color=myColor)
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
            else:
                ValueError("Time outside science or engineering run or more likely code not updated to reflect new science run")

            ax.set_title(title,fontsize=mylabelfontsize, color=myColor)

            # Make best fit tile bounding boxes and fill them in
            patches = []
            colors = []
            for iN in np.arange(0,len(startTimes)):

                # time coordinates of event bounding box
                # frequency coordinates of event bounding box
                boundary =  np.array([[startTimes[iN],stopTimes[iN],stopTimes[iN],startTimes[iN],startTimes[iN]],[lowFrequencies[iN],lowFrequencies[iN],highFrequencies[iN],highFrequencies[iN],lowFrequencies[iN]]])
                polygon = Polygon(boundary.T, True)

                patches.append(polygon)
                colors.append(normalizedEnergies[iN])

            p = PatchCollection(patches,cmap=cm.viridis)
            p.set_array(np.asarray(colors))

            ax.add_collection(p)

            # Set colormap
            p.set_clim(plotNormalizedERange)
            colorbarticks=range(0,30,5)
            cbar = fig.colorbar(p,ticks=colorbarticks)
            colorbarticklabels=["0","5","10","15","20","25"]
            colorbarlabel = 'Normalized energy'
            cbar.set_label(label=colorbarlabel, size=myfontsize, color=myColor)
            cbar.ax.set_yticklabels(colorbarticklabels,
                         verticalalignment='center', color=myColor)


            # Set x and y axis limits on event grams (log2 scale for y)
            xmin = min(times)
            xmax = max(times)
            ymin = np.min(frequencyRange)
            ymax = np.max(frequencyRange)
            dur = xmax-xmin
            plt.axis([xmin,xmax,ymin,ymax])
            xticks = np.linspace(xmin,xmax,5)
            xticklabels = []
            for i in xticks:
                xticklabels.append(str(i))
            ax.set_xticks(xticks)
            ax.set_xticklabels(xticklabels,fontsize=myfontsize)

            ax.set_yscale('log', basey=2, subsy=None)
            ax = plt.gca().yaxis
            ax.set_major_formatter(ScalarFormatter())

            fig.savefig(outDir + detectorName + '_' + IDstring + '_eventgram_' + str(dur) +'.png')
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

    if opts.condor:
        write_subfile()
        write_dagfile()
        sys.exit()

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

    # Plot Time Series at given plot durations
    if opts.plot_raw_timeseries:
        plot = data.plot()
        plot.set_title('TimeSeries')
        plot.set_ylabel('Gravitational-wave strain amplitude')
        for iTime in np.arange(0,len(plotTimeRanges)):
            halfTimeRange = plotTimeRanges[iTime]*0.5
            plot.set_xlim(opts.eventTime - halfTimeRange,opts.eventTime + halfTimeRange)
            plot.save(outDir + detectorName + '_' + IDstring + '_timeseries_' + str(plotTimeRanges[iTime]) + '.png')

    # resample data
    if data.sample_rate.decompose().value != sampleFrequency:
        data = data.resample(sampleFrequency)

    # generate search tiling
    highPassCutoff = []
    lowPassCutoff = []
    whiteningDuration = []
    tiling = wtile(blockTime, searchQRange, searchFrequencyRange, sampleFrequency, \
                 searchMaximumEnergyLoss, highPassCutoff, lowPassCutoff, \
                 whiteningDuration, transientFactor)

    # high pass filter and whiten data
    #data,lpefOrder = highpassfilt(data,tiling)

    # Plot HighPass Filtered Time  Series at given plot durations
    if opts.plot_highpassfiltered_timeseries:
        plot = data.plot()
        plot.set_title('HighPassFilter')
        plot.set_ylabel('Gravitational-wave strain amplitude')
        for iTime in np.arange(0,len(plotTimeRanges)):
            halfTimeRange = plotTimeRanges[iTime]*0.5
            plot.set_xlim(opts.eventTime - halfTimeRange,opts.eventTime + halfTimeRange)
            plot.save(outDir + detectorName + '_' + IDstring + '_highpassfiltered_' + str(plotTimeRanges[iTime]) + '.png')

    # Time to whiten the times series data
    # Our FFTlength is determined by the predetermined whitening duration found in wtile
    FFTlength = tiling['generalparams']['whiteningDuration']

    # We will condition our data using the median-mean approach. This means we will take two sets of non overlaping data from our time series and find the median S for both and then average them in order to be both safe against outliers and less bias. By default this ASD is one-sided.
    '''what is S? ASD?'''

    asd = data.asd(FFTlength, FFTlength/2., method='median-mean')
    #asd = asd *tiling['generalparams']['sampleFrequency']*data.size
    # Apply ASD to the data to whiten it
    white_data = data.whiten(FFTlength, FFTlength/2., asd=asd)

    # Plot whitened Time  Series at given plot durations
    if opts.plot_whitened_timeseries:
        plot = white_data.plot()
        plot.set_title('Whitened')
        plot.set_ylabel('Gravitational-wave strain amplitude')
        for iTime in np.arange(0,len(plotTimeRanges)):
            halfTimeRange = plotTimeRanges[iTime]*0.5
            plot.set_xlim(opts.eventTime - halfTimeRange,opts.eventTime + halfTimeRange)
            plot.save(outDir + detectorName + '_' + IDstring + '_whitened_' + str(plotTimeRanges[iTime]) + '.png')

    # Extract one-sided frequency-domain conditioned data.
    dataLength = tiling['generalparams']['sampleFrequency'] * tiling['generalparams']['duration']
    halfDataLength = int(dataLength / 2 + 1)
    white_data_fft = white_data.fft()

    # q transform whitened data
    coefficients = []
    coordinate = [np.pi/2,0]
    whitenedTransform = \
      wtransform(white_data_fft, tiling, outlierFactor, 'independent', channelName,coefficients, coordinate)

    # identify most significant whitened transform tile
    thresholdReferenceTime = centerTime
    thresholdTimeRange = 0.5 * searchWindowDuration * np.array([-1,1])
    thresholdFrequencyRange = []
    thresholdQRange = []
    whitenedProperties = \
      wmeasure(whitenedTransform, tiling, startTime, thresholdReferenceTime, \
               thresholdTimeRange, thresholdFrequencyRange, thresholdQRange)

    # Two properties of particular interest are peakNormalizedEnergy value
    # and the peak Q plane.

    # Select most siginficant Q plane
    mostSignificantQ = \
          whitenedProperties['channel0']['peakQ']
    loudestEnergy = whitenedProperties['channel0']['peakNormalizedEnergy']

    # Based on your white noise false alarm value determine if tiling
    # produced a significant event for this channel.
    normalizedEnergyThreshold = -np.log(whiteNoiseFalseRate * \
                                     tiling['generalparams']['duration'] / \
                                     (1.5 * tiling["generalparams"]["numberOfIndependents"]))

    if not alwaysPlotFlag:
        if loudestEnergy < normalizedEnergyThreshold:
            raise ValueError('This channel does not have a significant tiling at\
                        white noise false alarm rate provided')

    ############################################################################
    #                      plot whitened spectrogram                           #
    ############################################################################

    # plot whitened spectrogram
    wspectrogram(whitenedTransform, tiling, outDir,IDstring,startTime, centerTime, \
                 plotTimeRanges, plotFrequencyRange, \
                 mostSignificantQ, plotNormalizedERange, \
                 plotHorizontalResolution,detectorName)
    print('wscan finished: {0}'.format(time.time()-start))

    if opts.plot_eventgram:
        # identify significant whitened q transform tiles
        thresholdReferenceTime = centerTime
        # identify the lastPlane
        lastPlane = tiling['generalparams']['numberOfPlanes'] -1

        thresholdTimeRange = 0.5 * np.array([-1,1]) * \
               (max(plotTimeRanges) + \
               tiling['plane' + str(lastPlane)]['row0.0']\
               ['duration'] * plotDurationInflation)

        thresholdFrequencyRange = plotFrequencyRange
        thresholdQRange = None

        whitenedSignificants = \
          wthreshold(whitenedTransform, tiling, startTime, whiteNoiseFalseRate, \
               thresholdReferenceTime, thresholdTimeRange, \
               thresholdFrequencyRange, thresholdQRange, \
               maximumSignificants, None, None, None, None,None)

        # select non-overlapping whitened significant tiles
        whitenedSignificants = wselect(whitenedSignificants, \
               plotDurationInflation, plotBandwidthInflation, \
               maximumMosaics)

        ########################################################################
        #                  plot whitened eventgram                             #
        ########################################################################

        # Plot whitened eventgram
        weventgram(whitenedSignificants, tiling, startTime, centerTime, \
                 plotTimeRanges, plotFrequencyRange, \
                 plotDurationInflation, plotBandwidthInflation, \
                 plotNormalizedERange,IDstring,detectorName,outDir)

    if opts.runML:
        B1 = 1610
        B2 = 1934
        B3 = 1935
        A  = 2360
        M  = 2117
        if detectorName == 'H1':
            workflow_subject_set_dict_app = {
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
        classes = ["Air_Compressor","Blip","Chirp","Extremely_Loud","Helix","Koi_Fish","Light_Modulation","Low_Frequency_Burst","Low_Frequency_Lines","None_of_the_Above","No_Glitch","Paired_Doves","Power_Line","Repeating_Blips","Scattered_Light","Scratchy","Tomte","Violin_Mode","Wandering_Line","Whistle","uniqueID","Label","workflow","subjectset","Filename1","Filename2","Filename3","Filename4","UploadFlag"]
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
        scoresTable.to_hdf('{0}/ML_GSpy_{1}.h5'.format(opts.outDir,opts.ID),'gspy_ML_classification')

        system_call = "mv {0}*.png {1}".format(outDir,finalPath)
        os.system(system_call)
        shutil.rmtree(outDir.replace(lastPath,"",1))
        print('ML finished: {0}'.format(time.time()-start))

if __name__ == '__main__':
    main()
