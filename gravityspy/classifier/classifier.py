def main(inifile, eventTime, project_info_pickle, ID, outDir,
         path_to_cnn='/home/machinelearning/GravitySpy/gravityspy/ML/trained_model/multi_view_classifier.h5',
         path_to_similarity_search='/home/machinelearning/GravitySpy/gravityspy/ML/trained_model/semantic_idx_model.h5',
         uniqueID=True, runML=False, HDF5=False, PostgreSQL=False,
         verbose=False):

    if (runML == True) and (HDF5==False) and (PostgreSQL==False):
        raise ValueError('If you wish to run ML must select file format to '
                         'save with. Most cases HDF5=True is what you want')

    if not os.path.isfile(path_to_cnn):
        raise ValueError('The provided CNN model does not '
                         'exist.')
    if not os.path.isfile(path_to_similarity_search):
        raise ValueError('The provided similarity model does not '
                         'exist.')

    logger = log.Logger('Gravity Spy: OmegaScan')

    ###########################################################################
    #                                   Parse Ini File                        #
    ###########################################################################

    # ---- Create configuration-file-parser object and read parameters file.
    cp = ConfigParser()
    cp.read(inifile)
    logger.info('You have chosen the following ini file: {0}'.format(inifile))

    # ---- Read needed variables from [parameters] and [channels] sections.
    alwaysPlotFlag = cp.getint('parameters', 'alwaysPlotFlag')
    sampleFrequency = cp.getint('parameters', 'sampleFrequency')
    blockTime = cp.getint('parameters', 'blockTime')
    searchFrequencyRange = json.loads(cp.get('parameters',
                                             'searchFrequencyRange'))
    searchQRange = json.loads(cp.get('parameters', 'searchQRange'))
    searchMaximumEnergyLoss = cp.getfloat('parameters',
                                          'searchMaximumEnergyLoss')
    searchWindowDuration = cp.getfloat('parameters', 'searchWindowDuration')
    whiteNoiseFalseRate = cp.getfloat('parameters', 'whiteNoiseFalseRate')
    plotTimeRanges = json.loads(cp.get('parameters', 'plotTimeRanges'))
    plotFrequencyRange = json.loads(cp.get('parameters', 'plotFrequencyRange'))
    plotNormalizedERange = json.loads(cp.get('parameters',
                                             'plotNormalizedERange'))
    frameCacheFile = cp.get('channels', 'frameCacheFile')
    frameType = cp.get('channels', 'frameType')
    channelName = cp.get('channels', 'channelName')
    detectorName = channelName.split(':')[0]
    det = detectorName.split('1')[0]

    logger.info('You have chosen the following Q range: '
                '{0}'.format(searchQRange))
    logger.info('You have chosen the following search range: '
                '{0}'.format(searchFrequencyRange))
