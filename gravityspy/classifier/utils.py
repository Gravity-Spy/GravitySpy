#!/usr/bin/env python

from gwtrigfind import find_trigger_files
from gwpy.segments import DataQualityFlag
from gwpy.table import EventTable

from gravityspy.utils import log
import numpy
  
# ---- Import standard modules to the python path.

def get_triggers(start, end, channel, dqflag, verbose=True, **kwargs):
    """Obtain omicron triggers run gravityspy on

    Parameters:
    ----------

    Returns
    -------
    """
    duration_max = kwargs.pop('duration_max', None)
    duration_min = kwargs.pop('duration_min', None)
    frequency_max = kwargs.pop('frequency_max', None)
    frequency_min = kwargs.pop('frequency_min', None)
    snr_max = kwargs.pop('snr_max', None)
    snr_min = kwargs.pop('snr_min', None)

    detector = channel.split(':')[0]

    logger = log.Logger('Gravity Spy: Fetching Omicron Triggers')

    # Obtain segments that are analysis ready
    analysis_ready = DataQualityFlag.query('{0}:{1}'.format(detector, dqflag),\
                                          float(start), float(end))

    # Display segments for which this flag is true
    logger.info("Segments for which the {0} Flag "
                "is active: {1}".format(dqflag, analysis_ready.active))

    # get Omicron triggers
    files = find_trigger_files(channel,'Omicron',
                               float(start),float(end))

    omicrontriggers = EventTable.read(files, tablename='sngl_burst', format='ligolw')

    # filter table
    # Create mask assuming everything passes
    masks = np.ones(len(omicrontriggers),dtype=bool)

    if not duration_max is None:
        masks &= (omicrontriggers['duration'] <= duration_max)
    if not duration_min is None:
        masks &= (omicrontriggers['duration'] >= duration_min)
    if not frequency_max is None:
        masks &= (omicrontriggers['peak_frequency'] <= frequency_max)
    if not frequency_min is None:
        masks &= (omicrontriggers['peak_frequency'] >= frequency_min)
    if not snr_max is None:
        masks &= (omicrontriggers['snr'] <= snr_max)
    if not snr_min is None:
        masks &= (omicrontriggers['snr'] >= snr_min)

    omicrontriggers = omicrontriggers[masks]
    # Set peakGPS
    omicrontriggers['peakGPS'] = omicrontriggers['peak_time'] + (0.000000001)*omicrontriggers['peak_time_ns']

    logger.info("List of available metadata information for a given glitch provided by omicron: {0}".format(omicrontriggers.keys()))

    logger.info("Number of triggers after SNR and Freq cuts but before ANALYSIS READY flag filtering: {0}".format(len(omicrontriggers)))

    # Filter the raw omicron triggers against the ANALYSIS READY flag.
    vetoed = omicrontriggers['peakGPS'].in_segmentlist(analysis_ready.active)
    omicrontriggers = omicrontriggers[vetoed]

    logger.info("Final trigger length: {0}".format(len(omicrontriggers)))

    return omicrontriggers


def get_triggers_from_cache(filename, **kwargs):
    """Obtain omicron triggers run gravityspy on

    Parameters:
    ----------

    Returns
    -------
    """
    omicrontriggers = EventTable.read(omicron_trigger_file,
                                      tablename='sngl_burst', format='ligolw')
    return
