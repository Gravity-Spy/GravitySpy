#!/usr/bin/env python

import optparse,os,string,random,pdb
from gwpy.table.lsctables import SnglBurstTable
from gwpy.segments import DataQualityFlag
import pandas as pd

###############################################################################
##########################                             ########################
##########################   Func: parse_commandline   ########################
##########################                             ########################
###############################################################################
# Definite Command line arguments here

def parse_commandline():
    """
    Parse the options given on the command-line.
    """
    parser = optparse.OptionParser()
    parser.add_option("--applyallDQ", action="store_true", default=False,help="Generate a list of omicron triggers that exclude all of the DQ cuts already created by detchar. (Default:False)")
    parser.add_option("--channelname", help="channel name [Default:GDS-CALIB_STRAIN]",
                        default="GDS-CALIB_STRAIN")
    parser.add_option("--detector", help="detector name [L1 or H1]. [No Default must supply]")
    parser.add_option("--durHigh", help="The max duration of a glitch you want to make images for [Optional Input]",type=float)
    parser.add_option("--durLow", help="lower duration of a glitch you want to make images for[Optional Input]",type=float)
    parser.add_option("--gpsStart", type=int,help="gps Start Time of Query for meta data and omega scans. If not supplied will attempt to imply from datatable",default=0)
    parser.add_option("--gpsEnd", type=int,help="gps End Time If not supplied will attempt to imply from datatable",default=0)
    parser.add_option("--freqHigh", help="upper frequency bound cut off for the glitches. [Default: 2048]",type=int,default=2048)
    parser.add_option("--freqLow", help="lower frequency bound cut off for the glitches. [Default: 10]",type=int,default=10)
    parser.add_option("--maxSNR", help="This flag gives you the option to supply a upper limit to the SNR of the glitches [Optional Input]",type=float)
    parser.add_option("--outDir", help="Outdir of omega scan and omega scan webpage (i.e. your html directory)")
    parser.add_option("--pathToExec", help="Path to version of wscan.py you want to use")
    parser.add_option("--pathToIni", help="Path to ini file")
    parser.add_option("--pathToModel",default='./ML/trained_model/' help="Path to trained model")
    parser.add_option("--SNR", help="Lower bound SNR Threshold for omicron triggers, by default there is no upperbound SNR unless supplied throught the --maxSNR flag. [Default: 6]",type=float,default=6)
    parser.add_option("--uniqueID", action="store_true", default=False,help="Is this image being generated for the GravitySpy project, is so we will create a uniqueID strong to use for labeling images instead of GPS time")

    opts, args = parser.parse_args()

    return opts

###############################################################################
##########################                             ########################
##########################   Func: id_generator        ########################
##########################                             ########################
###############################################################################

def id_generator(x,size=10, chars=string.ascii_uppercase + string.digits +string.ascii_lowercase):
    return ''.join(random.SystemRandom().choice(chars) for _ in range(size))

###############################################################################
##########################                              #######################
##########################   Func: threshold            #######################
##########################                              #######################
###############################################################################
# Define threshold function that will filter omicron triggers based on SNR and frequency threshold the user sets (or the defaults.)

def threshold(row):
    passthresh = True
    if not opts.durHigh is None:
	passthresh = ((row.duration <= opts.durHigh) and passthresh)
    if not opts.durLow is None:
        passthresh = ((row.duration >= opts.durLow) and passthresh)
    if not opts.freqHigh is None:
        passthresh = ((row.peak_frequency <= opts.freqHigh) and passthresh)
    if not opts.freqLow is None:
        passthresh = ((row.peak_frequency >= opts.freqLow) and passthresh)
    if not opts.maxSNR is None:
        passthresh = ((row.snr <= opts.maxSNR) and passthresh)
    if not opts.SNR is None:
        passthresh = ((row.snr >= opts.SNR) and passthresh)
    return passthresh


###############################################################################
##########################                          ###########################
##########################   Func: get_triggers     ###########################
##########################                          ###########################
###############################################################################

# This function queries omicron to obtain trigger times of the glitches. It then proceeds to filter out these times based on whether the detector was in an analysis ready state or the glitch passes certain frequency and SNR cuts. 

def get_triggers(gpsStart,gpsEnd):

    # Obtain segments that are analysis ready
    analysis_ready = DataQualityFlag.query('{0}:DMT-ANALYSIS_READY:1'.format(opts.detector),float(gpsStart),float(gpsEnd))

    # Display segments for which this flag is true
    print "Segments for which the ANALYSIS READY Flag is active: {0}".format(analysis_ready.active)

    omicrontriggers = SnglBurstTable.fetch(detchannelname,'Omicron',\
    float(gpsStart),float(gpsEnd),filt=threshold)

    print "List of available metadata information for a given glitch provided by omicron: {0}".format(omicrontriggers.columnnames)

    print "Number of triggers after SNR and Freq cuts but before ANALYSIS READY flag filtering: {0}".format(len(omicrontriggers))

    # Filter the raw omicron triggers against the ANALYSIS READY flag.
    omicrontriggers = omicrontriggers.vetoed(analysis_ready.active)

    print "Final trigger length: {0}".format(len(omicrontriggers))

    return omicrontriggers

###############################################################################
##########################                            #########################
##########################   Func: write_dagfile     #########################
##########################                            #########################
###############################################################################

# Write dag_file file for the condor job

def write_dagfile(x):
    with open('gravityspy_{0}_{1}.dag'.format(oTriggers.peak_time.min(),oTriggers.peak_time.max()),'a+') as dagfile:
        dagfile.write('JOB {0} ./condor/gravityspy.sub\n'.format(x.peak_time))
        dagfile.write('RETRY {0} 3\n'.format(x.peak_time))
        dagfile.write('VARS {0} jobNumber="{0}" eventTime="{1}" ID="{2}"'.format(x.peak_time,x.peakGPS,x.uniqueID))
        dagfile.write('\n\n')


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
        subfile.write('executable = {0}\n'.format(opts.pathToExec))
        subfile.write('\n')
        subfile.write('arguments = "--inifile {0} --eventTime $(eventTime) --outDir {1} --pathToModel {2} --uniqueID --ID $(ID) --runML"\n'.format(opts.pathToIni,opts.outDir,opts.pathToModel))
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

############################################################################
###############          MAIN        #######################################
############################################################################

# Parse commandline arguments
opts = parse_commandline()

# Determine start and stop time of trigger query.
if not opts.gpsStart:
    tmp = pd.read_hdf('triggers.h5')
    gpsStart = tmp.peak_time.max()
else:
    gpsStart = opts.gpsStart

if not opts.gpsEnd:
    gpsEnd = gpsStart + 86400
else:
    gpsEnd = opts.gpsEnd

# Take the detector and the channel from the command line and combine them into one string. This is needed for some input later on.
detchannelname = opts.detector + ':' + opts.channelname

write_subfile()
omicrontriggers = get_triggers(gpsStart,gpsEnd)
oTriggers = pd.DataFrame(omicrontriggers.to_recarray(),omicrontriggers.get_peak()).reset_index()
oTriggers.rename(columns = {'index':'peakGPS'},inplace=True)
oTriggers['uniqueID'] = oTriggers.peakGPS.apply(id_generator)
oTriggers[['peak_time','peakGPS','uniqueID']].apply(write_dagfile,axis=1)
oTriggers.peakGPS = oTriggers.peakGPS.apply(float)
oTriggers.to_hdf('triggers.h5','gspy_triggers',append=True)
