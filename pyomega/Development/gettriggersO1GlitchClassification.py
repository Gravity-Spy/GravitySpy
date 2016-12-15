#!/usr/bin/env python

import optparse,os,string,random,pdb
import pandas as pd
from gwpy.table.lsctables import SnglBurstTable
from gwpy.segments import SegmentList, Segment,DataQualityFlag
from gwpy import time

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
    parser.add_option("--gpsStart", type=int,help="gps Start Time of Query for meta data and omega scans. [No Default must supply]")
    parser.add_option("--gpsEnd", type=int,help="gps End Time [No Default must supply]")
    parser.add_option("--freqHigh", help="upper frequency bound cut off for the glitches. [Default: 2048]",type=int,default=2048)
    parser.add_option("--freqLow", help="lower frequency bound cut off for the glitches. [Default: 10]",type=int,default=10)
    parser.add_option("--maxSNR", help="This flag gives you the option to supply a upper limit to the SNR of the glitches [Optional Input]",type=float)
    parser.add_option("--outDir", help="Outdir of omega scan and omega scan webpage (i.e. your html directory)")
    parser.add_option("--SNR", help="Lower bound SNR Threshold for omicron triggers, by default there is no upperbound SNR unless supplied throught the --maxSNR flag. [Default: 6]",type=float,default=6)
    parser.add_option("--triggerFile", help="[GPS Label Pipeline] Trigger List")
    parser.add_option("--uniqueID", action="store_true", default=False,help="Is this image being generated for the GravitySpy project, is so we will create a uniqueID strong to use for labeling images instead of GPS time")

    opts, args = parser.parse_args()

    return opts

###############################################################################
##########################                             ########################
##########################   Func: id_generator        ########################
##########################                             ########################
###############################################################################

def id_generator(size=10, chars=string.ascii_uppercase + string.digits +string.ascii_lowercase):
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

def get_triggers():

    # Obtain segments that are analysis ready
    analysis_ready = DataQualityFlag.query('{0}:DMT-ANALYSIS_READY:1'.format(opts.detector),float(opts.gpsStart),float(opts.gpsEnd))

    # Display segments for which this flag is true
    print "Segments for which the ANALYSIS READY Flag is active: {0}".format(analysis_ready.active)

    omicrontriggers = SnglBurstTable.fetch(detchannelname,'Omicron',\
    float(opts.gpsStart),float(opts.gpsEnd),filt=threshold)

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

def write_dagfile(eventTime,ID,i,label):
    with open('gravityspy_{0}_{1}.dag'.format(opts.gpsStart,opts.gpsEnd),'a+') as dagfile:
        dagfile.write('JOB {0} ./condor/gravityspy.sub\n'.format(i))
        dagfile.write('RETRY {0} 3\n'.format(i))
        dagfile.write('VARS {0} jobNumber="{0}" eventTime="{1}" ID="{2}" Label="{3}"'.format(i,eventTime,ID,label))
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
        subfile.write('executable = {0}/wscan.py\n'.format(os.getcwd()))
        subfile.write('\n')
        subfile.write('arguments = "--inifile wini.ini --eventTime $(eventTime) --outDir {0} --uniqueID --ID $(ID) --label $(Label)"\n'.format(opts.outDir))
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

# Take the detector and the channel from the command line and combine them into one string. This is needed for some input later on.
detchannelname = opts.detector + ':' + opts.channelname

write_subfile()

d = 'metadata'
if not os.path.isdir(d):
    os.makedirs(d)

triggers = pd.read_csv(open('{0}'.format(opts.triggerFile),'r'))

iN = 0
i  = 0
for GPStime in triggers.GPSTime:
    if opts.uniqueID:
        ID = id_generator()
        i = i +1
        write_dagfile(GPStime,ID,i,triggers.Label.iloc[iN])
        with open('./metadata/metadata_{0}.txt'.format(opts.detector),'a+') as f:
            f.write('{0} {1} {2} {3} {4} {5}\n'.format(GPStime,triggers.Pipeline.iloc[iN],triggers.Label.iloc[iN],ID,triggers.peak_frequency.iloc[iN],triggers.snr.iloc[iN]))
            f.close()

        iN = iN +1

# need to save omicron triggers variable to be loaded 
# by the condor job later
