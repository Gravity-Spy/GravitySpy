#!/usr/bin/env python

import optparse,os,string,random,pdb
from gwpy.table.lsctables import SnglBurstTable
from gwpy.segments import DataQualityFlag

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

def write_dagfile(eventTime,ID,i):
    with open('gravityspy_{0}_{1}.dag'.format(opts.gpsStart,opts.gpsEnd),'a+') as dagfile:
        dagfile.write('JOB {0} ./condor/gravityspy.sub\n'.format(i))
        dagfile.write('RETRY {0} 3\n'.format(i))
        dagfile.write('VARS {0} jobNumber="{0}" eventTime="{1}" ID="{2}"'.format(i,eventTime,ID))
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
        subfile.write('arguments = "--inifile wini.ini --eventTime $(eventTime) --outDir {0} --uniqueID --ID $(ID) --runML"\n'.format(opts.outDir))
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
omicrontriggers = get_triggers()

d = 'metadata'
if not os.path.isdir(d):
    os.makedirs(d)

i = 0

for omicrontrigger in omicrontriggers:
    if opts.uniqueID:
        ID = id_generator()
        i = i +1
        write_dagfile(omicrontrigger.get_peak(),ID,i)
        with open('./metadata/metadata_{0}_{1}_{2}.txt'.format(opts.detector,opts.gpsStart,opts.gpsEnd),'a+') as f:
            f.write('{0} {1} {2} {3} {4} {5} {6} {7} {8} {9} {10}:{11}\n'.format(omicrontrigger.snr,omicrontrigger.amplitude,omicrontrigger.peak_frequency,omicrontrigger.central_freq,omicrontrigger.duration,omicrontrigger.bandwidth,omicrontrigger.chisq,omicrontrigger.chisq_dof,omicrontrigger.get_peak(),ID,omicrontrigger.ifo,omicrontrigger.channel))
            f.close()

# need to save omicron triggers variable to be loaded 
# by the condor job later
filename = open("metadata/{0}_{1}_{2}.xml".format(opts.detector,opts.gpsStart,opts.gpsEnd),'w')
omicrontriggers.write(fileobj=filename)

summaryanduploadPage = open('summary_upload.sh','w')
summaryanduploadPage.write('#!/bin/bash\n')
summaryanduploadPage.write('\n')
summaryanduploadPage.write('cat {0}/**/scores.csv > {0}/allscores.csv\n'.format(opts.outDir))
summaryanduploadPage.write('\n')
summaryanduploadPage.write('cat {0}/metadata/metadata_{1}_*.txt > {0}/metadata/metadata_{1}.txt\n'.format(os.getcwd(),opts.detector))
summaryanduploadPage.write('\n')
summaryanduploadPage.write('python {0}/makewebpage.py --outPath {1} --dataPath {2} --metadata {0}/metadata/metadata_{3}.txt --ML\n'.format(os.getcwd(),'/'.join(opts.outDir.split('/')[0:-3]),opts.outDir,opts.detector)) 
summaryanduploadPage.write('\n')
summaryanduploadPage.write('python {0}/sortimages.py --dataPath {1} --detector {2}'.format(os.getcwd(),opts.outDir,opts.detector))
summaryanduploadPage.close()
