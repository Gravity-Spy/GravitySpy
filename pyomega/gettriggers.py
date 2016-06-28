#!/usr/bin/env python

import optparse,os,string,random
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
    parser.add_option("--gpsStart", help="gps Start Time of Query for meta data and omega scans. [No Default must supply]")
    parser.add_option("--gpsEnd", help="gps End Time [No Default must supply]")
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
    analysis_ready = DataQualityFlag.query('{0}:DMT-ANALYSIS_READY:1'.format(opts.detector),opts.gpsStart,opts.gpsEnd)

    # Display segments for which this flag is true
    print "Segments for which the ANALYSIS READY Flag is active: {0}".format(analysis_ready.active)

    if opts.applyallDQ:
        print("We are finding all previously created DQ cuts")
        # Obtain segments of all DQ cuts if requested DQ list can be found
        # https://code.pycbc.phy.syr.edu/detchar/veto-definitions/blob/master/burst/O1/H1L1-HOFT_C02_O1_BURST.xml
        # First obtain those flags that are for both H1 and L1
        O1_MISSING_HOFT_C02 = DataQualityFlag.query('{0}:DCS-MISSING_{0}_HOFT_C02:1'.format(opts.detector),opts.gpsStart,opts.gpsEnd)
        O1_ETMY_ESD_DAC_OVERFLOW = DataQualityFlag.query('{0}:DMT-ETMY_ESD_DAC_OVERFLOW:1'.format(opts.detector),opts.gpsStart,opts.gpsEnd)
        O1_OMC_DCPD_A_SATURATION = DataQualityFlag.query('{0}:DCH-OMC_DCPD_A_SATURATION:1'.format(opts.detector),opts.gpsStart,opts.gpsEnd)
        O1_OMC_DCPD_ADC_OVERFLOW = DataQualityFlag.query('{0}:DMT-OMC_DCPD_ADC_OVERFLOW:1'.format(opts.detector),opts.gpsStart,opts.gpsEnd)
        O1_ETMY_SATURATION_SNR200 = DataQualityFlag.query('{0}:DCH-ETMY_SATURATION_SNR200:1'.format(opts.detector),opts.gpsStart,opts.gpsEnd)
        O1_CW_INJECTION_TRANSITION = DataQualityFlag.query('{0}:DCH-CW_INJECTION_TRANSITION:1'.format(opts.detector),opts.gpsStart,opts.gpsEnd)
        O1_BAD_KAPPA_BASIC_CUT_HOFT_C02 = DataQualityFlag.query('{0}:DCS-BAD_KAPPA_BASIC_CUT_{0}_HOFT_C02:1'.format(opts.detector),opts.gpsStart,opts.gpsEnd)
        O1_PARTIAL_FRAME_LOSS_HOFT_C02 = DataQualityFlag.query('{0}:DCS-PARTIAL_FRAME_LOSS_{0}_HOFT_C02:1'.format(opts.detector),opts.gpsStart,opts.gpsEnd)

        # Obtain detector specific flags
        if opts.detector == "H1":
            O1_RF45_AM_STABILIZATION = DataQualityFlag.query('{0}:DCH-RF45_AM_STABILIZATION:4'.format(opts.detector),opts.gpsStart,opts.gpsEnd)
            O1_ETMY_SATURATION = DataQualityFlag.query('{0}:DCH-ETMY_SATURATION:2'.format(opts.detector),opts.gpsStart,opts.gpsEnd)
            O1_BAD_DATA_BEFORE_LOCKLOSS = DataQualityFlag.query('{0}:DCH-BAD_DATA_BEFORE_LOCKLOSS:1'.format(opts.detector),opts.gpsStart,opts.gpsEnd)
            O1_ETMY_VIOLIN_MODE_2NDHARMONIC_RINGING = DataQualityFlag.query('{0}:DCH-ETMY_VIOLIN_MODE_2NDHARMONIC_RINGING:1'.format(opts.detector),opts.gpsStart,opts.gpsEnd)
            O1_RF45_SEVERE_GLITCHING = DataQualityFlag.query('{0}:DCH-RF45_SEVERE_GLITCHING:1'.format(opts.detector),opts.gpsStart,opts.gpsEnd)
            O1_EY_BECKHOFF_CHASSIS_PROBLEM = DataQualityFlag.query('{0}:DCH-EY_BECKHOFF_CHASSIS_PROBLEM:1'.format(opts.detector),opts.gpsStart,opts.gpsEnd)
            O1_ASC_AS_B_RF36_GLITCHING = DataQualityFlag.query('{0}:DCH-ASC_AS_B_RF36_GLITCHING:1'.format(opts.detector),opts.gpsStart,opts.gpsEnd)
            O1_BAD_STRAIN_HOFT_C02 = DataQualityFlag.query('{0}:DCS-BAD_STRAIN_{0}_HOFT_C02:2'.format(opts.detector),opts.gpsStart,opts.gpsEnd)
            O1_TOGGLING_BAD_KAPPA_HOFT_C02= DataQualityFlag.query('{0}:DCS-TOGGLING_BAD_KAPPA_{0}_HOFT_C02:1'.format(opts.detector),opts.gpsStart,opts.gpsEnd)
        else:
            O1_ETMY_SATURATION = DataQualityFlag.query('{0}:DCH-ETMY_SATURATION:1'.format(opts.detector),opts.gpsStart,opts.gpsEnd)
            O1_BAD_DATA_BEFORE_LOCKLOSS = DataQualityFlag.query('{0}:DCH-BAD_DATA_BEFORE_LOCKLOSS:2'.format(opts.detector),opts.gpsStart,opts.gpsEnd)
            O1_PCAL_GLITCHES_GT_20P = DataQualityFlag.query('{0}:DCH-PCAL_GLITCHES_GT_20P:1'.format(opts.detector),opts.gpsStart,opts.gpsEnd)
            O1_SUDDEN_PSD_CHANGE = DataQualityFlag.query('{0}:DCH-SUDDEN_PSD_CHANGE:1'.format(opts.detector),opts.gpsStart,opts.gpsEnd)
            O1_BAD_VCO_OFFSET = DataQualityFlag.query('{0}:DCH-BAD_VCO_OFFSET:1'.format(opts.detector),opts.gpsStart,opts.gpsEnd)
            O1_SEVERE_60_200_HZ_NOISE = DataQualityFlag.query('{0}:DCH-SEVERE_60_200_HZ_NOISE:1'.format(opts.detector),opts.gpsStart,opts.gpsEnd)

    # Fetch raw omicron triggers and apply filter which is defined in a function above.
    omicrontriggers = SnglBurstTable.fetch(detchannelname,'Omicron',\
    opts.gpsStart,opts.gpsEnd,filt=threshold)

    print "List of available metadata information for a given glitch provided by omicron: {0}".format(omicrontriggers.columnnames)

    print "Number of triggers after SNR and Freq cuts but before ANALYSIS READY flag filtering: {0}".format(len(omicrontriggers))

    # Filter the raw omicron triggers against the ANALYSIS READY flag.
    omicrontriggers = omicrontriggers.vetoed(analysis_ready.active)
    # If requested filter out DQ flags
    if opts.applyallDQ:
        print("We are applying all previously created DQ cuts")
        # Obtain segments of all DQ cuts if requested DQ list can be found
        # https://code.pycbc.phy.syr.edu/detchar/veto-definitions/blob/master/burst/O1/H1L1-HOFT_C02_O1_BURST.xml
        # First obtain those flags that are for both H1 and L1
        omicrontriggers = omicrontriggers.veto(O1_MISSING_HOFT_C02.active)
        omicrontriggers = omicrontriggers.veto(O1_ETMY_ESD_DAC_OVERFLOW.active)
        omicrontriggers = omicrontriggers.veto(O1_OMC_DCPD_A_SATURATION.active)
        omicrontriggers = omicrontriggers.veto(O1_OMC_DCPD_ADC_OVERFLOW.active)
        omicrontriggers = omicrontriggers.veto(O1_ETMY_SATURATION_SNR200.active)
        omicrontriggers = omicrontriggers.veto(O1_CW_INJECTION_TRANSITION.active)
        omicrontriggers = omicrontriggers.veto(O1_BAD_KAPPA_BASIC_CUT_HOFT_C02.active)
        omicrontriggers = omicrontriggers.veto(O1_PARTIAL_FRAME_LOSS_HOFT_C02.active)

        # Obtain detector specific flags
        if opts.detector == "H1":
            omicrontriggers = omicrontriggers.veto(O1_RF45_AM_STABILIZATION.active)
            omicrontriggers = omicrontriggers.veto(O1_ETMY_SATURATION.active)
            omicrontriggers = omicrontriggers.veto(O1_BAD_DATA_BEFORE_LOCKLOSS.active)
            omicrontriggers = omicrontriggers.veto(O1_ETMY_VIOLIN_MODE_2NDHARMONIC_RINGING.active)
            omicrontriggers = omicrontriggers.veto(O1_RF45_SEVERE_GLITCHING.active)
            omicrontriggers = omicrontriggers.veto(O1_EY_BECKHOFF_CHASSIS_PROBLEM.active)
            omicrontriggers = omicrontriggers.veto(O1_ASC_AS_B_RF36_GLITCHING.active)
            omicrontriggers = omicrontriggers.veto(O1_BAD_STRAIN_HOFT_C02.active)
            omicrontriggers = omicrontriggers.veto(O1_TOGGLING_BAD_KAPPA_HOFT_C02.active)
        else:
            omicrontriggers = omicrontriggers.veto(O1_ETMY_SATURATION.active)
            omicrontriggers = omicrontriggers.veto(O1_BAD_DATA_BEFORE_LOCKLOSS.active)
            omicrontriggers = omicrontriggers.veto(O1_PCAL_GLITCHES_GT_20P.active)
            omicrontriggers = omicrontriggers.veto(O1_SUDDEN_PSD_CHANGE.active)
            omicrontriggers = omicrontriggers.veto(O1_BAD_VCO_OFFSET.active)
            omicrontriggers = omicrontriggers.veto(O1_SEVERE_60_200_HZ_NOISE.active)
        


    print "Final trigger length: {0}".format(len(omicrontriggers))

    return omicrontriggers

###############################################################################
##########################                            #########################
##########################   Func: write_dagfile     #########################
##########################                            #########################
###############################################################################

# Write dag_file file for the condor job

def write_dagfile(eventTime,ID):
    with open('gravityspy.dag','a+') as dagfile:
        dagfile.write('JOB {0} ./condor/gravityspy.sub\n'.format(eventTime))
        dagfile.write('RETRY {0} 3\n'.format(eventTime))
        dagfile.write('VARS {0} jobNumber="{0}" eventTime="{0}" ID="{1}"'.format(eventTime,ID))
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
        subfile.write('arguments = "--inifile wini.ini --eventTime $(eventTime) --outDir {0} --uniqueID --ID $(ID) --plot-eventgram"\n'.format(opts.outDir))
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

with open('./metadata/metadata_L1PostDQ.txt','a+') as f:
    f.write('# snr,amplitude,peak_frequency,central_freq,duration,bandwidth,chisq,chisq_dof,GPStime,ID\n')

for omicrontrigger in omicrontriggers:
    if opts.uniqueID:
        ID = id_generator()
        write_dagfile(omicrontrigger.get_peak(),ID)
        with open('./metadata/metadata_L1PostDQ.txt','a+') as f:
            f.write('{0} {1} {2} {3} {4} {5} {6} {7} {8} {9} {10}:{11}\n'.format(omicrontrigger.snr,omicrontrigger.amplitude,omicrontrigger.peak_frequency,omicrontrigger.central_freq,omicrontrigger.duration,omicrontrigger.bandwidth,omicrontrigger.chisq,omicrontrigger.chisq_dof,omicrontrigger.get_peak(),ID,omicrontrigger.ifo,omicrontrigger.channel))
            f.close()

# need to save omicron triggers variable to be loaded 
# by the condor job later
filename = open("test.xml",'w')
omicrontriggers.write(fileobj=filename)
