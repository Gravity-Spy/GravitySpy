#!/usr/bin/env python

import optparse,os,string,random,pdb
from gwpy.table.lsctables import SnglBurstTable
from gwpy.segments import DataQualityFlag
import pandas as pd
from sqlalchemy.engine import create_engine
import cPickle as pickle
import numpy as np

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
    parser.add_option("--pathToModel",default='./ML/trained_model/' ,help="Path to trained model")
    parser.add_option("--SNR", help="Lower bound SNR Threshold for omicron triggers, by default there is no upperbound SNR unless supplied throught the --maxSNR flag. [Default: 6]",type=float,default=6)
    parser.add_option("--uniqueID", action="store_true", default=False,help="Is this image being generated for the GravitySpy project, is so we will create a uniqueID strong to use for labeling images instead of GPS time")
    parser.add_option("--HDF5", action="store_true", default=False,help="Store triggers in local HDF5 table format")
    parser.add_option("--PostgreSQL", action="store_true", default=False,help="Store triggers in local PostgreSQL format")
    parser.add_option("--Pipeline", help="Pipeline whose triggers you are downloading")
    parser.add_option("--URL", help="Pipeline whose triggers you are downloading")
    parser.add_option("--Label", help="Provide the label assocaited with the text file if known")

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
        dagfile.write('JOB {0}{1}{2} ./condor/gravityspy.sub\n'.format(x.peak_time,x.peak_time_ns,x.event_id))
        dagfile.write('RETRY {0}{1}{2} 3\n'.format(x.peak_time,x.peak_time_ns,x.event_id))
        dagfile.write('VARS {0}{1}{4} jobNumber="{0}{1}{4}" eventTime="{2}" ID="{3}"'.format(x.peak_time,x.peak_time_ns,x.peakGPS,x.uniqueID,x.event_id))
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
        subfile.write('universe = local\n')
        subfile.write('executable = {0}\n'.format(opts.pathToExec))
        subfile.write('\n')
        subfile.write('arguments = "--inifile {0} --eventTime $(eventTime) --outDir {1} --pathToModel {2} --uniqueID --ID $(ID)"\n'.format(opts.pathToIni,opts.outDir,opts.pathToModel))
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

# SQL table to store results
engine = create_engine('postgresql://{0}:{1}@gravityspy.ciera.northwestern.edu:5432/gravityspy'.format(os.environ['QUEST_SQL_USER'],os.environ['QUEST_SQL_PASSWORD']))

# Take the detector and the channel from the command line and combine them into one string. This is needed for some input later on.
detchannelname = opts.detector + ':' + opts.channelname

# Disentangle URL
Path = '{0}'.format(opts.URL).replace('jobs','pcdev1')
PathList = Path.split('/')[2::]
PathList.insert(2,'public_html')
PathToTrig = '/'.join(PathList).replace('edu','edu:').replace('~','/home/')
fileName = opts.Pipeline + '.txt'
os.system('gsiscp {0} {1}'.format(PathToTrig,fileName))
oTriggers = pd.DataFrame()

if opts.Pipeline == 'LAL':
    tmp = pd.read_csv(fileName,names=['GPS'])
    # Find Omicron trigger associated with these times
    for iGPS in tmp.GPS:
        try:
            omicrontriggers = SnglBurstTable.fetch(detchannelname,'Omicron',iGPS-1,iGPS+1)
            tmp1 = pd.DataFrame(omicrontriggers.to_recarray(),omicrontriggers.get_peak()).reset_index()
            tmp1 = tmp1.loc[tmp1['index'] == min(omicrontriggers.get_peak(), key=lambda x:abs(x-iGPS))]
            oTriggers = oTriggers.append(tmp1)
        except:
            print(iGPS)

    oTriggers['Label'] = opts.Label
    oTriggers['Pipeline'] = opts.Pipeline
    oTriggers.rename(columns = {'index':'peakGPS'},inplace=True)

if opts.Pipeline == 'PCAT':
    tmp = pd.read_csv(fileName,names=['GPS'])
    oTriggers = pd.DataFrame()
    for iGPS in tmp.GPS:
        try:
            omicrontriggers = SnglBurstTable.fetch(detchannelname,'Omicron',iGPS-2,iGPS+2)
            tmp1 = pd.DataFrame(omicrontriggers.to_recarray(),omicrontriggers.get_peak()).reset_index()
            tmp1 = tmp1.loc[tmp1['index'] == min(omicrontriggers.get_peak(), key=lambda x:abs(x-iGPS))]
            oTriggers = oTriggers.append(tmp1)
        except:
            print(iGPS)

    oTriggers['Label'] = opts.Label
    oTriggers['Pipeline'] = opts.Pipeline
    oTriggers.rename(columns = {'index':'peakGPS'},inplace=True)

if opts.Pipeline == 'WDF':
    oTriggers = pd.DataFrame()
    tmp = pd.read_csv(fileName)
    for (iGPS, iSNR, iDur, iLabel, iPeakT, iPeakF) in zip(tmp.GPSMax, tmp.SNRMax, tmp.Duration, tmp.LABEL, tmp.GPSstart, tmp.FreqMax):
        try:
            omicrontriggers = SnglBurstTable.fetch(detchannelname,'Omicron',iGPS-2,iGPS+2)
            tmp1 = pd.DataFrame(omicrontriggers.to_recarray(),omicrontriggers.get_peak()).reset_index()
            tmp1 = tmp1.loc[tmp1['index'] == min(omicrontriggers.get_peak(), key=lambda x:abs(x-iGPS))]
            tmp1['Label']    = iLabel
            tmp1['WDF_snr'] = iSNR
            tmp1['WDF_duration'] = iDur
            tmp1['WDF_iPeakT'] = iPeakT
            tmp1['WDF_iPeakF'] = iPeakF
            oTriggers = oTriggers.append(tmp1)
        except:
            print(iGPS)

    oTriggers['Pipeline'] = opts.Pipeline
    oTriggers.rename(columns = {'index':'peakGPS'},inplace=True)

if opts.Pipeline == 'DBNN':
    tmp = pd.read_csv(fileName,names=['GPS'])
    oTriggers = pd.DataFrame()
    for iGPS in tmp.GPS:
        try:
            omicrontriggers = SnglBurstTable.fetch(detchannelname,'Omicron',iGPS-2,iGPS+2)
            tmp1 = pd.DataFrame(omicrontriggers.to_recarray(),omicrontriggers.get_peak()).reset_index()
            tmp1 = tmp1.loc[tmp1['index'] == min(omicrontriggers.get_peak(), key=lambda x:abs(x-iGPS))]
            oTriggers = oTriggers.append(tmp1)
        except:
            print(iGPS)

    oTriggers['Label'] = opts.Label
    oTriggers['Pipeline'] = opts.Pipeline
    oTriggers.rename(columns = {'index':'peakGPS'},inplace=True)


# Read file
# Can we determine the label from either the name of the file or a column in 

write_subfile()
oTriggers['uniqueID'] = oTriggers.peakGPS.apply(id_generator)
oTriggers.drop_duplicates(['peakGPS'],inplace=True)
oTriggers[['peak_time','peak_time_ns','peakGPS','uniqueID','event_id']].apply(write_dagfile,axis=1)
oTriggers.peakGPS = oTriggers.peakGPS.apply(float)
oTriggers = oTriggers[['peakGPS', 'ifo', 'peak_time', 'peak_time_ns', 'start_time','start_time_ns', 'duration', 'search', 'process_id', 'event_id','peak_frequency', 'central_freq', 'bandwidth', 'channel','amplitude', 'snr', 'confidence', 'chisq', 'chisq_dof','param_one_name', 'param_one_value', 'Label', 'Pipeline','uniqueID']]
oTriggers.to_csv(open('{0}_test.csv'.format(opts.Pipeline),'w'),index=False)
oTriggers.to_sql('O1GlitchClassificationUpdate',engine,index=False,if_exists='append',chunksize=100)
#os.system('/bin/condor_submit_dag -maxjobs 10 gravityspy_{0}_{1}.dag'.format(oTriggers.peak_time.min(),oTriggers.peak_time.max())) 
