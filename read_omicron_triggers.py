#!/usr/bin/env python

# Goal of this function is to complete the following work flow
# 1) Get some Omicron triggers (source: ~detchar/triggers XML)
# 2) Get ANALYSIS_READY segments from the segdb
# 3) Filter 1 to take out anything that is not in 2.
# 4) Filter 3 to only include triggers with SNR>6 and 10<=freq<2048.  
# 5) Write some new Omicron XML files (or even text files or HDF5 files) that contain the triggers that were in ANALYSIS_READY and have SNR>6. 
# 6) Create Omega Scans of the remaining omicron triggers

# -------------------------------------------------------------------------
#      Setup.
# -------------------------------------------------------------------------

# ---- Import standard modules to the python path.
from gwpy.table.lsctables import SnglBurstTable
from gwpy.segments import DataQualityFlag
import os,sys,time, glob, optparse, subprocess
import numpy as np

def parse_commandline():
    """
    Parse the options given on the command-line.
    """
    parser = optparse.OptionParser()
    parser.add_option("-u", "--username", help="user name for the keytab: albert.einstein@LIGO.ORG",default="")
    parser.add_option("-k", "--keytab", help="full path to the keytab\
                        for that user. FOLLOW SECURITY GUIDLINES for permission"
                        ,default="")
    parser.add_option("-c", "--channelname", help="channel name",
                        default="L1:GDS-CALIB_STRAIN")
    parser.add_option("-b", "--boxtime", help="Different time ranges (1 would represent +-0.5 around central time) for displaying the glitch) For Example: '[0.5 1 4 16]' would give you four differnt images with those different times boxes",
                        default="[0.5 1 4 16]")
    parser.add_option("-n", "--nds2name", help="ip address or URL of the nds2\
                        server to use [nds.ligo.caltech.edu]",
                        default="nds.ligo.caltech.edu")
    parser.add_option("-o", "--outpath", help="path to output directory [./png]",
                        default=os.getcwd() + '/png')
    parser.add_option("-j", "--imagepath", help="path to directory for images",
                        default='/home/scoughlin/public_html/GlitchZoo/L_O1_Plots/')
    parser.add_option("-s", "--submitpath", help="path to script/submit directory [./submits]",
                        default=os.getcwd() + '/submits')
    parser.add_option("-g", "--gpsStart", help="gps Start Time of Query for meta data and omega scans",default="0")
    parser.add_option("-e", "--gpsEnd", help="gps End Time",default=0)
    parser.add_option("-m", "--maxJobs", help="How many subjects in a given subfolder",default=1000)
    parser.add_option("-l", "--runlocal", help="run locally",default=1)
    parser.add_option("-v", "--verbose", action="store_true", default=False,
                      help="Run verbosely. (Default: False)")

    opts, args = parser.parse_args()

    return opts

# Define snr_freq_threshold function that removes rows where SNR <6 and removes any triggers whose peak_freqeuency was above 2048 Hz and below 10.

def snr_freq_threshold(row):
    if (row.snr >= 6) and (row.peak_frequency <= 2048) and (row.peak_frequency >= 10):
        passthresh = True
    else:
        passthresh = False
    return passthresh

####################
## MAIN CODE ######
####################
# Parse commandline arguments
opts = parse_commandline()

# Obtain segments from L1 that are analysis ready
analysis_ready = DataQualityFlag.query('L1:DMT-ANALYSIS_READY:1',opts.gpsStart,opts.gpsEnd)

# Display segments for which this flag is true
print "Segments for which the ANALYSIS READY Flag is active: {0}".format(analysis_ready.active)

# Fetch raw omicron triggers and apply filter
omicrontriggers = SnglBurstTable.fetch(opts.channelname,'Omicron',\
opts.gpsStart,opts.gpsEnd,filt=snr_freq_threshold)

print omicrontriggers.columnnames
print "Original trigger length before ANALYSIS READY flag filter: {0}".format(len(omicrontriggers) )

# Filter the raw omicron triggers against the ANALYSIS READY flag.
omicrontriggers = omicrontriggers.vetoed(analysis_ready.active)

print "Final trigger length: {0}".format(len(omicrontriggers))

# Write some metadata to a .txt file

with open('metadata_' + str(opts.gpsStart) + '_' + str(opts.gpsEnd) + '.txt','w+') as f:
    # SNR, Amplitude, peak_freq, cent_freq, duration,bandwidth
    f.write('# SNR, Amplitude, peak_freq, cent_freq, duration,bandwidth\n')
    for omicrontrigger in omicrontriggers:
	f.write('{0} {1} {2} {3} {4} {5}\n'.format(omicrontrigger.snr,omicrontrigger.amplitude,omicrontrigger.peak_frequency,omicrontrigger.central_freq,omicrontrigger.duration,omicrontrigger.bandwidth))
    f.close()

# This script takes the peak_time and creates a number of jobs to generate Omega plots centered on those times. These images are used using mkOmega.py.

# create the paths if necessary

system_call = 'mkdir -p ' + opts.outpath
os.system(system_call)
system_call = 'mkdir -p ' + opts.submitpath
os.system(system_call)
system_call = 'mkdir -p ' + os.getcwd() + '/IDFolder/'
os.system(system_call)

# If verbose spit out input parameters to terminal

if opts.verbose == True:
    print('output directory:' + opts.outpath)
    print('channel name:'     + opts.channelname)
    print('User Name:'        + opts.username)
    print('KeyTabfile:'       + opts.keytab)
    print('NDS2 file:'        + opts.nds2name)
    print('SubmitDir:'        + opts.submitpath)

# Depending on if you are running locally or not. If running locally then you must run kinit before submitting mkOmega jobs. If you have a keytab account then you must specificy your username and password.
 
if int(opts.runlocal) == 1:
    print('Did you remember to run kinit albert.einstein@LIGO.ORG?')
if int(opts.runlocal) == 0:
    if opts.keytab == "":
        print >> sys.stderr, "No key-tab specified."
        print >> sys.stderr, "Use --keytab to specify it."
        sys.exit(1)
    if opts.keytab == "":
        print >> sys.stderr, "No Username specified."
        print >> sys.stderr, "Use --username to specify it."
        sys.exit(1)

# Initialize some variables 

curJob  = 1
numJobs = 1
iUpload = 1

# Take imagepath (such as L_O1_Plots) add a directory indicating the gpsStart and gpsEnd times
imagepathname = opts.imagepath + '/' + opts.gpsStart + '_' + opts.gpsEnd + '/'
system_call = 'mkdir -p ' + imagepathname + str(iUpload)
print(system_call)
os.system(system_call)

metadata =  open(imagepathname + str(iUpload) + '/metadata.txt', "w+")
metadata.write('ID Date Filename1 Filename2 Filename3 Filename4\n')
metadata.close()

outputLocation = opts.submitpath
script         = os.path.join(outputLocation,'job-' + str(curJob) + '.sh')
g =  open(script, "w+") # write mode

if opts.verbose == True:
    for omicrontrigger in omicrontriggers:
        g.write('python ' + os.getcwd() + '/mkOmega.py -v -t {0}.{1} -u {2} -k {3} -n {4} -c {5} -o {6} -j {7}{8}/\n'.format(omicrontrigger.peak_time,omicrontrigger.peak_time_ns,opts.username,opts.keytab,opts.nds2name,opts.channelname,opts.outpath,imagepathname,iUpload))
	numJobs = numJobs + 1

        if numJobs > opts.maxJobs:
            numJobs = 1

            # move to next image and image metadata outpath
            iUpload = iUpload+1
            system_call = 'mkdir -p ' + imagepathname + str(iUpload)
            print(system_call)
            os.system(system_call)
            metadata =  open(imagepathname + str(iUpload) + '/metadata.txt', "w+")
            metadata.write('ID Date Filename1 Filename2 Filename3 Filename4\n')
            metadata.close()
else:   
    for omicrontrigger in omicrontriggers:
        g.write('python ' + os.getcwd() + '/mkOmega.py -t {0}.{1} -u {2} -k {3} -n {4} -c {5} -o {6} -j {7}{8}/\n'.format(omicrontrigger.peak_time,omicrontrigger.peak_time_ns,opts.username,opts.keytab,opts.nds2name,opts.channelname,opts.outpath,imagepathname,iUpload))
        numJobs = numJobs + 1

        if numJobs > opts.maxJobs:
            numJobs = 1

            # move to next image and image metadata outpath
            iUpload = iUpload+1
            system_call = 'mkdir -p ' + imagepathname + str(iUpload)
            print(system_call)
            os.system(system_call)
            metadata =  open(imagepathname + str(iUpload) + '/metadata.txt', "w+")
            metadata.write('ID Date Filename1 Filename2 Filename3 Filename4\n')
            metadata.close()
# At the end there are two things we want to do. First we want to take the uniqueIDs which are created during the mkOmega part of the processing and add them to the glitch metadata we created before generating the images.

g.write("sed -i '1 i\#Detector GPSTime UniqueID' ./IDFolder/ID.txt\n")
g.write("awk < ./IDFolder/ID.txt '{print $3}' > justID.txt\n")
g.write('cp metadata_' + str(opts.gpsStart) + '_' + str(opts.gpsEnd) + '.txt metadata_' + str(opts.gpsStart) + '_' + str(opts.gpsEnd) + '_backup.txt\n')
g.write('paste -d" " metadata_' + str(opts.gpsStart) + '_' + str(opts.gpsEnd) + '.txt justID.txt>metadatamerge.txt\n')
g.write('mv metadatamerge.txt metadata_' + str(opts.gpsStart) + '_' + str(opts.gpsEnd) + '.txt\n')
g.write('cp metadata_' + str(opts.gpsStart) + '_' + str(opts.gpsEnd) + '.txt ' +imagepathname + '\n')  

# Run converttocsv.py to convert image metadata to CSV to prep for upload.

g.write('python converttocsv.py --imagepath {0} --iUploadMax {1}'.format(imagepathname,(iUpload+1)))
# Secondly, we want to make a call to API to take the images we just created (and CSV metadata) and upload them in batches to the API.
g.write('python callAPI.py --imagepath{')
g.close()
