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
                        default=[0.5,1,2,4])
    parser.add_option("-n", "--nds2name", help="ip address or URL of the nds2\
                        server to use [nds.ligo.caltech.edu]",
                        default="nds.ligo.caltech.edu")
    parser.add_option("-o", "--outpath", help="path to output directory [./png]",
                        default=os.getcwd() + '/png')
    parser.add_option("-j", "--imagepath", help="path to directory for images",
                        default='/home/scoughlin/public_html/GlitchZoo/L_O1_Plots/')
    parser.add_option("-s", "--submitpath", help="path to script/submit directory [./submits]",
                        default=os.getcwd() + '/submits')
    parser.add_option("-r", "--SNR", help="SNR Threshold",default="6")
    parser.add_option("-z", "--normalizedSNR", help="SNR Normalization value",default="25.5")
    parser.add_option("-g", "--gpsStart", help="gps Start Time of Query for meta data and omega scans",default="0")
    parser.add_option("-e", "--gpsEnd", help="gps End Time",default=0)
    parser.add_option("-f", "--sampfrequency", help="sample frequency", default=4096)
    parser.add_option("-m", "--maxJobs", help="How many subjects in a given subfolder",default=1000)
    parser.add_option("-l", "--runlocal", help="run locally",default=1)
    parser.add_option("-v", "--verbose", action="store_true", default=False,
                      help="Run verbosely. (Default: False)")

    opts, args = parser.parse_args()

    return opts

# Define snr_freq_threshold function that removes rows where SNR <6 and removes any triggers whose peak_freqeuency was above 2048 Hz and below 10.

def snr_freq_threshold(row):
    if (row.snr >= float(opts.SNR)) and (row.peak_frequency <= 2048) and (row.peak_frequency >= 10):
        passthresh = True
    else:
        passthresh = False
    return passthresh

#taking place of mkOmega.py
def make_images(centraltime,nds2name,channelname,outpath,imagepathname,normalizedSNR,boxtime,verbose,imagepath,sampfrequency):

    # Make temporary directory to create the images in
    system_call = 'mktemp -d {0}/AAA.XXXXXXXXXX'.format(outpath)
    os.system(system_call)
    
    tempname = os.listdir("{0}/".format(outpath))
    tempdir = ('{0}/'.format(outpath) + tempname[0])
    uniqueid = tempname[0].split('.')[1]
    print('unique id is {0}'.format(uniqueid))

    # Use random folder name as I.D. Tag but first check if this gps time 
    # already has a I.D. tag and use that if it does.
    idfile =  open(os.getcwd() + '/IDFolder/ID.txt', "a+")
    idfile.write('{0} {1} {2}\n'.format(channelname,centraltime,uniqueid))
    idfile.close()
    # Open file for writing metadata for images
    metadata =  open(imagepath + '/metadata.txt', "a+")


    script = os.path.join(tempdir,'temprun.sh')
    g =  open(script, "w+") # write mode
    g.write("#!/bin/bash\n")
    g.write("java -Xmx16m -jar {0}/packwplot.jar \
        frameType=NDS2 \
        ndsServer={1} \
        channelName={2} \
        eventTime={3} \
        outputDirectory={4} \
        plotType=spectrogram_whitened \
        plotTimeRanges='{5}' \
        sampleFrequency={6} \
        colorMap=jet \
        plotFrequencyRange='[10 inf]' \
        plotNormalizedEnergyRange='[0.0 {7}]'  \
        searchTimeRange=64 \
        searchFrequencyRange='[0 inf]' \
        searchQRange='[4.0 64.0]'\n".format(os.getcwd(),nds2name,channelname,centraltime,tempdir,boxtime,sampfrequency,normalizedSNR))
    g.close()

    if verbose == True:
        system_call = 'cat {0}/temprun.sh'.format(tempdir)
        os.system(system_call)

    system_call = 'source {0}/temprun.sh'.format(tempdir)
    os.system(system_call)

    os.chdir('{0}'.format(tempdir))

    pngnames = os.listdir('./')

    # If image is not created server must be down. Exit fucntion
    if len(pngnames) ==1:
	sys.exit()

    for iFile in xrange(1,len(pngnames)):

    	if channelname == "L1:GDS-CALIB_STRAIN":
             system_call = 'convert {0} -chop 0x35 {1}/L1_{2}_{3}.png'.format(pngnames[iFile],outpath,uniqueid,boxtime[iFile-1])
    	elif channelname == "H1:GDS-CALIB_STRAIN":
             system_call = 'convert {0} -chop 0x35 {1}/H1_{2}_{3}.png'.format(pngnames[iFile],outpath,uniqueid,boxtime[iFile-1])
        os.system(system_call)

    # Create .csv to upload metadata of images to the project builder
    if channelname == "L1:GDS-CALIB_STRAIN":
        metadata.write('{0} 20151016 L1_{1}_{2}.png L1_{3}_{4}.png L1_{5}_{6}.png L1_{7}_{8}.png\n'.format(uniqueid,uniqueid,boxtime[0],uniqueid,boxtime[1],uniqueid,boxtime[2],uniqueid,boxtime[3]))
    elif channelname == "H1:GDS-CALIB_STRAIN":
    	metadata.write('{0} 20151016 H1_{1}_{2}.png H1_{3}_{4}.png H1_{5}_{6}.png H1_{7}_{8}.png\n'.format(uniqueid,uniqueid,boxtime[0],uniqueid,boxtime[1],uniqueid,boxtime[2],uniqueid,boxtime[3]))
    metadata.close()

    system_call = 'rm -rf {0}/'.format(tempdir)
    os.system(system_call)

    if channelname == "L1:GDS-CALIB_STRAIN":
    	system_call = 'cp {0}/L1_{1}*.png {2}'.format(outpath,uniqueid,imagepath)
    elif channelname == "H1:GDS-CALIB_STRAIN":
    	system_call = 'cp {0}/H1_{1}*.png {2}'.format(outpath,uniqueid,imagepath)
    os.system(system_call)

    system_call = 'rm -rf {0}/*'.format(outpath)
    os.system(system_call)

    os.chdir('..')
    print(os.listdir('.')) 
    return uniqueid

####################
## MAIN CODE ######
####################
# Parse commandline arguments
opts = parse_commandline()

print "You have selected a SNR cut of {0}".format(opts.SNR)

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
    f.close

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

# Take imagepath (such as L_O1_Plots) add a directory indicating the gpsStart and gpsEnd times
imagepathname = opts.imagepath + '/' + opts.gpsStart + '_' + opts.gpsEnd + '/'
system_call = 'mkdir -p ' + imagepathname
print(system_call)
os.system(system_call)

metadata =  open(imagepathname + '/metadata.txt', "w+")
metadata.write('ID Date Filename1 Filename2 Filename3 Filename4\n')
metadata.close()
idfile =  open(os.getcwd() + '/IDFolder/ID.txt', "a+")
idfile.write('# Channel GPSTime UniqueID\n')
idfile.close()

if opts.verbose == True:
    for omicrontrigger in omicrontriggers:
        uniqueid = make_images('{0}.{1}'.format(omicrontrigger.peak_time,omicrontrigger.peak_time_ns),opts.nds2name,opts.channelname,opts.outpath,imagepathname,opts.normalizedSNR,opts.boxtime,opts.verbose,imagepathname,opts.sampfrequency)
	with open('metadata_' + str(opts.gpsStart) + '_' + str(opts.gpsEnd) + '.txt','a+') as f:
	    f.write('{0} {1} {2} {3} {4} {5} {6}\n'.format(omicrontrigger.snr,omicrontrigger.amplitude,omicrontrigger.peak_frequency,omicrontrigger.central_freq,omicrontrigger.duration,omicrontrigger.bandwidth,uniqueid))
	    f.close()
else:   
    for omicrontrigger in omicrontriggers:
        uniqueid = make_images('{0}.{1}'.format(omicrontrigger.peak_time,omicrontrigger.peak_time_ns),opts.nds2name,opts.channelname,opts.outpath,imagepathname,opts.normalizedSNR,opts.boxtime,opts.verbose,imagepathname,opts.sampfrequency)

        f.write('{0} {1} {2} {3} {4} {5} {6}\n'.format(omicrontrigger.snr,omicrontrigger.amplitude,omicrontrigger.peak_frequency,omicrontrigger.central_freq,omicrontrigger.duration,omicrontrigger.bandwidth,uniqueid))
# We must now convert image metadata to CSV to prep for upload.

txt_file = opts.imagepath  + '/metadata.txt'
csv_file = opts.imagepath  + '/metadata.csv'
in_txt = csv.reader(open(txt_file, "rb"), delimiter = ' ')
out_csv = csv.writer(open(csv_file, 'wb'))
out_csv.writerows(in_txt)
system_call = 'tar -czvf ' + opts.imagepath + '/L_O1_plots.tar ' + opts.imagepath  + '/*.png'
os.system(system_call)
