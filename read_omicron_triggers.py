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
import os,sys,time, glob, optparse, subprocess, csv
import numpy as np

def parse_commandline():
    """
    Parse the options given on the command-line.
    """
    parser = optparse.OptionParser()
    parser.add_option("--boxtime", help="Different time ranges (1 would represent +-0.5 around central time) for displaying the glitch) For Example: '[0.5 1 4 16]' would give you four differnt images with those different times boxes [Default:[0.5,1,2,4]",
                        default=[0.5,1,2,4])
    parser.add_option("--channelname", help="channel name [Default:GDS-CALIB_STRAIN]",
                        default="GDS-CALIB_STRAIN")
    parser.add_option("--colorMap", help="What color would you like the omegascans to be? [Default: jet] [Options: bone, copper,hot]", default="jet")
    parser.add_option("--detector", help="detector name [L1 or H1]. [No Default must supply]")
    parser.add_option("--gpsStart", help="gps Start Time of Query for meta data and omega scans. [No Default must supply]")
    parser.add_option("--gpsEnd", help="gps End Time [No Default must supply]")
    parser.add_option("--imagepath", help="path to directory for images NO matter what path you select {detectorname}_{gpsStart}_{gpsEnd} will be added as a subdirectory [Default ~/public_html/GravitySpy/]",
                        default='~/public_html/GravitySpy/')
    parser.add_option("--nds2name", help="ip address or URL of the nds2\
                        server to use [Default: nds.ligo.caltech.edu]",
                        default="nds.ligo.caltech.edu")
    parser.add_option("--normalizedSNR", help="SNR Normalization value. [Default: 25.5]",default="25.5")
    parser.add_option("--outpath", help="path to output directory [Default: ./png] (This is where the images are created but you will not have to go in this folder as the iamges are transferred to your public_html imagepath folder)",
                        default=os.getcwd() + '/png')
    parser.add_option("--runlocal", help="run locally (running as a condor job has not been set up yet)",default=1)
    parser.add_option("--sampfrequency", help="sample frequency for omegascan iamges [Default: 4096]", default=4096)
    parser.add_option("--SNR", help="SNR Threshold for omicron triggers. [Default: 6]",default="6")
    parser.add_option("--verbose", action="store_true", default=False,
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
def make_images(centraltime,nds2name,detchannelname,outpath,imagepathname,normalizedSNR,boxtime,verbose,imagepath,sampfrequency,colorMap):

    # Make temporary directory to create the images in
    system_call = 'mktemp -d {0}/AAA.XXXXXXXXXX'.format(outpath)
    os.system(system_call)
    
    tempname = os.listdir("{0}/".format(outpath))
    tempdir = ('{0}/'.format(outpath) + tempname[0])
    uniqueid = tempname[0].split('.')[1]
    print('unique id is {0}'.format(uniqueid))

    # Use random folder name as I.D. Tag but first check if this gps time 
    # already has a I.D. tag and use that if it does.
    idfile =  open('ID.txt', "a+")
    idfile.write('{0} {1} {2}\n'.format(detchannelname,centraltime,uniqueid))
    idfile.close()
    # Open file for writing metadata for images
    metadata =  open(imagepath + '/imagemetadata.txt', "a+")


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
        colorMap={7} \
        plotFrequencyRange='[10 inf]' \
        plotNormalizedEnergyRange='[0.0 {8}]'  \
        searchTimeRange=64 \
        searchFrequencyRange='[0 inf]' \
        searchQRange='[4.0 64.0]'\n".format(os.getcwd(),nds2name,detchannelname,centraltime,tempdir,boxtime,sampfrequency,colorMap,normalizedSNR))
    g.close()

    if verbose == True:
        system_call = 'cat {0}/temprun.sh'.format(tempdir)
        os.system(system_call)

    system_call = 'source {0}/temprun.sh'.format(tempdir)
    os.system(system_call)

    os.chdir('{0}'.format(tempdir))

    pngnames = os.listdir('./')
    if verbose == True:
	print(pngnames)

    # If image is not created server must be down. Exit fucntion
    if len(pngnames) ==1:
	print "OH NO!"
	# write file to indicate at what GPStime the image generation failed.
	sys.exit()

    for iTime in xrange(0,len(boxtime)):

        system_call = 'convert *{0}_{1}* -chop 0x35 {2}/{3}_{4}_{5}.png'.format(detchannelname,boxtime[iTime],outpath,opts.detector,uniqueid,boxtime[iTime])
        os.system(system_call)

    # Create .csv to upload metadata of images to the project builder
    metadata.write('{0} 20151016 {1}_{2}_{3}.png {4}_{5}_{6}.png {7}_{8}_{9}.png {10}_{11}_{12}.png\n'.format(uniqueid,opts.detector,uniqueid,boxtime[0],opts.detector,uniqueid,boxtime[1],opts.detector,uniqueid,boxtime[2],opts.detector,uniqueid,boxtime[3]))
    metadata.close()

    system_call = 'rm -rf {0}/'.format(tempdir)
    os.system(system_call)

    system_call = 'cp {0}/{1}_{2}*.png {3}'.format(outpath,opts.detector,uniqueid,imagepath)
    os.system(system_call)

    system_call = 'rm -rf {0}/*'.format(outpath)
    os.system(system_call)

    os.chdir('../..')
    return uniqueid

####################
## MAIN CODE ######
####################
# Parse commandline arguments
opts = parse_commandline()

detchannelname = opts.detector + ':' +  opts.channelname

# Let the user know what detector and channel combo they have selected.
print "You have selected detector channel combination {0}".format(detchannelname)

# Let the user know which SNR cut they have selected.
print "You have selected a SNR cut of {0}".format(opts.SNR)

# Let the user know which SNR normalization max they have selected.
print "You have selected a SNR normalization max of {0}".format(opts.normalizedSNR)

# Let the user know which colormpa color they have  selected.
print "You have selected a colormap color of {0}".format(opts.colorMap)

# Obtain segments from L1 that are analysis ready
analysis_ready = DataQualityFlag.query('{0}:DMT-ANALYSIS_READY:1'.format(opts.detector),opts.gpsStart,opts.gpsEnd)

# Display segments for which this flag is true
print "Segments for which the ANALYSIS READY Flag is active: {0}".format(analysis_ready.active)

# Fetch raw omicron triggers and apply filter
omicrontriggers = SnglBurstTable.fetch(detchannelname,'Omicron',\
opts.gpsStart,opts.gpsEnd,filt=snr_freq_threshold)

print "List of available metadata information for a given glitch provided by omicron: {0}".format(omicrontriggers.columnnames)

print "Number of triggers after SNR and Freq cuts but before ANALYSIS READY flag filtering: {0}".format(len(omicrontriggers))

# Filter the raw omicron triggers against the ANALYSIS READY flag.
omicrontriggers = omicrontriggers.vetoed(analysis_ready.active)

print "Final trigger length: {0}".format(len(omicrontriggers))

# Create a string of detector name, gpsStart and gpsEnd time that will be used repeatedly throughout code

detGPSstr = opts.detector + '_'  + str(opts.gpsStart) + '_' + str(opts.gpsEnd)

# Take imagepath add a directory indicating the detector and the gpsStart and gpsEnd times
imagepathname = os.path.expanduser(opts.imagepath) + '/' + detGPSstr + '/'
system_call = 'mkdir -p ' + imagepathname
os.system(system_call)

# create a path where the images will get generated

system_call = 'mkdir -p ' + opts.outpath
os.system(system_call)
# Create the path to the glitch metadata
system_call = 'mkdir -p glitchmetadata'
os.system(system_call)

# If verbose spit out input parameters to terminal

if opts.verbose == True:
    print('output directory:' + opts.outpath)
    print('channel name:'     + opts.channelname)
    print('NDS2 file:'        + opts.nds2name)
    print('Path to Images:'   + imagepathname)

# Depending on if you are running locally or not. If running locally then you must run kinit before submitting mkOmega jobs. If you have a keytab account then you must specificy your username and password.
 
if int(opts.runlocal) == 1:
    print('Did you remember to run kinit albert.einstein@LIGO.ORG?')
if int(opts.runlocal) == 0:
        print >> sys.stderr, "FIXME:Not running locally has not been set up yet"
        sys.exit(1)

# open a txt file where the image metadata for consumption by the Zooniverse servers will be stored.
metadata =  open(imagepathname + '/imagemetadata.txt', "w+")
metadata.write('ID Date Filename1 Filename2 Filename3 Filename4\n')
metadata.close()

# Open metadata file for GLITCH metadata information
glitchmetadata = open('./glitchmetadata/' + detGPSstr + '.txt',"w+") 
# Headers indicating the metadata WE have selected as of right now.
# SNR, Amplitude, peak_freq, cent_freq, duration, bandwidth, UniqueID
glitchmetadata.write('# SNR, Amplitude, peak_freq, cent_freq, duration, bandwidth, UniqueID\n')
glitchmetadata.close()

# open Id.txt which will store the link between GPS time and Randomly Generated Unqiue ID of an image
# if there is already an Id.txt file, just append the new IDs to it
if os.path.isfile('ID.txt')==False:
	idfile =  open('ID.txt', "w+")
	idfile.write('# Channel GPSTime UniqueID\n')
	idfile.close()
else:
	idfile =  open('ID.txt', "a+")
	idfile.close()

for omicrontrigger in omicrontriggers:
    # Run the function make_images which will generate the iamge and create an uniqueID to assign to that image
    uniqueid = make_images('{0}.{1}'.format(omicrontrigger.peak_time,omicrontrigger.peak_time_ns),opts.nds2name,detchannelname,opts.outpath,imagepathname,opts.normalizedSNR,opts.boxtime,opts.verbose,imagepathname,opts.sampfrequency,opts.colorMap)
	
    # For this trigger write all the metadata of the trigger plus the unqiueID generated during make_images 
    with open('./glitchmetadata/' + detGPSstr + '.txt','a+') as f:
	f.write('{0} {1} {2} {3} {4} {5} {6}\n'.format(omicrontrigger.snr,omicrontrigger.amplitude,omicrontrigger.peak_frequency,omicrontrigger.central_freq,omicrontrigger.duration,omicrontrigger.bandwidth,uniqueid))
	f.close()

# House Cleaning

system_call = 'rm -rf {0}'.format(opts.outpath)
os.system(system_call)

# We must now convert image metadata to CSV to prep for upload.

txt_file = imagepathname  + '/imagemetadata.txt'
csv_file = imagepathname  + '/imagemetadata.csv'
in_txt = csv.reader(open(txt_file, "rb"), delimiter = ' ')
out_csv = csv.writer(open(csv_file, 'wb'))
out_csv.writerows(in_txt)
system_call = 'tar -czvf ' + imagepathname + '/' + detGPSstr + '.tar ' + imagepathname  + '/*.png'
os.system(system_call)
