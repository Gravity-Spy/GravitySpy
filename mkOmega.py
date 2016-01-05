#!/usr/bin/env python

# generate a 4 panel Omega plot for a specific time
# design to be run from command line or as a condor job

# -------------------------------------------------------------------------
#      Setup.
# -------------------------------------------------------------------------

# ---- Import standard modules to the python path.
import os,sys,time, glob, optparse, subprocess
import numpy as np
#import matplotlib
#matplotlib.use('Agg')
#from matplotlib import pyplot as plt
#import matplotlib.image as mpimg

def parse_commandline():
    """
    Parse the options given on the command-line.
    mkOmega is a script to control generation of a 4 panel Omega plot (dmt_wplot
    at a specific (sub-second) GPS time.  It is meant to be run from the command
    line or as a condor job.  It uses NDS2 to transfer data so a Kerberos ticket
    is need for use from the command line or a keytab to be run from condor.
    """
    parser = optparse.OptionParser()
    parser.add_option("-u", "--username", help="user name for the keytab: albert.einstein@LIGO.ORG",default="")
    parser.add_option("-k", "--keytab", help="full path to the keytab\
                        for that user. FOLLOW SECURITY GUIDLINES for permission"
                        ,default="")
    parser.add_option("-c", "--channelname", help="channel name",
                        default="L1:GDS-CALIB_STRAIN")
    parser.add_option("-b", "--boxtime", help="Different time ranges (1 would represent +-0.5 around central time) for displaying the glitch) For Example: [0.5 1 4 16] would give you four differnt images with those different times boxes",
                        default=[0.5,1,2,4])
    parser.add_option("-n", "--nds2name", help="ip address or URL of the nds2\
                        server to use [nds.ligo.caltech.edu]",
                        default="nds.ligo.caltech.edu")
    parser.add_option("-o", "--outpath", help="path to output directory [./png]",
                        default=os.getcwd() + '/png')
    parser.add_option("-j", "--imagepath", help="path to directory for images",
                        default='/home/scoughlin/public_html/GlitchZoo/L_O1_Plots/')
    parser.add_option("-f", "--sampfrequency", help="sample frequency", default=4096)
    parser.add_option("-t", "--centraltime", help="center GPS time (sub second accuracy is available eg. 1089567667.341)")
    parser.add_option("-v", "--verbose", action="store_true", default=False,
                      help="Run verbosely. (Default: False)")
    opts, args = parser.parse_args()

    return opts

# =============================================================================
#
#                                    MAIN
#
# =============================================================================

opts = parse_commandline()

if opts.verbose == True:
        print('output directory:' + opts.outpath)
        print('channel name:'     + opts.channelname)
        print('User Name:'        + opts.username)
        print('KeyTabfile:'       + opts.keytab)
        print('NDS2 file:'        + opts.nds2name)
        print('Central Time:'       + opts.centraltime)
        print('Format:'           + str(opts.sampfrequency))

# Make temporary directory to create the images in
system_call = 'mktemp -d {0}/AAA.XXXXXXXXXX'.format(opts.outpath)
os.system(system_call)

tempname = os.listdir("{0}/".format(opts.outpath))
tempdir = ('{0}/'.format(opts.outpath) + tempname[0])
uniqueid = tempname[0].split('.')[1]
print('unique id is {0}'.format(uniqueid)) 

# Use random folder name as I.D. Tag but first check if this gps time already has a I.D. tag and use that if it does.
idfile =  open(os.getcwd() + '/IDFolder/ID.txt', "a+")
idfile.write('{0} {1} {2}\n'.format(opts.channelname,opts.centraltime,uniqueid))
idfile.close()
# Open file for writing metadata for images
metadata =  open(opts.imagepath + '/metadata.txt', "a+")


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
        plotNormalizedEnergyRange='[0.0 25.5]'  \
        searchTimeRange=64 \
        searchFrequencyRange='[0 inf]' \
        searchQRange='[4.0 64.0]'\n".format(os.getcwd(),opts.nds2name,opts.channelname,opts.centraltime,tempdir,opts.boxtime,opts.sampfrequency))
g.close()

if opts.verbose == True:
        system_call = 'cat {0}/temprun.sh'.format(tempdir)
        os.system(system_call)

system_call = 'source {0}/temprun.sh'.format(tempdir)
os.system(system_call)

os.chdir('{0}'.format(tempdir))

pngnames = os.listdir('./')
strformontage = ""

for iFile in xrange(1,len(pngnames)):

    if opts.channelname == "L1:GDS-CALIB_STRAIN":
        system_call = 'convert {0} -chop 0x35 {1}/L1_{2}_{3}.png'.format(pngnames[iFile],opts.outpath,uniqueid,opts.boxtime[iFile-1])
    elif opts.channelname == "H1:GDS-CALIB_STRAIN":
        system_call = 'convert {0} -chop 0x35 {1}/H1_{2}_{3}.png'.format(pngnames[iFile],opts.outpath,uniqueid,opts.boxtime[iFile-1])
    os.system(system_call)

# Create .csv to upload metadata of images to the project builder
if opts.channelname == "L1:GDS-CALIB_STRAIN":
    metadata.write('{0} 20151016 L1_{1}_{2}.png L1_{3}_{4}.png L1_{5}_{6}.png L1_{7}_{8}.png\n'.format(uniqueid,uniqueid,opts.boxtime[0],uniqueid,opts.boxtime[1],uniqueid,opts.boxtime[2],uniqueid,opts.boxtime[3]))
elif opts.channelname == "H1:GDS-CALIB_STRAIN":
    metadata.write('{0} 20151016 H1_{1}_{2}.png H1_{3}_{4}.png H1_{5}_{6}.png H1_{7}_{8}.png\n'.format(uniqueid,uniqueid,opts.boxtime[0],uniqueid,opts.boxtime[1],uniqueid,opts.boxtime[2],uniqueid,opts.boxtime[3]))
metadata.close()

system_call = 'rm -rf {0}/'.format(tempdir)
os.system(system_call)

if opts.channelname == "L1:GDS-CALIB_STRAIN":
    system_call = 'cp {0}/L1_{1}*.png {2}'.format(opts.outpath,uniqueid,opts.imagepath)
elif opts.channelname == "H1:GDS-CALIB_STRAIN":
    system_call = 'cp {0}/H1_{1}*.png {2}'.format(opts.outpath,uniqueid,opts.imagepath)
os.system(system_call)

system_call = 'rm -rf {0}/*'.format(opts.outpath)
os.system(system_call)

#system_call = 'cp ' + os.getcwd() + '/IDFolder/ID.txt /home/scoughlin/public_html/GlitchZoo/L_ER8_Plots/'
#os.system(system_call)
