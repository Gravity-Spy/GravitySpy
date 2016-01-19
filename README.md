# GravitySpy
This repo contains code for the GravitySpy Citizen Science project.

RUN source /home/detchar/opt/gwpysoft/etc/gwpy-user-env.sh
RUN ligo-proxy-init albert.einstein
RUN kinit albert.einstein@LIGO.ORG
1.) RUN python read_omicron_triggers.py --gpsStart 1127700000 --gpsEnd 1127701500 --detector H1 or L1
(Pick the same detector as the cluster you are on)

Here is the current help information. If you update the function with new flags or change the meaning of a flag please edit this part of the README.

Options:
  -h, --help            show this help message and exit
  --boxtime=BOXTIME     Different time ranges (1 would represent +-0.5 around
                        central time) for displaying the glitch) For Example:
                        '[0.5 1 4 16]' would give you four differnt images
                        with those different times boxes [Default:[0.5,1,2,4]
  --channelname=CHANNELNAME
                        channel name [Default:GDS-CALIB_STRAIN]
  --colorMap=COLORMAP   What color would you like the omegascans to be?
                        [Default: jet] [Options: bone, copper,hot]
  --detector=DETECTOR   detector name [L1 or H1]. [No Default must supply]
  --gpsStart=GPSSTART   gps Start Time of Query for meta data and omega scans.
                        [No Default must supply]
  --gpsEnd=GPSEND       gps End Time [No Default must supply]
  --freqHigh=FREQHIGH   upper frequency bound cut off for the glitches.
                        [Default: 2048]
  --freqLow=FREQLOW     lower frequency bound cut off for the glitches.
                        [Default: 10]
  --imagepath=IMAGEPATH
                        path to directory for images NO matter what path you
                        select {detectorname}_{gpsStart}_{gpsEnd} will be
                        added as a subdirectory [Default
                        ~/public_html/GravitySpy/]
  --nds2name=NDS2NAME   ip address or URL of the nds2
                        server to use [Default: nds.ligo.caltech.edu]
  --normalizedSNR=NORMALIZEDSNR
                        SNR Normalization value. [Default: 25.5]
  --maxSNR=MAXSNR       This flag gives you the option to supply a upper limit
                        to the SNR of the glitches [Default: 0 (i.e. none)]
  --outpath=OUTPATH     path to output directory [Default: ./png] (This is
                        where the images are created but you will not have to
                        go in this folder as the iamges are transferred to
                        your public_html imagepath folder)
  --runlocal=RUNLOCAL   run locally (running as a condor job has not been set
                        up yet)
  --sampfrequency=SAMPFREQUENCY
                        sample frequency for omegascan iamges [Default: 4096]
  --SNR=SNR             Lower bound SNR Threshold for omicron triggers, by
                        default there is no upperbound SNR unless supplied
                        throught the --maxSNR flag. [Default: 6]
  --verbose             Run verbosely. (Default: False)
