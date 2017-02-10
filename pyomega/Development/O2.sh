#!/bin/bash

cd /home/scoughlin/O2/GravitySpy/pyomega/Development
STARTTIME=$(tail -1 StartTime.txt)
ENDTIME=$(/bin/lalapps_tconvert 1 hours ago)
echo ${ENDTIME} >>StartTime.txt
/home/scoughlin/opt/O2/GravitySpy/bin/python gettriggers.py --gpsStart ${STARTTIME} --gpsEnd ${ENDTIME} --detector L1 --outDir /home/scoughlin/public_html/O2/GravitySpy/images/ML/ --uniqueID --SNR 7.5
condor_submit_dag gravityspy_${STARTTIME}_${ENDTIME}.dag 
