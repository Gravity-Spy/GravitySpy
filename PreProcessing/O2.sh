#!/bin/bash

source ~/.bash_profile
cd /home/scoughlin/O2/automated/GravitySpy//pyomega/PreProcessing/

/home/scoughlin/opt/O2/GravitySpy/bin/python /home/scoughlin/O2/automated/GravitySpy//pyomega/PreProcessing/gettriggers.py --detector L1 --outDir /home/scoughlin/public_html/O2/GravitySpy/Format/Feb152017/images/ML/ --uniqueID --SNR 7.5 --pathToIni /home/scoughlin/O2/automated/GravitySpy//pyomega/Production/wini.ini --pathToExec /home/scoughlin/O2/automated/GravitySpy//pyomega//src/wscan.py --pathToModel /home/scoughlin/O2/automated/GravitySpy//pyomega/src/ML/trained_model/ --PostgreSQL --upload --pathToExecUpload /home/scoughlin/O2/automated/GravitySpy/pyomega/PostProcessing/upload.py 
