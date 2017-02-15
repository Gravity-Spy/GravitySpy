#!/bin/bash

cd /home/scoughlin/O2/Format/Feb152017/GravitySpy/pyomega/PreProcessing/

/home/scoughlin/opt/O2/GravitySpy/bin/python /home/scoughlin/O2/Format/Feb152017/GravitySpy/pyomega/PreProcessing/gettriggers.py --detector L1 --outDir /home/scoughlin/public_html/O2/GravitySpy/Format/Feb152017/images/ML/ --uniqueID --SNR 7.5 --pathToIni /home/scoughlin/O2/GravitySpy/pyomega/Production/wini.ini --pathToExec /home/scoughlin/O2/GravitySpy/pyomega/src/wscan.py --pathToModel /home/scoughlin/O2/GravitySpy/pyomega/src/ML/trained_model/ --gpsStart 1161935086 --PostgreSQL
