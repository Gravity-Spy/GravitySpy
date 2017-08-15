#!/bin/bash

source ~/.bash_profile

cd ~/Project/GravitySpy/PreProcessing

gettriggers --outDir /home/scoughlin/public_html/O2/GravitySpy/Format/Feb152017/images/ML/ --uniqueID --SNR 7.5 --pathToIni /home/scoughlin/O2/automated/GravitySpy//pyomega/Production/wini.ini --pathToModel /home/scoughlin/Project/GravitySpy/pyomega/ML/trained_model/ --PostgreSQL --upload 
