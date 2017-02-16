#!/bin/bash

PATH=/home/scoughlin/opt/postgres/bin:$PATH
export PATH
LD_LIBRARY_PATH=/home/scoughlin/opt/postgres/lib
export LD_LIBRARY_PATH
export LIGO_DATAFIND_SERVER=ldrslave.ldas.ligo-la.caltech.edu:80

#PATH=$HOME/software/GWpyAndML/bin:$PATH:$HOME/bin
. /home/scoughlin/opt/O2/GravitySpy/bin/activate
source ~/software/lscsoft/lal/etc/lal-user-env.sh
source ~/software/lscsoft/lalframe/etc/lalframe-user-env.sh

cd /home/scoughlin/O2/Format/Feb152017/GravitySpy/pyomega/PreProcessing/

#/home/scoughlin/opt/O2/GravitySpy/bin/python /home/scoughlin/O2/Format/Feb152017/GravitySpy/pyomega/PreProcessing/gettriggers.py --detector L1 --outDir /home/scoughlin/public_html/O2/GravitySpy/Format/Feb152017/images/ML/ --uniqueID --SNR 7.5 --pathToIni /home/scoughlin/O2/GravitySpy/pyomega/Production/wini.ini --pathToExec /home/scoughlin/O2/Format/Feb152017/GravitySpy/pyomega//src/wscan.py --pathToModel /home/scoughlin/O2//Format/Feb152017/GravitySpy/pyomega/src/ML/trained_model/ --PostgreSQL
