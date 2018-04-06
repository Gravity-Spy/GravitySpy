#!/bin/bash

# enable strict test flags

coverage run ./setup.py test --addopts gravityspy/tests/
coverage run --append `which wscan` --help
coverage run --append `which gettriggers` --help
#coverage run --append `which wscan` --inifile Production/wini.ini --eventTime 1127700030.877928972 --outDir $HOME/public_html/GravitySpy/Test/ --uniqueID --ID 123abc1234 --HDF5 --runML
