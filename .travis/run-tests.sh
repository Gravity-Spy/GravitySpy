#!/bin/bash

# enable strict test flags
if [ "$STRICT" = true ]; then
    _strict="-x --strict"
else
    _strict=""
fi

coverage run -m py.test -v -r s ${_strict} pyomega/
coverage run --append `which wscan` --help
#coverage run --append `which wscan` --inifile Production/wini.ini --eventTime 1127700030.877928972 --outDir $HOME/public_html/GravitySpy/Test/ --uniqueID --ID 123abc1234 --HDF5 --runML
