#!/bin/bash

# enable strict test flags
if [ "$STRICT" = true ]; then
    _strict="-x --strict"
else
    _strict=""
fi

coverage run -m py.test -v -r s ${_strict} gravityspy/
coverage run --append `which wscan` --help
coverage run --append `which testgpu` --help
coverage run --append `which trainingset_psql` --help
coverage run --append `which modify_subject` --help
coverage run --append `which filter_psql` --help
coverage run --append `which trainmodel` --help
coverage run --append `which labelimages` --help
coverage run --append `which gettriggers` --help
coverage run --append `which gspysearch` --help
#coverage run --append `which wscan` --inifile Production/wini.ini --eventTime 1127700030.877928972 --outDir $HOME/public_html/GravitySpy/Test/ --uniqueID --ID 123abc1234 --HDF5 --runML
