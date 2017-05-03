#!/bin/bash

# enable strict test flags
if [ "$STRICT" = true ]; then
    _strict="-x --strict"
else
    _strict=""
fi

coverage run -m py.test -v -r s ${_strict} pyomega/
coverage run --append `which wscan` --help
