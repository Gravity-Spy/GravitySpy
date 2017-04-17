#!/usr/bin/python

# Copyright (C) 2013 Michael Coughlin
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

import os, sys
import numpy as np
import optparse
from sqlalchemy.engine import create_engine
import pandas as pd

__author__ = "Michael Coughlin <michael.coughlin@ligo.org>"
__version__ = 1.0
__date__    = "9/22/2013"

# =============================================================================
#
#                               DEFINITIONS
#
# =============================================================================

def parse_commandline():
    """@parse the options given on the command-line.
    """
    parser = optparse.OptionParser(usage=__doc__,version=__version__)

    parser.add_option("-d", "--detector", help="Detector.",default ="H1")
    parser.add_option("-b", "--database", help="Database (O1GlitchClassification,classification,glitches).", default="O1GlitchClassification")
    parser.add_option("-l", "--label", help="Label.",default ="Blip")
    parser.add_option("-o", "--outfile", help="Output file.",default ="test.csv")

    parser.add_option("-v", "--verbose", action="store_true", default=False,
                      help="Run verbosely. (Default: False)")

    opts, args = parser.parse_args()

    # show parameters
    if opts.verbose:
        print >> sys.stderr, ""
        print >> sys.stderr, "running network_eqmon..."
        print >> sys.stderr, "version: %s"%__version__
        print >> sys.stderr, ""
        print >> sys.stderr, "***************** PARAMETERS ********************"
        for o in opts.__dict__.items():
          print >> sys.stderr, o[0]+":"
          print >> sys.stderr, o[1]
        print >> sys.stderr, ""

    return opts

# Parse command line
opts = parse_commandline()

detector = opts.detector
database = opts.database
label = opts.label
outfile = opts.outfile

engine = create_engine('postgresql://{0}:{1}@gravityspy.ciera.northwestern.edu:5432/gravityspy'.format(os.environ['QUEST_SQL_USER'],os.environ['QUEST_SQL_PASSWORD']))
tmp = pd.read_sql(database,engine)
tmp = tmp.loc[tmp.ifo == detector]
tmp = tmp.loc[tmp.Label == label]
columns=["peakGPS","peak_frequency", "snr"]
tmp.to_csv(outfile,columns=columns,index=False,header=True)

print "GPS: %.0f - %.0f"%(tmp["peakGPS"].min(),tmp["peakGPS"].max())
