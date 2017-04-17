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
import subprocess
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

    parser.add_option("-d", "--detector", help="What IFO am I running from?",default ="H1")
    parser.add_option("-b", "--database", help="Database (O1GlitchClassification,classification,glitches).", default="glitches")
    parser.add_option("-o", "--outdir", help="Output directory.",default ="./TrainingSet")

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

def cp_file(Filename,ifo,ThisTrainingFolder):
    if Filename == None: return

    FilenamePNG = Filename.split("/")[-1]
    outfile = os.path.join(ThisTrainingFolder,FilenamePNG)
    if os.path.isfile(outfile): return

    if ifo == "H1":
        hostpath = "ldas-pcdev2.ligo-wa.caltech.edu"
    elif ifo == "L1":
        hostpath = "ldas-pcdev2.ligo-la.caltech.edu"

    filepath = "%s:%s"%(hostpath,Filename)
    outfilepath = "%s/%s"%(ThisTrainingFolder,FilenamePNG)

    if currentifo == ifo:
        cp_command = "cp %s %s"%(Filename,outfilepath)
    else:
        cp_command = "gsiscp %s %s"%(filepath,outfilepath)
    os.system(cp_command)

# Parse command line
opts = parse_commandline()

currentifo = opts.detector
TrainingFolder = opts.outdir
if not os.path.isdir(TrainingFolder):
    os.mkdir(TrainingFolder) 
database = opts.database

engine = create_engine('postgresql://{0}:{1}@gravityspy.ciera.northwestern.edu:5432/gravityspy'.format(os.environ['QUEST_SQL_USER'],os.environ['QUEST_SQL_PASSWORD']))
tmp = pd.read_sql(database,engine)

for label in tmp.Label.unique():
    ThisTrainingFolder = os.path.join(TrainingFolder,label)
    if not os.path.isdir(ThisTrainingFolder):
        os.mkdir(ThisTrainingFolder)

    tmp2 = tmp.loc[tmp.Label == label]
    tmp2 = tmp.loc[tmp.ImageStatus == "Training"]
    for index,row in tmp2.iterrows():

        cp_file(row.Filename1,row.ifo,ThisTrainingFolder)
        cp_file(row.Filename2,row.ifo,ThisTrainingFolder)
        cp_file(row.Filename3,row.ifo,ThisTrainingFolder)
        cp_file(row.Filename4,row.ifo,ThisTrainingFolder)

