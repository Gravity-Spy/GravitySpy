#!/usr/bin/env python

# ---- Import standard modules to the python path.

import os,csv,ast
import optparse
import pandas as pd
from panoptes_client import *
import numpy as np
#Hold
import pdb
from sqlalchemy.engine import create_engine
import time

def parse_commandline():
    """
    Parse the options given on the command-line.
    """
    parser = optparse.OptionParser()
    parser.add_option("--SubjectSet", help="Zooniverse subject set.",type=int,default=6584)
    parser.add_option("--SubjectID", help="Zooniverse subject id.",type=int,default=3775127)
    parser.add_option("--doAdd", action="store_true", default=False,
          help="Add SubjectID to SubjectSet. (Default: False)")
    parser.add_option("--doRemove", action="store_true", default=False,
          help="Remove SubjectID from SubjectSet. (Default: False)")

    opts, args = parser.parse_args()

    return opts

############################################################################
###############          MAIN        #######################################
############################################################################

# Parse commandline arguments
opts = parse_commandline()

Panoptes.connect()
project = Project.find(slug='zooniverse/gravity-spy')

# Remember to set
# export PANOPTES_USERNAME
# export PANOPTES_PASSWORD

subjectset=SubjectSet.find(opts.SubjectSet)
iSubject = Subject.find(opts.SubjectID)
iSubjectMeta = iSubject.raw["metadata"]

subject = Subject()
subject.links.project = project
#subject.add_location(iSubjectMeta['Filename1'])
#subject.add_location(iSubjectMeta['Filename2'])
#subject.add_location(iSubjectMeta['Filename3'])
#subject.add_location(iSubjectMeta['Filename4'])
subject.metadata['date']          = iSubjectMeta['date']
subject.metadata['subject_id']    = iSubjectMeta['subject_id']
subject.metadata['Filename1']     = iSubjectMeta['Filename1'].split('/')[-1]
subject.metadata['Filename2']     = iSubjectMeta['Filename2'].split('/')[-1]
subject.metadata['Filename3']     = iSubjectMeta['Filename3'].split('/')[-1].split('/')[-1]
subject.metadata['Filename4']     = iSubjectMeta['Filename4'].split('/')[-1].split('/')[-1]
subject.save()

if opts.doAdd:
    subjectset.add(subject)

if opts.doRemove:
    for iSubject in subjectset.subjects():
        if iSubject.raw["metadata"]['Filename1'] == subject.metadata['Filename1']:
            subject_id = iSubject.id 
            subjectset.remove(subject_id)
