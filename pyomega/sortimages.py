import os,csv,ast
import optparse
import pandas as pd
from panoptes_client import *
#Hold
import pdb

def parse_commandline():
    """
    Parse the options given on the command-line.
    """
    parser = optparse.OptionParser()
    parser.add_option("--triggerFile", help="Trigger file with information for upload to Gravity Spy server")
    opts, args = parser.parse_args()

    return opts

############################################################################
###############          MAIN        #######################################
############################################################################

# Parse commandline arguments
opts = parse_commandline()

Panoptes.connect()
project = Project.find(slug='zooniverse/gravity-spy')

triggers = pd.read_hdf('{0}'.format(opts.triggerFile))
triggers = triggers.loc[triggers.UploadFlag == 0]

subjectsToUpload = []

labels    = triggers.Label.unique()
workflows = triggers.workflow.unique()

for iLabel in labels:
    for iWorkflow in workflows:
        tmp = triggers.loc[(triggers.Label == iLabel) & (triggers.workflow == iWorkflow)]
        subjectset = SubjectSet.find(tmp.subjectset.iloc[0])
        for index,iSubject in tmp.iterrows():
            subject = Subject()
            subject.links.project = project
            subject.add_location(iSubject['Filename1'])
            subject.add_location(iSubject['Filename2'])
            subject.add_location(iSubject['Filename3'])
            subject.add_location(iSubject['Filename4'])
            subject.metadata['date']          = '20170108'
            subject.metadata['subject_id']    = iSubject['uniqueID']
            subject.metadata['Filename1']     = iSubject['Filename1'].split('/')[-1]
            subject.metadata['Filename2']     = iSubject['Filename2'].split('/')[-1]
            subject.metadata['Filename3']     = iSubject['Filename3'].split('/')[-1].split('/')[-1]
            subject.metadata['Filename4']     = iSubject['Filename4'].split('/')[-1].split('/')[-1]
            subject.metadata['#ML_Posterior'] = str(iSubject.values[0:20].tolist())
            subject.save()
            subjectsToUpload.append(subject)
        subjectset.add(subjectsToUpload)
        triggers.loc[(triggers.Label == iLabel) & (triggers.workflow == iWorkflow),'uploadFlag'] = 1
triggers.to_hdf('ML_GSpy_upload.h5','gspy_ML_classification')
