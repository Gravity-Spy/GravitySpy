import os,csv,ast
import optparse
import pandas as pd
from panoptes_client import *
#Hold
import pdb
from sqlalchemy.engine import create_engine
import time

############################################################################
###############          MAIN        #######################################
############################################################################

Panoptes.connect()
project = Project.find(slug='zooniverse/gravity-spy')

engine = create_engine('postgresql://scoughlin@localhost:5432/gravityspy')
triggers = pd.read_sql('glitches',engine)
triggers = triggers.loc[triggers.UploadFlag == 0]

labels    = triggers.Label.unique()

startTime = time.time()
for iLabel in labels:
    tmp1 = triggers.loc[(triggers.Label == iLabel)]
    for iSubjectSet in tmp1.subjectset.unique():
        subjectset = SubjectSet.find(int(iSubjectSet))
        tmp = tmp1.loc[tmp1.subjectset == iSubjectSet]
        subjectsToUpload = []
        for index,iSubject in tmp.iterrows():
            subject = Subject()
            subject.links.project = project
            subject.add_location(iSubject['Filename1'])
            subject.add_location(iSubject['Filename2'])
            subject.add_location(iSubject['Filename3'])
            subject.add_location(iSubject['Filename4'])
            subject.metadata['date']          = '20170215'
            subject.metadata['subject_id']    = iSubject['uniqueID']
            subject.metadata['Filename1']     = iSubject['Filename1'].split('/')[-1]
            subject.metadata['Filename2']     = iSubject['Filename2'].split('/')[-1]
            subject.metadata['Filename3']     = iSubject['Filename3'].split('/')[-1].split('/')[-1]
            subject.metadata['Filename4']     = iSubject['Filename4'].split('/')[-1].split('/')[-1]
            subject.metadata['#ML_Posterior_20170215'] = str(iSubject[["1080Lines","1400Ripples","Air_Compressor","Blip","Chirp","Extremely_Loud","Helix","Koi_Fish","Light_Modulation","Low_Frequency_Burst","Low_Frequency_Lines","No_Glitch","None_of_the_Above","Paired_Doves","Power_Line","Repeating_Blips","Scattered_Light","Scratchy","Tomte","Violin_Mode","Wandering_Line","Whistle"]].values.tolist())

            subject.save()
            subjectsToUpload.append(subject)
        subjectset.add(subjectsToUpload)
        SQLCommand = 'UPDATE glitches SET \"UploadFlag\" = \"UploadFlag\" + 1 WHERE \"Label\" = \'{0}\' AND \"subjectset\" = {1}'.format(iLabel,iSubjectSet)
        print(time.time() - startTime)
        engine.execute(SQLCommand)
