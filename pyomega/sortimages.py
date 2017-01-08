import os,csv,ast
import optparse
import pandas as pd
from panoptes_client import *
#Hold
import pdb

Panoptes.connect()
project = Project.find(slug='zooniverse/gravity-spy')

subject = Subject()
subject.links.project = project
subject.add_location()
subject.add_location('{0}/{1}/Beginner/{2}'.format(dataPath,Type,iXX[3]))
subject.add_location('{0}/{1}/Beginner/{2}'.format(dataPath,Type,iXX[4]))
subject.add_location('{0}/{1}/Beginner/{2}'.format(dataPath,Type,iXX[5]))
subject.metadata['date']          = '20170108'
subject.metadata['subject_id']    = iXX[1]
subject.metadata['Filename1']     = iXX[2]
subject.metadata['Filename2']     = iXX[3]
subject.metadata['Filename3']     = iXX[4]
subject.metadata['Filename4']     = iXX[5]
subject.metadata['#ML_Posterior'] = iXX[6]
subject.save()
tmp.append(subject)
subjectset.add(tmp)
