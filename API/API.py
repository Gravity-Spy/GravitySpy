from panoptes_client import *

import pandas as pd
import ast
import numpy as np
import os, sys
import ast
import pdb
import datetime
import collections
import operator

########################
####### Functions ######
#########################

pathToFiles = '/home/scoughlin/O2/Test/GravitySpy/API/'

# This function translates the string answer to an integer value.
def extract_choiceINT(x):
    try:
        return label_dict[x]
    except:
        return -1

# This is the current version of the integer to string dict
label_dict = {
'50HZ':0,'RCMPRSSR50HZ':0,'AIRCOMPRESSOR50HZ':0,
'BLP':1,'BLIP':1,
'CHRP':2,'CHIRP':2,
'XTRMLLD':3,'EXTREMELYLOUD':3,
'HLX':4,'HELIX':4,
'KFSH':5,'KOIFISH':5,
'45MHZLGHTMDLTN':6,'LGHTMDLTN':6,'LIGHTMODULATION':6,
'LWFRQNCBRST':7,'LOWFREQUENCYBURST':7,
'LWFRQNCLN':8,'LOWFREQUENCYLINE':8,
'NNFTHBV':9,'NONEOFTHEABOVE':9,
'NGLTCH':10,'DNTSGLTCH':10,'NOGLITCH':10,
'PRDDVS':11,'PAIREDDOVES':11,
'60HZPWRLN':12,'60HZPWRMNS':12,'PWRLN60HZ':12,'POWERLINE60HZ':12,
'RPTNGBLPS':13,'REPEATINGBLIPS':13,
'SCTTRDLGHT':14,'SCATTEREDLIGHT':14,
'SCRTCH':15,'SCRATCHY':15,
'TMT':16,'TOMTE':16,
'VLNHRMNC500HZ':17,'VLNMDHRMNC500HZ':17, 'HRMNCS':17,'VIOLINMODEHARMONIC500HZ':17,
'WNDRNGLN':18,'WANDERINGLINE':18,
'WHSTL':19,'WHISTLE':19
}

#This function generically flatten a dict
def flatten(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        if v and (type(v) is list):
            v = v[0]
        new_key = parent_key + sep + k if parent_key else k
        try:
            items.extend(flatten(v, new_key, sep=sep).items())
        except:
            items.append((new_key, v))
    return dict(items)

# Load lastID that was parsed
lastID = pd.read_hdf('{0}/GravitySpy.h5'.format(pathToFiles),columns=['id']).max().iloc[0]

print(lastID)
# Connect to panoptes and query all classifications done on project 1104 (i.e. GravitySpy)
Panoptes.connect()

# Created empty list to store the previous classifications
classificationsList = []

# Query the last 100 classificaitons made (this is the max allowable)
allClassifications = Classification.where(scope='project',project_id='1104',last_id='{0}'.format(lastID),page_size='100')

# Loop until no more classifications
for iN in range(0,allClassifications.object_count):
    try:
        classification = allClassifications.next()
    except:
        break

    # Generically with no hard coding we want to parse all aspects of the
    # classification metadata. This will ease any changes on the API side and
    # any changes to the metadata on our side.

    try:
        classificationsList.append(flatten(classification.raw))
    except:
        continue

if not classificationsList:
    raise ValueError('No New Classifications')

# Now we want to make a panda data structure with all this information
classifications = pd.DataFrame(classificationsList)
classifications = classifications.convert_objects(convert_numeric=True) 
classifications.created_at = pd.to_datetime(classifications.created_at,infer_datetime_format=True)
classifications.metadata_started_at = pd.to_datetime(classifications.metadata_started_at,infer_datetime_format=True)
classifications.metadata_finished_at = pd.to_datetime(classifications.metadata_finished_at,infer_datetime_format=True)

# At this point we have generically parsed the classification of the user. The label given by the parser is a string and for the purposes of our exercise converting these strings to ints is useful. After we will append the classifications to the old classifications and save. Then we tackle the information about the image that was classified. 

classifications['annotations_value_choiceINT'] = classifications['annotations_value_choice'].apply(extract_choiceINT)
classifications = classifications.select_dtypes(exclude=['object'])
classifications = classifications[['created_at','id','links_project','links_subjects','links_user','links_workflow','metadata_finished_at','metadata_started_at','metadata_workflow_version','annotations_value_choiceINT']]
classifications.loc[classifications.links_user.isnull(),'links_user'] = 0
classifications.links_user = classifications.links_user.astype(int)
classifications.to_hdf('{0}/GravitySpy.h5'.format(pathToFiles),'classifications',mode='a',format='table',append=True,data_columns=['id','links_subjects','links_user','links_workflow','annotations_value_choiceINT'])

"""
# Load subjects info in case this is an image that has been previously labeled and getting the metadata information is superfluous
imagestmp = pd.read_hdf('{0}/images.h5'.format(pathToFiles)) 

# Only get new subjects because otherwise we have all the image info we need.
newSubjects = []
for iSubject in classifications.loc[~classifications['links_subjects'].isin(imagestmp.links_subjects.tolist()),'links_subjects'].unique(): 
    tmpSubject = Subject.find(iSubject)
    newSubjects.append(flatten(tmpSubject.raw))

images = pd.DataFrame(newSubjects)
images = images.loc[~images.links_subject_sets.isnull()]

# Check if gold (In an effort to improve table storage. These are now integers.
#     Testing (only ML Label) = 1
#     Gold    = 2
#     Retired = 3

def determine_type(x):
    if x == 'Gold':
        return 2
    else:
        return 1

# The label on the gold are in lowercase so need to make them uppercase
def determine_label(x):
    return label_dict[str(x).upper().translate(None,'AEIOUY ()')]

# If metadata_#Type is not a field then all images are testing
try:
    images['metadata_Type'] = images['metadata_#Type'].apply(determine_type)
except:
    images['metadata_Type'] = 1

# Again if there is not field for metadata_#Label then all are testing and no gold images
try:
    images.loc[images['metadata_#Label'].isnull(),'metadata_#Label'] = -1
except:
    images['metadata_#Label'] = -1

# For all images that have a label determine what that label is in terms of its integer.
images.loc[images['metadata_#Label'] != -1,'metadata_#Label'] = images.loc[images['metadata_#Label'] != -1,'metadata_#Label'].apply(determine_label)

# For those that are testing images, take their unicode string make it a string and attempt to determine the max value from the vector (which is equivalent to the ML label)
images.loc[images['metadata_#Label'] == -1,'metadata_#ML_Posterior'] = images.loc[images['metadata_#Label'] == -1,'metadata_#ML_Posterior'].apply(str)
def check_ML(x):
    try:
        return np.argmax(ast.literal_eval(x))
    except:
        return -1
images.loc[images['metadata_#Label'] == -1,'metadata_#ML_Posterior'] = images.loc[images['metadata_#Label'] == -1,'metadata_#ML_Posterior'].apply(check_ML)
# For those that are golden images mark them as having no #ML_Posterior
images.loc[images['metadata_#Label'] != -1,'metadata_#ML_Posterior'] = -1

# Make all floats and ints into floats and ints
images = images.convert_objects(convert_numeric=True)
images.rename(columns={'id':'links_subjects'},inplace=True)
images = images[['links_subjects','links_collections','links_project','links_subject_sets','metadata_#Label','metadata_#ML_Posterior','metadata_date','metadata_Type','metadata_subject_id']]
# Turna  unicode into a string
images.metadata_subject_id = images.metadata_subject_id.apply(str)
#append this to the image h5 table (This file has mostly ints and flaots making it very space and search efficient.
images.to_hdf('{0}/images.h5'.format(pathToFiles),'classifications',mode='a',format='table',append=True,data_columns=['links_subjects','metadata_Type'])

images = pd.read_hdf('{0}/images.h5'.format(pathToFiles))

# Merge classificaitons and images
image_and_classification = classifications.merge(images)

c =  max(label_dict.iteritems(), key=operator.itemgetter(1))[1] + 1

def make_conf_matrices(x):
    return np.zeros((c,c))

confusion_matrices = pd.DataFrame(classifications.links_user.unique(),columns=['userID'])
confusion_matrices['confusion_matrices'] = confusion_matrices.userID.apply(make_conf_matrices)
confusion_matrices['currentworkflow'] = 1
confusion_matrices['Level2'] = 0
confusion_matrices['Level3'] = 0
confusion_matrices['Level4'] = 0
confusion_matrices['Level5'] = 0

alpha = .7*np.ones(c)

def update_conf_matrix(x):
    # Define what classes belongt o what levels
    promotion_1 = [1,19]
    promotion_2 = [17, 12, 5, 1, 19]
    promotion_3 = [5, 7, 10, 12, 14, 2,  1, 19, 17]
    promotion_4 = [ 13, 16,  3,  7,  6, 15,  4,  1,  5, 18, 11,  8, 14, 12,  0, 10,19, 17,  2]
    # Increase the matrix at [true_label,userlabel]
    confusion_matrices.loc[confusion_matrices.userID == x.links_user,'confusion_matrices'].iloc[0][x['metadata_#Label'],x['annotations_value_choiceINT']] += 1
    for iWorkflow in range(1,5):
        if (confusion_matrices.loc[confusion_matrices.userID == x.links_user,'currentworkflow'].iloc[0] == iWorkflow) and np.diag(confusion_matrices.loc[confusion_matrices.userID == x.links_user,'confusion_matrices'].iloc[0])[locals()['promotion_' + str(iWorkflow)]].all():
            # Take normalized diagonal.
            conf_divided,a1,a2,a3 = np.linalg.lstsq(np.diag(np.sum(confusion_matrices.loc[confusion_matrices.userID == x.links_user,'confusion_matrices'].iloc[0],axis=1)),confusion_matrices.loc[confusion_matrices.userID == x.links_user,'confusion_matrices'].iloc[0])
            alphaTmp = np.diag(conf_divided)
            if (confusion_matrices.loc[confusion_matrices.userID == x.links_user,'currentworkflow'].iloc[0] == iWorkflow) and ((alphaTmp[locals()['promotion_' + str(iWorkflow)]] >= alpha[locals()['promotion_' + str(iWorkflow)]]).all()):
                confusion_matrices.loc[confusion_matrices.userID == x.links_user,'currentworkflow'] = iWorkflow + 1
                confusion_matrices.loc[confusion_matrices.userID == x.links_user,'Level{0}'.format(iWorkflow +1)] = x['id']

image_and_classification = image_and_classification.sort_values('created_at')
image_and_classification.loc[image_and_classification.metadata_Type == 2,['links_user','metadata_#Label','annotations_value_choiceINT','id']].apply(update_conf_matrix,axis=1)
for iWorkflow in range(1,6):
    print('Level {0}: {1}'.format(iWorkflow,len(confusion_matrices.loc[confusion_matrices.currentworkflow == iWorkflow])))
"""
