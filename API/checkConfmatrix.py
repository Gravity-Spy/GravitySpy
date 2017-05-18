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

from sqlalchemy.engine import create_engine
from pyomega.API.getLabelDict import getAnswers
from pyomega.API import getGoldenImages
from scipy.sparse import coo_matrix

def levelDict(x):
    return workflowLevelDict[x]

engine = create_engine('postgresql://{0}:{1}@gravityspy.ciera.northwestern.edu:5432/gravityspy'.format(os.environ['QUEST_SQL_USER'],os.environ['QUEST_SQL_PASSWORD']))

# Load clasificaitons and current user Status from DB
classifications = pd.read_sql('classifications',engine) 
userStatus = pd.read_sql('userStatus', engine)

workflowGoldenSetDict = getGoldenImages.getGoldenSubjectSets('1104')
workflowOrder = [int(str(i)) for i in Project.find('1104').raw['configuration']['workflow_order']]
levelWorkflowDict = dict(enumerate(workflowOrder))
workflowLevelDict = dict((v, k + 1) for k,v in levelWorkflowDict.iteritems())
print 'Retrieving Golden Images from Zooniverse API'
goldenDF = getGoldenImages.getGoldenImagesAsInts(workflowGoldenSetDict)

# Filter classifications
classifications.loc[classifications.links_workflow == 3063, 'links_workflow'] = 2360
classifications = classifications.loc[classifications.links_workflow.isin(workflowOrder)]
classifications['Level'] = classifications.links_workflow.apply(levelDict)
classifications = classifications.loc[classifications.annotations_value_choiceINT != -1]
classifications = classifications.loc[classifications.links_user != 0]

# Initialize empty user DB
userStatusInit = pd.DataFrame({'userID' : classifications.groupby('links_user').Level.max().index.tolist(), 'workflow' : classifications.groupby('links_user').Level.max().tolist()})

# Retrieve Answers
answers = getAnswers('1104')
answersDictRev =  dict(enumerate(sorted(answers[2360].keys())))
answersDict = dict((str(v),k) for k,v in answersDictRev.iteritems())

numClasses = max(answersDict.iteritems(), key=operator.itemgetter(1))[1] + 1

image_and_classification = classifications.merge(goldenDF)

# Merge classificaitons and images
test = image_and_classification.groupby(['links_user','annotations_value_choiceINT','GoldLabel'])
test = test.count().links_subjects.to_frame().reset_index()

promotion_Level1 = [answersDict[iAnswer] for iAnswer in answers[1610].keys() if iAnswer not in['NONEOFTHEABOVE']]
promotion_Level2 = [answersDict[iAnswer] for iAnswer in answers[1934].keys() if iAnswer not in['NONEOFTHEABOVE']]
promotion_Level3 = [answersDict[iAnswer] for iAnswer in answers[1935].keys() if iAnswer not in['NONEOFTHEABOVE']]
promotion_Level4 = [answersDict[iAnswer] for iAnswer in answers[2360].keys() if iAnswer not in['NONEOFTHEABOVE']]

alpha = .7*np.ones(numClasses)

for iUser in test.groupby('links_user'):
    columns = iUser[1].annotations_value_choiceINT        
    rows = iUser[1]['GoldLabel']
    entry = iUser[1]['links_subjects']
    tmp = coo_matrix((entry,(rows,columns)),shape=(numClasses, numClasses))
    conf_divided,a1,a2,a3 = np.linalg.lstsq(np.diagflat(tmp.sum(axis=1)), tmp.todense())
    alphaTmp = np.diag(conf_divided)
    userCurrentLevel = userStatusInit.loc[userStatusInit.userID == iUser[0], 'currentworkflow']
    if (alphaTmp[promotion_Level1] > alpha[promotion_Level1]) and (userCurrentLevel < 2):
        userStatusInit.loc[userStatusInit.userID == iUser[0], 'currentworkflow'] = 2

    if (alphaTmp[promotion_Level2] > alpha[promotion_Level2]) and (userCurrentLevel < 3):
        userStatusInit.loc[userStatusInit.userID == iUser[0], 'currentworkflow'] = 3

    if (alphaTmp[promotion_Level3] > alpha[promotion_Level3]) and (userCurrentLevel < 4):
        userStatusInit.loc[userStatusInit.userID == iUser[0], 'currentworkflow'] = 4

    if (alphaTmp[promotion_Level4] > alpha[promotion_Level4]) and (userCurrentLevel < 5):
        userStatusInit.loc[userStatusInit.userID == iUser[0], 'currentworkflow'] = 5


tmp = userStatusInit.merge(userStatus,how='outer')
tmp = tmp.fillna(0)
tmp = tmp.astype(int)
updates = tmp.loc[tmp.workflow > tmp.currentworkflow]
tmp.loc[tmp.workflow > tmp.currentworkflow, 'currentworkflow'] = tmp.loc[tmp.workflow > tmp.currentworkflow, 'workflow']
userStatus = tmp[['userID', 'currentworkflow']]

# Now update user settings
Panoptes.connect()
project = Project.find(slug='zooniverse/gravity-spy')

def updateSettings(x):
    user = User.find(x.userID)
    new_settings = {"workflow_id": "{0}".format(levelWorkflowDict[x.workflow - 1])}
    print(user)
    print(new_settings)
    ProjectPreferences.save_settings(project=project, user=user, settings=new_settings) 

updates.apply(updateSettings,axis=1)

# save new user Status
userStatus[['userID', 'currentworkflow']].to_sql('userStatus', engine, index=False, if_exists='append', chunksize=100)
