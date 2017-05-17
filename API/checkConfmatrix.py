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

engine = create_engine('postgresql://{0}:{1}@gravityspy.ciera.northwestern.edu:5432/gravityspy'.format(os.environ['QUEST_SQL_USER'],os.environ['QUEST_SQL_PASSWORD']))

classifications = pd.read_sql('classifications',engine) 

workflowGoldenSetDict = getGoldenImages.getGoldenSubjectSets('1104')
print 'Retrieving Golden Images from Zooniverse API'
goldenDF = getGoldenImages.getGoldenImagesAsInts(workflowGoldenSetDict)

# Filter classifications
classifications = classifications.loc[classifications.links_workflow.isin([1610,1934,1935,2360,3063])]
classifications = classifications.loc[classifications.annotations_value_choiceINT != -1]
classifications = classifications.loc[classifications.links_user != 0]

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

for iUser in test.groupby('links_user'):
    columns = iUser[1].annotations_value_choiceINT        
    rows = iUser[1]['GoldLabel']
    entry = iUser[1]['links_subjects']
    tmp = coo_matrix((entry,(rows,columns)),shape=(numClasses, numClasses))
    conf_divided,a1,a2,a3 = np.linalg.lstsq(np.diagflat(tmp.sum(axis=1)), tmp.todense())
    print tmp.sum(axis=0)
    print tmp.sum(axis=1)
    # print np.diag(conf_divided)[promotion_Level4]


alpha = .7*np.ones(c)
Panoptes.connect()
project = Project.find(slug='zooniverse/gravity-spy')

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
                user = User.find(x.links_user)
                new_settings = {"workflow_id": "{0}".format(leveltoworkflow[iWorkflow + 1])}
                print(user)
                print(new_settings)
                ProjectPreferences.save_settings(project=project, user=user, settings=new_settings)

image_and_classification = image_and_classification.sort_values('created_at')
lastID = pd.read_csv('{0}/lastIDPromotion.csv'.format(pathToFiles))['lastID'].iloc[0]
newlastID = image_and_classification['id'].max()
print('lastID of Promotion ' + str(lastID))
image_and_classification = image_and_classification.loc[image_and_classification['id']>lastID]
image_and_classification.loc[image_and_classification.metadata_Type == 2,['links_user','metadata_#Label','annotations_value_choiceINT','id']].apply(update_conf_matrix,axis=1)
for iWorkflow in range(1,6):
    print('Level {0}: {1}'.format(iWorkflow,len(confusion_matrices.loc[confusion_matrices.currentworkflow == iWorkflow])))

confusion_matrices.to_pickle('{0}/confusion_matrices.pkl'.format(pathToFiles))
pd.DataFrame({'lastID':newlastID},index=[0]).to_csv(open('{0}/lastIDPromotion.csv'.format(pathToFiles),'w'),index=False)
print('New lastID of Promotion ' + str(newlastID))
