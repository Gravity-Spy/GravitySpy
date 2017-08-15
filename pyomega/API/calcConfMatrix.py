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

def main():
    def levelDict(x):
        return workflowLevelDict[x]

    engine = create_engine('postgresql://{0}:{1}@gravityspy.ciera.northwestern.edu:5432/gravityspy'.format(os.environ['QUEST_SQL_USER'],os.environ['QUEST_SQL_PASSWORD']))

    # Load classifications, current user DB status and golden images from DB
    classifications = pd.read_sql('SELECT links_user, links_subjects, links_workflow, "annotations_value_choiceINT" FROM classificationsdev', engine)
    userStatus = pd.read_sql('userStatus', engine)
    goldenDF = pd.read_sql('goldenimages', engine)

    # Obtain workflow order
    workflowOrder = [int(str(i)) for i in Project.find('1104').raw['configuration']['workflow_order']]
    levelWorkflowDict = dict(enumerate(workflowOrder))
    workflowLevelDict = dict((v, k + 1) for k,v in levelWorkflowDict.iteritems())

    # Filter classifications
    # Specific to Gravity Spy based on study done earlier
    classifications.loc[classifications.links_workflow == 3063, 'links_workflow'] = 2360

    # Make sure classification was done in an actual active workflow
    classifications = classifications.loc[classifications.links_workflow.isin(workflowOrder)]

    # Convert workflow ID into 'Level'
    classifications['Level'] = classifications.links_workflow.apply(levelDict)

    # Make sure choice is a valid index
    classifications = classifications.loc[classifications.annotations_value_choiceINT != -1]

    # Make sure to evaluate only logged in users
    classifications = classifications.loc[classifications.links_user != 0]

    # Ignore NONEOFTHEABOVE classificatios when constructing confusion matrix
    classifications = classifications.loc[classifications.annotations_value_choiceINT != 12]
    # Initialize empty user DB by setting currentworkflow to the maximimum level a user had made a classification in
    # (This logic will capture new users not in the DB loaded above)
    userStatusInit = pd.DataFrame({'userID' : classifications.groupby('links_user').Level.max().index.tolist(), 'workflowInit' : classifications.groupby('links_user').Level.max().tolist()})

    # Retrieve Answers
    answers = getAnswers('1104')
    answersDictRev =  dict(enumerate(sorted(answers[2360].keys())))
    answersDict = dict((str(v),k) for k,v in answersDictRev.iteritems())

    # From ansers Dict determine number of classes
    numClasses = max(answersDict.iteritems(), key=operator.itemgetter(1))[1] + 1

    # merge the golden image DF with th classification (this merge is on links_subject (i.e. the zooID of the image classified)
    image_and_classification = classifications.merge(goldenDF, on=['links_subjects'])

    # This is where the power of pandas comes in...on the fly in very quick order we can fill all users confusion matrices by smartly chosen groupby
    test = image_and_classification.groupby(['links_user','annotations_value_choiceINT','GoldLabel'])
    test = test.count().links_subjects.to_frame().reset_index()

    # Create "Sparse Matrices" and perform a normalization task on them. 
    # Afterwards determine if the users diagonal is above the threshold set above
    confusion_matrices = pd.DataFrame()
    for iUser in test.groupby('links_user'):
	columns = iUser[1].annotations_value_choiceINT        
	rows = iUser[1]['GoldLabel']
	entry = iUser[1]['links_subjects']
	tmp = coo_matrix((entry,(rows,columns)),shape=(numClasses, numClasses))
	conf_divided,a1,a2,a3 = np.linalg.lstsq(np.diagflat(tmp.sum(axis=1)), tmp.todense())
	confusion_matrices = confusion_matrices.append(pd.DataFrame({'userID' : iUser[0], 'conf_matrix' : [conf_divided]}, index=[iUser[0]]))

    return confusion_matrices
