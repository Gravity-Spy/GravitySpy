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

def main(userID = None, returnMatrixPerImage = False):

    engine = create_engine('postgresql://{0}:{1}@gravityspy.ciera.northwestern.edu:5432/gravityspy'.format(os.environ['QUEST_SQL_USER'],os.environ['QUEST_SQL_PASSWORD']))
    goldenDF = pd.read_sql('goldenimages', engine)
    #goldenDF = pd.read_pickle('../data/goldenDF.pkl')

    if userID:
        # Only retrieve classication by that particular user. 
        classifications = pd.read_sql('SELECT links_user, links_subjects, links_workflow, "annotations_value_choiceINT", id FROM classificationsdev WHERE \"links_user\" = {0}'.format(userID), engine)
        # classifications = 
        if returnMatrixPerImage:
            testing_images_labeled_by_user = classifications.sort_values('id').loc[~classifications.links_subjects.isin(goldenDF.links_subjects)].id.unique() 
    else:
        # Load classifications, current user DB status and golden images from DB
        classifications = pd.read_sql('SELECT links_user, links_subjects, links_workflow, "annotations_value_choiceINT" FROM classificationsdev', engine)

    # order classifications
    classifications = classifications.sort_values('id')
    # Make sure choice is a valid index
    classifications = classifications.loc[classifications.annotations_value_choiceINT != -1]

    # Make sure to evaluate only logged in users
    classifications = classifications.loc[classifications.links_user != 0]

    # Ignore NONEOFTHEABOVE classificatios when constructing confusion matrix
    classifications = classifications.loc[classifications.annotations_value_choiceINT != 12]

    # Retrieve Answers
    answers = getAnswers('1104')
    answersDictRev =  dict(enumerate(sorted(answers[2360].keys())))
    answersDict = dict((str(v),k) for k,v in answersDictRev.items())

    # From ansers Dict determine number of classes
    numClasses = max(answersDict.items(), key=operator.itemgetter(1))[1] + 1

    # merge the golden image DF with the classification (this merge is on links_subject (i.e. the zooID of the image classified)
    image_and_classification = classifications.merge(goldenDF, on=['links_subjects'])

    # This is where the power of pandas comes in...on the fly in very quick order we can fill all users confusion matrices by smartly chosen groupby
    if returnMatrixPerImage:

        confusion_matrices = pd.DataFrame()
        for iclassificationid in testing_images_labeled_by_user:
            test = image_and_classification.loc[image_and_classification.id < iclassificationid].groupby(['links_user','annotations_value_choiceINT','GoldLabel'])
            test = test.count().links_subjects.to_frame().reset_index()

            # Create "Sparse Matrices" and perform a normalization task on them.
            # Afterwards determine if the users diagonal is above the threshold set above
            for iUser in test.groupby('links_user'):
                columns = iUser[1].annotations_value_choiceINT
                rows = iUser[1]['GoldLabel']
                entry = iUser[1]['links_subjects']
                tmp = coo_matrix((entry,(rows,columns)), shape=(numClasses, numClasses))
                conf_divided,a1,a2,a3 = np.linalg.lstsq(np.diagflat(tmp.sum(axis=1)), tmp.todense())
                confusion_matrices = confusion_matrices.append(pd.DataFrame({'links_user' : iUser[0], 'conf_matrix' : [conf_divided], 'id' : [iclassificationid]}, index=[iUser[0]]))

    else:

        test = image_and_classification.groupby(['links_user','annotations_value_choiceINT','GoldLabel'])
        test = test.count().links_subjects.to_frame().reset_index()

        # Create "Sparse Matrices" and perform a normalization task on them.
        # Afterwards determine if the users diagonal is above the threshold set above
        confusion_matrices = pd.DataFrame()
        for iUser in test.groupby('links_user'):
            columns = iUser[1].annotations_value_choiceINT
            rows = iUser[1]['GoldLabel']
            entry = iUser[1]['links_subjects']
            tmp = coo_matrix((entry,(rows,columns)), shape=(numClasses, numClasses))
            conf_divided,a1,a2,a3 = np.linalg.lstsq(np.diagflat(tmp.sum(axis=1)), tmp.todense())
            confusion_matrices = confusion_matrices.append(pd.DataFrame({'userID' : iUser[0], 'conf_matrix' : [conf_divided]}, index=[iUser[0]]))

    return confusion_matrices
