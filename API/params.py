# Classes which contain various methods for priors and weighting
import numpy as np
import pandas as pd
import pdb

class Priors:
    def __init__(self):
        '''
        We can read in the ML training set in __init__
        '''


    def uniform(self, numClasses):
        """
        """
        prior = np.ones((numClasses))/numClasses
        return prior

class Weighting:
    def __init__(self):
        '''
        Read in the classifications table in __init__
        We can sort by classification ID, and see the highest workflow that
        the user has classified in at the time of the classification
        '''


    def default(self, data, user, glitch):
        """
        our default weighting is to give beginner users 0.5 the weight of the machine, apprentice user 1.0, master users 2.0
        """
        # make dict for relating workflow to weighting
        workflow_dict = {'1610': 'B1', '1934': 'B2', '1935': 'B3', '2360': 'A', '2117': 'M'}
        weight_dict = {'B1': 0.5, 'B2': 0.5, 'B3': 0.5, 'A': 1.0, 'M': 2.0}

        # sort classifications by date
        userClassifications = data[data.links_user == user]
        userClassifications = userClassifications.sort_values(by=['metadata_finished_at'])

        # find the ID of the glitch classification in question (should only be 1 classification)
        classificationNum=np.argwhere(userClassifications.links_subjects == glitch).min()
        # find IDs of apprentice and master workflow classificaitons
        apprenticeClass = np.argwhere(userClassifications.links_workflow == 2360)
        if len(apprenticeClass) > 0:
            apprenticeNum = apprenticeClass.min()
        else:
            apprenticeNum = -1
        masterClass = np.argwhere(userClassifications.links_workflow == 2117)
        if len(masterClass) > 0:
            masterNum = masterClass.min()
        else:
            masterNum = -1

        # apply the proper weight
        if (masterNum > 0) & (masterNum < classificationNum):
            weight = 2.0
        elif (apprenticeNum > 0) & (apprenticeNum < classificationNum):
            weight = 1.0
        else:
            weight = 0.5
        return weight

class NOA:
    def __init__(self):
        '''
        we'll probably need to read in the entire classifications table here as well
        '''


    def default(self, links_user):
        """
        """
        prior = np.ones((numClasses))/numClasses
        return prior

