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
        we read in the classificaitons in the main function already, so these will just be arguments to ther pertinenet weighting schemes
        '''
        # make dict for relating workflow to weighting
        workflow_dict = {'1610': 'B1', '1934': 'B2', '1935': 'B3', '7765': 'B4', '7766': 'A', '7767': 'M'}
        b05_a1_m2_dict = {'B1': 0.5, 'B2': 0.5, 'B3': 0.5, 'B4' : 0.5, 'A': 1.0, 'M': 2.0}

    def uniform(self, data, user, glitch):
        """
        default weighting scheme where all users are created equal
        """
        weight = 1.0
        return weight

    def ranked(self, data, user, glitch):
        """
        our default weighting is to give beginner users 0.5 the weight of the machine, apprentice user 1.0, master users 2.0
        """


        # select classifications for user, ordered by id
        userClassifications = data[data.links_user == user]

        # find the ID of the glitch classification in question (should only be 1 classification)
        classificationNum = userClassifications[userClassifications.links_subjects == glitch].id.values[0]

        # find IDs of apprentice and master workflow classificaitons
        apprenticeClass = userClassifications[userClassifications.links_workflow.isin([2360,7766])].id.values
        if len(apprenticeClass) > 0:
            apprenticeNum = apprenticeClass.min()
        else:
            apprenticeNum = -1
        masterClass = userClassifications[userClassifications.links_workflow.isin([2117,7767])].id.values
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

        if (apprenticeNum > masterNum) & (masterNum > 0):
            raise ValueError

        return weight

    def machine_dominated(self, data, user, glitch):
        """
        machine_dominated weighted humans relative to each other by the default method, but makes the total weight from the human equal to the weight of the machine
        """
        # sort classifications by date
        userClassifications = data[data.links_user == user]
        userClassifications = userClassifications.sort_values(by=['metadata_finished_at'])

        # find the ID of the glitch classification in question (should only be 1 classification)
        classificationNum=np.argwhere(userClassifications.links_subjects == glitch).min()
        # find IDs of apprentice and master workflow classificaitons
        apprenticeClass = np.argwhere(userClassifications.links_workflow == 7766)
        if len(apprenticeClass) > 0:
            apprenticeNum = apprenticeClass.min()
        else:
            apprenticeNum = -1
        masterClass = np.argwhere(userClassifications.links_workflow == 7767)
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

