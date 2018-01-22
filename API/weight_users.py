from panoptes_client import *

import pandas as pd
import numpy as np
import os, sys
import pdb

from pyomega.API import projectStructure
from pyomega.API import calcConfMatrix

# Obtain number of classes from API
workflowDictSubjectSets = projectStructure.main('1104','O2')
classes = sorted(workflowDictSubjectSets[2117].keys())

# From ansers Dict determine number of classes
numClasses = len(classes)

# Load info about classifications and glitches
print '\nreading classifications...'
classifications = pd.read_pickle('pickled_data/classifications.pkl')
classifications = classifications.loc[~(classifications.annotations_value_choiceINT == -1)]

# Make an empty DataFrame for the classification weight, starting at beginner's weight
classifications['weight'] = 0.5

# Make a list of all the users
users = classifications.links_user.unique()

# Cycle through users, and append weight for each classification
# NOTE: B1=1610, B2=1934, B3=1935, A=2360, M=2117
for user in users:
    # sort classifications by date
    user_classifications = classifications[classifications.links_user == user]
    user_classifications = user_classifications.sort_values(by=['metadata_finished_at'])

    # find the index at which they became master, or set to nan
    master_index = user_classifications[user_classifications.links_workflow == 2117].index
    if master_index.size != 0:
        master_index = master_index[0]
        master_index = user_classifications[user_classifications.metadata_finished_at >= user_classifications.loc[master_index, 'metadata_finished_at']].index
    # find the indices at which they are an apprentice, or set to -1
    apprentice_index = user_classifications[user_classifications.links_workflow == 2360].index
    if apprentice_index.size != 0:
        apprentice_index = apprentice_index[0]
        apprentice_index = user_classifications[user_classifications.metadata_finished_at >= user_classifications.loc[apprentice_index, 'metadata_finished_at']].index

    # add weights to the user classifications
    if apprentice_index.size != 0:
        classifications.loc[apprentice_index, 'weight'] = 1.0
    if master_index.size != 0:
        classifications.loc[master_index, 'weight'] = 2.0
    
    print user

classifications.to_pickle('weighted_classifications.pkl')

