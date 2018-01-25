from panoptes_client import *

import pandas as pd
import numpy as np
import os, sys
import pdb

from pyomega.API import projectStructure
from pyomega.API import calcConfMatrix

# List of defunct workflows
defunct_workflows = [1479, 3063, 5106]

# Load info about classifications and glitches
print '\nreading classifications...'
classifications = pd.read_pickle('../pickled_data/classifications.pkl')
classifications = classifications.loc[~(classifications.annotations_value_choiceINT == -1)]

# Make an empty DataFrame for the classification weight, starting at beginner's weight
classifications['weight'] = 0.5

# Make a list of all the users
users = classifications.links_user.unique()

# Cycle through users, and append weight for each classification
# NOTE: B1=1610, B2=1934, B3=1935, A=2360, M=2117
for idx, user in enumerate(users):
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

    # see how far into the loop we are
    if idx%100 == 0:
        sys.stderr.write('\r {:04.2f}% complete'.format(100*float(idx)/len(users)))

# weight all defunct classes with a weight of 0.0
classifications.loc[classifications.links_workflow.isin(defunct_workflows), 'weight'] = 0.0

print '\nwriting to pickle...'
classifications.to_pickle('../pickled_data/weighted_classifications.pkl')

