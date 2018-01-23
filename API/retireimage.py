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

# Flat retirement criteria
retired_thres = .9*np.ones(numClasses)

# Flat priors b/c we do not know what category the image is in
priors = np.ones((numClasses))/numClasses

# Load info about classifications and glitches
print '\nreading classifications...'
classifications = pd.read_pickle('pickled_data/weighted_classifications.pkl')
classifications = classifications.loc[~(classifications.annotations_value_choiceINT == -1)]
# NOTE: we remove all classifications that were done on defunct workflows
classifications = classifications.loc[~(classifications.weight == 0.0)]

print 'reading glitches...'
glitches = pd.read_pickle('pickled_data/glitches.pkl')
# filter glitches for only testing images
glitches = glitches.loc[glitches.ImageStatus != 'Training']
glitches['MLScore'] = glitches[classes].max(1)
glitches['MLLabel'] = glitches[classes].idxmax(1)

# Merge DBs
print 'combining data...'
combined_data = classifications.merge(glitches)

#Must start with earlier classifications and work way to new ones
combined_data.drop_duplicates(['links_subjects','links_user'],inplace=True)

# Create imageDB
columnsForImageDB = sorted(workflowDictSubjectSets[2117].keys())
columnsForImageDB.extend(['uniqueID','links_subjects','MLScore','MLLabel','id'])
image_db = combined_data[columnsForImageDB].drop_duplicates(['links_subjects'])
image_db.set_index(['links_subjects'],inplace=True)
image_db['numLabel'] = 0
image_db['retired'] = 0
image_db['numRetire'] = 0
image_db['finalScore'] = 0.0
image_db['finalLabel'] = ''
image_db['cum_weight'] = 0.0

# Load confusion matrices
print 'reading confusion matrices...'

conf_matrices = pd.read_pickle('pickled_data/conf_matrices_chron.pkl')
#conf_matrices_old = pd.read_pickle('pickled_data/confusion_matrices.pkl')
#confusion_matrices = calcConfMatrix.main() 

def get_post_contribution(x):
    # NOTE: the subject link is the variable x
    print x
    # find all classifications for a particular subject
    glitch = classifications[classifications.links_subjects==x]
    # NOTE: for now only take classifications from registered users
    glitch = glitch[glitch.links_user != 0]
    # ensure each classification id has a confusion matrix
    matrices = conf_matrices[conf_matrices.id.isin(glitch.id)]
    glitch = glitch[glitch.id.isin(matrices.id)]
    # sort based on when the classification was made
    glitch = glitch.sort_values('metadata_finished_at')
    # counter to keep track of the weighting normalization, starts at 1.0 for machine
    weight_ctr = 1.0

    # loop through all people that classified until retirement is reached
    for person in glitch.links_user:
        classification = glitch[glitch.links_user == person]
        # if they classified the image multiple times, take the most recent classification
        if len(classification) > 1:
            classification = classification.iloc[-1]
        # save the correct confusion matrix
        matrix = matrices[matrices.id == int(classification.id)].conf_matrix.values[0]
        # for every image they classifiy as a certain type, a users contribution to the posterior for that type is the same for every image. Therefore, it is in our interest to pre-compute all of these values.
        post_contribution = matrix/np.sum(matrix, axis=1)
        # find the row associated with the annotation the user made
        row = int(classification.annotations_value_choiceINT)
        # grab the posterior contribution for that class, weighted by classification weight
        posteriorToAdd = float(classification.weight)*post_contribution[row, :]
        if np.isnan(posteriorToAdd).any(): 
            return
        # keep track of weighting counter for normalization purposes
        weight_ctr += float(classification.weight)
        # for now, only use posteriors for users that have seen and classified a golden image of this particular class
        # update image_db with the posterior contribution
        image_db.loc[x, classes] = image_db.loc[x, classes].add(np.asarray(posteriorToAdd).squeeze())
        # add 1 to numLabels for all images
        image_db.loc[x, 'numLabel'] = image_db.loc[x, 'numLabel'] + 1
        # check if we have more than 1 label for an image and check for retirement
        # Check if posterior is above threshold, add 1 for the ML component
        #posterior = image_db.loc[x][classes].divide(image_db.loc[x]['numLabel'] + 1)
        posterior = image_db.loc[x][classes].divide(weight_ctr)
        if ((posterior > retired_thres).any() and image_db.loc[x, 'numLabel'] > 1):
            # save count that was needed to retire image
            image_db.loc[x, 'numRetire'] = image_db.loc[x, 'numLabel']
            image_db.loc[x, 'finalScore'] = posterior.max()
            image_db.loc[x, 'finalLabel'] = posterior.idxmax()
            image_db.loc[x, 'retired'] = 1
            image_db.loc[x, 'cum_weight'] = weight_ctr
            print '...number of classifications to retire: %i' % image_db.loc[x, 'numRetire']
            return

        # for now, let's assume everything with >20 classifications and no retirement has not retired
        if image_db.loc[x, 'numLabel'] > 20:
            return



print 'determining retired images...'
# sort data based on subjects number
subjects = combined_data.links_subjects.unique()
subjects.sort()
#confusion_matrices.apply(get_post_contribution, axis=1)
for idx, g in enumerate(subjects):
    get_post_contribution(g)
    
retired_db = image_db.loc[image_db.retired == 1]
retired_db.to_pickle('pickled_data/ret_subjects_chron.pkl')
