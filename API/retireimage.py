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
classifications = pd.read_pickle('pickled_data/classifications.pkl')
classifications = classifications.loc[~(classifications.annotations_value_choiceINT == -1)]

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

# Load confusion matrices
print 'reading confusion matrices...'

confusion_matrices = pd.read_pickle('pickled_data/conf_matrices_chron.pkl')
#conf_matrices_old = pd.read_pickle('pickled_data/confusion_matrices.pkl')
#confusion_matrices = calcConfMatrix.main() 

def get_post_contribution(x):
    # find all classifications for a particular subject
    glitch = classifications[classifications.links_subjects==x]
    # FIXME: for now only take classifications from registered users
    glitch = glitch[glitch.links_user != 0]
    # ensure each classification id has a confusion matrix
    matrices = confusion_matrices[confusion_matrices.classificationid.isin(glitch.id)]
    glitch = glitch[glitch.id.isin(matrices.classificationid)]
    # sort based on when the classification was made
    glitch = glitch.sort_values('metadata_finished_at')
    # NOTE: the subject link is the variable x

    # loop through all people that classified until retirement is reached
    for person in glitch.links_user:
        classification = glitch[glitch.links_user == person]
        matrix = matrices[matrices.classificationid == classification.id.values[0]].conf_matrix.values[0]
        if not matrix.any():
            return
        # for every image they classifiy as a certain type, a users contribution to the posterior for that type is the same for every image. Therefore, it is in our interest to pre-compute all of these values.
        post_contribution = matrix/np.sum(matrix, axis=1)
        # find the row associated with the annotation the user made
        row = classification.annotations_value_choiceINT.as_matrix()
        # grab the posterior contribution for that class
        posteriorToAdd = post_contribution[row, :]
        # for now, only use posteriors for users that have seen and classified a golden image of this particular class
        if np.isnan(posteriorToAdd.any()): 
            continue
        # update image_db with the posterior contribution
        image_db.loc[x, classes] = image_db.loc[x, classes].add(np.asarray(posteriorToAdd).squeeze())
        # add 1 to numLabels for all images
        image_db.loc[x, 'numLabel'] = image_db.loc[x, 'numLabel'] + 1
        # check if we have more than 1 label for an image and check for retirement
        # Check if posterior is above threshold, add 1 for the ML component
        posterior = image_db.loc[x][classes].divide(image_db.loc[x]['numLabel'] + 1)
        if ((posterior > retired_thres).any() and image_db.loc[x, 'numLabel'] > 1):
            # save count that was needed to retire image
            image_db.loc[x, 'numRetire'] = image_db.loc[x, 'numLabel']
            image_db.loc[x, 'finalScore'] = posterior.max()
            image_db.loc[x, 'finalLabel'] = posterior.idxmax()
            image_db.loc[x, 'retired'] = 1
            return



print 'determining retired images...'
# sort data based on subjects number
subjects = combined_data.links_subjects.unique()
subjects.sort()
#confusion_matrices.apply(get_post_contribution, axis=1)
for g in subjects:
    get_post_contribution(g)
pdb.set_trace()

retired_db = image_db.loc[image_db.retired == 1]
retired_db.to_pickle('pickled_data/ret_subjects.pkl')
