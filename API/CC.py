import pandas as pd
import numpy as np
from sqlalchemy.engine import create_engine
import ast
import os
import pdb
import ast

from matplotlib import use
use('agg')
from matplotlib import (pyplot as plt, cm)

# Get images from MySQL

# Open connection to mysql table
SQL_USER = os.environ['SQL_USER']
SQL_PASS = os.environ['SQL_PASS']
engine = create_engine('mysql://{0}:{1}@localhost/GravitySpy'.format(SQL_USER,SQL_PASS))

images           = pd.read_sql('SELECT * FROM images',engine)
classifications  = pd.read_sql('SELECT * FROM classifications',engine)
classifications  = classifications[['choiceINT','choice', 'userID','workflow','classificationID','zooID','classification_number']]

# Convert from string to list
images.choice                = images.choice.apply(ast.literal_eval)
images.choiceINT             = images.choiceINT.apply(ast.literal_eval)
images.classificationID      = images.classificationID.apply(ast.literal_eval)
images.userID                = images.userID.apply(ast.literal_eval)
images.workflow              = images.workflow.apply(ast.literal_eval)
images.ML_posterior          = images.ML_posterior.apply(ast.literal_eval)
images.classification_number = images.classification_number.apply(ast.literal_eval)

##############################
# Start of the CC classifier #
##############################
# initialize variables
r_lim = 4 # Make 23         # Max citizens who can look at image before it is given to upper class if threshold not reached
c = len(images.ML_posterior[0])                      # Classes
priors = np.ones((1,c))/c   # Flat priors b/c we do not know what category the image is in
alpha = .9*np.ones((c,1))   # Threshold vector for user promotion
g_c = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]     # Threshold vector for updating confusion matrix
t   = [0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9]       # Threshold vector for image retirement

# initilize empty pp_matrices pandas. Only for plotting do not need to save
# pp_matrices    = pd.DataFrame({ 'imageID' : images.imageID,'pp_matrix' : np.empty((len(images.imageID), 0)).tolist()})

def make_pp_matrices(x):
    
    if x['type'] == 'T':
        tmp = np.zeros((c,len(x.userID)+1))
        tmp[:,0] = x.ML_posterior
        return [tmp]
    else:
        pass

images['pp_matrix'] = images[['userID','type','ML_posterior']].apply(make_pp_matrices, axis = 1)

# This function either receives a pre existing confusion matrix or creates a new one.
def get_conf_matrices(x):
    tmp = np.zeros((c,c))
    return tmp

# Creat list of unique userIDs from the images variable
unique_users = pd.DataFrame({'userID' : classifications.userID.unique().tolist()})

# See if we already have a confusion matrix or not for these users. If not create one
unique_users['conf_matrix'] = unique_users.userID.apply(get_conf_matrices)

# We must update the confusion matrix of a user in order and then create pp_matrix for the image with the confusion matrix of the user at the time they labelled the image.
classifications = classifications[classifications.zooID.isin(images.zooID)]
classifications = classifications.sort_values(['userID','classification_number'])

# set index to zooID
images.set_index('zooID',inplace=True)

# Loop over class data
for imageID,userID,user_label in zip(classifications.zooID,classifications.userID,classifications.choiceINT):
    
                
    if (images.loc[imageID,'type'] == 'G') or (images.loc[imageID,'type'] == 'R'): # If golden image
        
        true_label = images.loc[imageID,'true_label']
        unique_users.conf_matrix[unique_users.userID==userID].iloc[0][true_label,user_label] += 1

    elif (images.loc[imageID,'type'] == 'T') and (images.loc[imageID,'ML_confidence']>g_c[images.loc[imageID,'ML_label']]):

        true_label = images.loc[imageID,'ML_label']
        unique_users.conf_matrix[unique_users.userID==userID].iloc[0][true_label,user_label] += 1
        
        #print('Confusion matrix updated')        
    
    
    if images.loc[imageID,'type'] == 'T': # If training image
                
        conf_divided,a1,a2,a3 = np.linalg.lstsq(np.diag(np.sum(unique_users.conf_matrix[unique_users.userID==userID].iloc[0],axis=1)),unique_users.conf_matrix[unique_users.userID==userID].iloc[0])
        
        temp_matrix = priors
        
        if sum(conf_divided[:,user_label]) != 0: # If column of conf_divided corresponding to user label is not blank
        
            temp_matrix = (conf_divided[:,user_label]*priors[0][user_label])/sum(conf_divided[:,user_label]*priors[0])
        
        pp_matrix = images.loc[imageID,'pp_matrix'][0]
        pp_index  = images.loc[imageID,'userID'].index(userID) + 1
        pp_matrix[:,pp_index] = temp_matrix
        images.set_value(imageID,'pp_matrix',[pp_matrix])

### Function to apply decisions to images ###

true_confidences = []
true_labels = []

def decider(x):
    
    v = np.sum(x['pp_matrix'][0], axis=1)/np.sum(np.sum(x['pp_matrix'][0])) # Create vector of normalized sums of pp_matrix2
    maximum = np.amax(v) # Initialize maximum, max value of v
    maxIdx = np.argmax(v) # Initialize maxIdx, index of max value of v
    true_confidences.append(maximum)
    true_labels.append(maxIdx)

    if maximum >= t[maxIdx]: # If maximum is above threshold for given class, retire image
        
        true_label = maxIdx # true_label is index of maximum value
        #images.set_value(x.name, 'true_label', true_label) # Change true_label of image
        #images.set_value(x.name, 'type', 'R') # Change type of image
            
        print('Image is retired to class', true_label)
        return 1

    elif len(x['choice']) >= r_lim: # Pass to a higher workflow if more than r_lim annotators and no decision reached
            
        print('Image is given to a higher workflow')
        return 2
            

    else: # If fewer than r_lim annotators have looked at image, keep image
            
        print('More labels are needed for the image')
        return 3
    
images['decision'] = images[images['type']=='T'][['pp_matrix','choice']].apply(decider,axis=1)

#images['decision'] = images[['imageID','zooID','userID','true_label','choiceINT','type','ML_posterior','ML_label','ML_confidence']].apply(cc_classifier, axis = 1)
def prep_for_sql(x):
    try:
        x = str(x.flatten().tolist())
    except:
        x = False
    return x

def prep_for_sql2(x):
    try:
        x = str(x[0].flatten().tolist())
    except:
        x = False
    return x


unique_users.conf_matrix     = unique_users.conf_matrix.apply(prep_for_sql)
unique_users.to_sql(con=engine, name='confusion_matrices', if_exists='replace', flavor='mysql')
images.choice                = images.choice.apply(str)
images.choiceINT             = images.choiceINT.apply(str)
images.classificationID      = images.classificationID.apply(str)
images.userID                = images.userID.apply(str)
images.classification_number = images.classification_number.apply(str)
images.workflow              = images.workflow.apply(str)
images.ML_posterior          = images.ML_posterior.apply(str)
images.pp_matrix             = images.pp_matrix.apply(prep_for_sql2)
images.to_sql(con=engine, name='images_for_pp', if_exists='replace', flavor='mysql',index=False)
