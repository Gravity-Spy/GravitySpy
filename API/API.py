from panoptes_client import *

import pandas as pd
import ast
import numpy as np
import os, sys
import ast
import pdb
import datetime
import collections

########################
####### Functions ######
#########################

pathToFiles = '/home/scoughlin/O2/Development/API/'

def extract_choiceINT(x):
    try:
        return label_dict[x]
    except:
        return -1

label_dict = {
'50HZ':0,'RCMPRSSR50HZ':0,'AIRCOMPRESSOR50HZ':0,
'BLP':1,'BLIP':1,
'CHRP':2,'CHIRP':2,
'XTRMLLD':3,'EXTREMELYLOUD':3,
'HLX':4,'HELIX':4,
'KFSH':5,'KOIFISH':5,
'45MHZLGHTMDLTN':6,'LGHTMDLTN':6,'LIGHTMODULATION':6,
'LWFRQNCBRST':7,'LOWFREQUENCYBURST':7,
'LWFRQNCLN':8,'LOWFREQUENCYLINE':8,
'NNFTHBV':9,'NONEOFTHEABOVE':9,
'NGLTCH':10,'DNTSGLTCH':10,'NOGLITCH':10,
'PRDDVS':11,'PAIREDDOVES':11,
'60HZPWRLN':12,'60HZPWRMNS':12,'PWRLN60HZ':12,'POWERLINE60HZ':12,
'RPTNGBLPS':13,'REPEATINGBLIPS':13,
'SCTTRDLGHT':14,'SCATTEREDLIGHT':14,
'SCRTCH':15,'SCRATCHY':15,
'TMT':16,'TOMTE':16,
'VLNHRMNC500HZ':17,'VLNMDHRMNC500HZ':17, 'HRMNCS':17,'VIOLINMODEHARMONIC500HZ':17,
'WNDRNGLN':18,'WANDERINGLINE':18,
'WHSTL':19,'WHISTLE':19
}

#This function generically flatten a dict
def flatten(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        if v and (type(v) is list):
            v = v[0]
        new_key = parent_key + sep + k if parent_key else k
        try:
            items.extend(flatten(v, new_key, sep=sep).items())
        except:
            items.append((new_key, v))
    return dict(items)

# Load lastID that was parsed
lastID = pd.read_hdf('{0}/GravitySpy.h5'.format(pathToFiles),columns=['id']).max().iloc[0]


# Connect to panoptes and query all classifications done on project 1104 (i.e. GravitySpy)
Panoptes.connect()

# Created empty list to store the previous classifications
classificationsList = []

# Query the last 100 classificaitons made (this is the max allowable)
allClassifications = Classification.where(scope='project',project_id='1104',last_id='{0}'.format(lastID),page_size='100')

# Loop until no more classifications
for iN in range(0,100):
    try:
        classification = allClassifications.next()
    except:
        break

    # Generically with no hard coding we want to parse all aspects of the
    # classification metadata. This will ease any changes on the API side and
    # any changes to the metadata on our side.

    try:
        classificationsList.append(flatten(classification.raw))
    except:
        continue

if not classificationsList:
    raise ValueError('No New Classifications')

# Now we want to make a panda data structure with all this information
classifications = pd.DataFrame(classificationsList)
classifications = classifications.convert_objects(convert_numeric=True) 
classifications.created_at = pd.to_datetime(classifications.created_at,infer_datetime_format=True)
classifications.metadata_started_at = pd.to_datetime(classifications.metadata_started_at,infer_datetime_format=True)
classifications.metadata_finished_at = pd.to_datetime(classifications.metadata_finished_at,infer_datetime_format=True)

# At this point we have generically parsed the classification of the user. The label given by the parser is a string and for the purposes of our exercise converting these strings to ints is useful. After we will append the classifications to the old classifications and save. Then we tackle the information about the image that was classified. 

classifications['annotations_value_choiceINT'] = classifications['annotations_value_choice'].apply(extract_choiceINT)
classifications = classifications.select_dtypes(exclude=['object'])
classifications = classifications[['created_at','id','links_project','links_subjects','links_user','links_workflow','metadata_finished_at','metadata_started_at','metadata_workflow_version','annotations_value_choiceINT']]
classifications.to_hdf('GravitySpy.h5','classifications',mode='a',format='table',append=True,data_columns=['id','links_subjects','links_user','links_workflow','annotations_value_choiceINT'])

# Load subjects info in case this is an image that has been previously labeled and getting the metadata information is superfluous

# Load images
newSubjects = []
for iSubject in classifications['links_subjects'].unique():
    tmpSubject = Subject.find(iSubject)
    newSubjects.append(flatten(tmpSubject.raw))

images = pd.DataFrame(newSubjects)

# Check if gold (In an effort to improve table storage. These are now integers.
#     Testing (only ML Label) = 1
#     Gold    = 2
#     Retired = 3

def determine_type(x):
    if x == 'Gold':
        return 2
    else:
        return 1

def determine_label(x):
    return label_dict[str(x).upper().translate(None,'AEIOUY ()')]

try:
    images['metadata_Type'] = images['metadata_#Type'].apply(determine_type)
except:
    images['metadata_Type'] = 1

try:
    images.loc[images['metadata_#Label'].isnull(),'metadata_#Label'] = -1 
except:
    images['metadata_#Label'] = -1

images.loc[images['metadata_#Label'] != -1,'metadata_#Label'] = images.loc[images['metadata_#Label'] != -1,'metadata_#Label'].apply(determine_label)

images.loc[images['metadata_#Label'] == -1,'metadata_#ML_Posterior'] = images.loc[images['metadata_#Label'] == -1,'metadata_#ML_Posterior'].apply(str)
images.loc[images['metadata_#Label'] == -1,'metadata_#ML_Posterior'] = images.loc[images['metadata_#Label'] == -1,'metadata_#ML_Posterior'].apply(ast.literal_eval)
images.loc[images['metadata_#Label'] == -1,'metadata_#ML_Posterior'] = images.loc[images['metadata_#Label'] == -1,'metadata_#ML_Posterior'].apply(np.argmax)

images.loc[images['metadata_#Label'] != -1,'metadata_#ML_Posterior'] = -1

images = images.convert_objects(convert_numeric=True)
images = images.select_dtypes(exclude=['object'])
images.rename(columns={'id':'links_subjects'},inplace=True)

#Out[117]: 
#Index([        u'links_subjects',      u'links_collections',
#                u'links_project',     u'links_subject_sets',
#              u'metadata_#Label', u'metadata_#ML_Posterior',
#                u'metadata_date',          u'metadata_Type'],
#      dtype='object')


# Merge classificaitons and images
image_and_classification = classifications.merge(images)

c =  max(label_dict.iteritems(), key=operator.itemgetter(1))[1] + 1

def make_conf_matrices(x):
    return np.zeros((c,c))

confusion_matrices = pd.DataFrame(classifications.links_user.unique(),columns=['userID'])
confusion_matrices['confusion_matrices'] = confusion_matrices.userID.apply(make_conf_matrices)
confusion_matrices['currentworkflow'] = 1610
confusion_matrices['Level2'] = 0
confusion_matrices['Level3'] = 0
confusion_matrices['Level4'] = 0
confusion_matrices['Level5'] = 0

alpha = .7*np.ones(c)

promotion_1610 = [1,19]
promotion_1934 = [17, 12, 5, 1, 19]
promotion_1935 = [5, 7, 10, 12, 14, 2,  1, 19, 17]
promotion_2360 = [ 13, 16,  3,  7,  6, 15,  4,  1,  5, 18, 11,  8, 14, 12,  0, 10,19, 17,  2]


def update_conf_matrix(x):
    pdb.set_trace()
    # Increase the matrix at [true_label,userlabel]
    confusion_matrices.loc[confusion_matrices.userID == x.links_user,'confusion_matrices'].iloc[0][x['metadata_#Label'],x['annotations_value_choiceINT']] += 1
    # Take normalized diagonal.
    conf_divided,a1,a2,a3 = np.linalg.lstsq(np.diag(np.sum(confusion_matrices.loc[confusion_matrices.userID == x.links_user,'confusion_matrices'].iloc[0],axis=1)),confusion_matrices.loc[confusion_matrices.userID == x.links_user,'confusion_matrices'].iloc[0])
    alphaTmp = np.diag(conf_divided)
    # Check is promotion is necessary and note it.
    if (confusion_matrices.loc[confusion_matrices.userID == x.links_user,'currentworkflow'] == 1610) and (alphaTmp[promotion_1610] >= alpha[promotion_1610]):
        confusion_matrices.loc[confusion_matrices.userID == x.links_user,'currentworkflow'] = 1934
        confusion_matrices.loc[confusion_matrices.userID == x.links_user,'Level2'] = x['id']

    if (confusion_matrices.loc[confusion_matrices.userID == x.links_user,'currentworkflow'] == 1934) and (alphaTmp[promotion_1934] >= alpha[promotion_1934]):
        confusion_matrices.loc[confusion_matrices.userID == x.links_user,'currentworkflow'] = 1935
        confusion_matrices.loc[confusion_matrices.userID == x.links_user,'Level3'] = x['id']

    if (confusion_matrices.loc[confusion_matrices.userID == x.links_user,'currentworkflow'] == 1935) and (alphaTmp[promotion_1935] >= alpha[promotion_1935]):
        confusion_matrices.loc[confusion_matrices.userID == x.links_user,'currentworkflow'] = 2360
        confusion_matrices.loc[confusion_matrices.userID == x.links_user,'Level4'] = x['id']

    if (confusion_matrices.loc[confusion_matrices.userID == x.links_user,'currentworkflow'] == 2360) and (alphaTmp[promotion_2360] >= alpha[promotion_2360]):
        confusion_matrices.loc[confusion_matrices.userID == x.links_user,'currentworkflow'] = 2117
        confusion_matrices.loc[confusion_matrices.userID == x.links_user,'Level5'] = x['id']

    elif confusion_matrices.loc[confusion_matrices.userID == x.links_user,'currentworkflow'] == 2117:
    else:
        ValueError("User has workflow assigned that is not known")

image_and_classification.loc[image_and_classification.metadata_Type == 2,['links_user','metadata_#Label','annotations_value_choiceINT','id']].apply(update_conf_matrix,axis=1)

"""
############
###########
# At this point we have a nice classifications based data frame. Now we want to turn this into a subject based data frame. This is done below.

def decider(x,imageID):

    v = np.sum(x[0], axis=1)/np.sum(np.sum(x[0])) # Create vector of normalized sums of pp_matrix2
    maximum = np.amax(v) # Currentialize maximum, max value of v
    maxIdx = np.argmax(v) # Currentialize maxIdx, index of max value of v
    true_label = maxIdx # true_label is index of maximum value
    images.set_value(imageID, 'true_label', true_label)
    numLabels = np.unique(np.nonzero(x[0])[1]).size - 1

    if (maximum >= t[maxIdx]) and (numLabels >=2): # If maximum is above threshold for given class, retire image

        images.set_value(imageID, 'type', 'R') # Change type of image
        images.set_value(imageID, 'numUntilRetired', numLabels)

        #print('Image is retired to class', true_label)

    elif (numLabels >= r_lim) or (images.loc[imageID,'NOACount'] >=3): # Pass to a higher workflow if more than r_lim annotators and no decision reached

        #print('Image is given to a higher workflow')
        images.set_value(imageID, 'type', 'M')

def det_promoted(x,userID,classNum):
    if (x[np.where(x !=0)] > alpha[np.where(x !=0)]).all():
        if (len(x[np.where(x !=0)]) == 2) and (unique_users.loc[unique_users.userID == userID,'promoted'].iloc[0] != 'B2'):
            unique_users.loc[unique_users.userID == userID,'promoted'] = 'B2'
            unique_users.loc[unique_users.userID == userID].classification.iloc[0].append(len(classifications.loc[(classifications.userID == userID) & (classifications.workflow == 1610) & (classifications.classificationID< classNum)]))
        elif (len(x[np.where(x !=0)]) == 5) and (unique_users.loc[unique_users.userID == userID,'promoted'].iloc[0] != 'B3'):
            unique_users.loc[unique_users.userID == userID,'promoted'] = 'B3'
            unique_users.loc[unique_users.userID == userID].classification.iloc[0].append(len(classifications.loc[(classifications.userID == userID) & (classifications.workflow == 1934) & (classifications.classificationID< classNum)]))
        elif (len(x[np.where(x !=0)]) == 9) and (unique_users.loc[unique_users.userID == userID,'promoted'].iloc[0] != 'A'):
            unique_users.loc[unique_users.userID == userID,'promoted'] = 'A'
            unique_users.loc[unique_users.userID == userID].classification.iloc[0].append(len(classifications.loc[(classifications.userID == userID) & (classifications.workflow == 1935) & (classifications.classificationID< classNum)]))
        elif (len(x[np.where(x !=0)]) == 20) and (unique_users.loc[unique_users.userID == userID,'promoted'].iloc[0] != 'M'):
            print(userID)
            unique_users.loc[unique_users.userID == userID,'promoted'] = 'M'
            unique_users.loc[unique_users.userID == userID].classification.iloc[0].append(len(classifications.loc[(classifications.userID == userID) & (classifications.workflow == 2360) & (classifications.classificationID< classNum)]))

##############################
# Start of the CC classifier #
##############################
# initialize variables

# Max citizens who can look at image before it is given to upper class if threshold not reached
r_lim = 10 # Make 23

# Number of classes
c = len(images[images['type']=='T'].ML_posterior.iloc[0])

# Flat priors b/c we do not know what category the image is in
priors = np.ones((1,c))/c

 # Threshold vector for user promotion
alpha = .7*np.ones(c)

# Threshold vector for what minimum ML confidence we are willing to update the confusion matrix of a user who labels that image.
#g_c = [0.725,0.999,0.5,0.99999,0,0.99,0.995,0.99999999,.95,.95,.95,.95,.95,.95,.95,.95,.95,.95,.95,.95]
g_c_l = [0.75,0.9998,0.5,0.96,1,0.991,0.995,0.99999999,0.99995,1,0.9993,1,0.9994,0.9958,1,0.99999995,0.7,1,1,0.998]
g_c_h = [0.9058,0.99996,1,0.9995,1,0.9972,0.999983,0.99999952,0.99875,1,1,1,0.9998,1,0.999979,1,0.977,1,1,1]
g_c   = [0.9058,0.99996,1,0.9995,1,0.9972,0.999983,0.99999952,0.99995,1,1,1,0.9998,1,1,1,0.977,1,1,1]
#g_c = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

# Threshold vector for image retirement
t = [0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.999,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9]

# Workflow numbers

B1 = 1610
B2 = 1934
B3 = 1935
A  = 2360
A1 = 3063
M  = 2117

images_with_pp     = pd.read_csv('/home/gravityspyweb/API/ProjectSoftware/API/images_for_pp_{0}_{1}.csv'.format(timestamp.year,'11'))
confusion_matrices = pd.read_csv('/home/gravityspyweb/API/ProjectSoftware/API/unique_users_{0}_{1}.csv'.format(timestamp.year,'11'))

def reshape_array(x):
    try:
        x = np.asarray(x)
        x = x.reshape(c,-1)
    except:
        x = False
    return x

confusion_matrices.conf_matrix = confusion_matrices.conf_matrix.apply(ast.literal_eval)
confusion_matrices.conf_matrix = confusion_matrices.conf_matrix.apply(reshape_array)
confusion_matrices.classification = confusion_matrices.classification.apply(ast.literal_eval)
def stringx(x):
    try:
        return ast.literal_eval(x)
    except:
        return False

images_with_pp.pp_matrix       = images_with_pp.pp_matrix.apply(stringx)
images_with_pp.pp_matrix       = images_with_pp.pp_matrix.apply(reshape_array)


# Identify retired or moved images
imageStatus = images_with_pp[['NOACount','numUntilRetired','type','zooID']]
images = images.merge(imageStatus,on='zooID',how='outer')
images.loc[images.type_y.isnull(),'type_y'] = images.loc[images.type_y.isnull(),'type_x']
images['type'] = images['type_y']
images.loc[images.type_x == 'G','type'] = 'G'
images.drop(['type_y','type_x'],axis=1,inplace=True)

def make_pp_matrices(x):
    if x.zooID in images_with_pp.zooID.tolist():
        tmp = np.array(images_with_pp[images_with_pp.zooID == x.zooID].pp_matrix)[0]
        if tmp.shape[1] != len(x.userID)+1:
            hold = np.zeros((c,len(x.userID)+1))
            hold[:,:(tmp.shape[1]-(len(x.userID)+1))] = tmp
            tmp = hold
    else:
        tmp = np.zeros((c,len(x.userID)+1))
        tmp[:,0] = x.ML_posterior

    return [tmp]

# Currentialize PP_Matrix for the images that are in the testing set
#(no need for golden images whose label is already known)
images['pp_matrix'] = images[images['type']=='T'][['userID','zooID','ML_posterior']].apply(make_pp_matrices, axis = 1)
del images_with_pp

# Create list of unique userIDs from the classifications variable which contains all the classifcations done for the project
unique_users_tmp = pd.DataFrame({'userID' : classifications.userID.unique().tolist()})

# Currentialize some confusion matrices for all the users
unique_users = confusion_matrices[['userID','conf_matrix','classification','promoted']]

for iNN in unique_users_tmp.userID:
    if iNN not in unique_users.userID.tolist():
        tmpframe = pd.DataFrame({'userID': [iNN], 'conf_matrix' : [np.zeros((c,c))],'classification' : [[]],'promoted' : ['S']},index = [len(unique_users)])
        unique_users = unique_users.append(tmpframe)

# We must update the confusion matrix of a user in order and then create pp_matrix for the image with the confusion matrix of the user at the time they labelled the image.
classifications = classifications[classifications.zooID.isin(images.zooID)]
classifications = classifications.drop_duplicates('classificationID')
classifications['classification_number'] = classifications.sort_values('classificationID').groupby('userID').cumcount()
classifications = classifications.sort_values(['userID','classification_number'])
classifications = classifications.sort_values(['classificationID'])

# set index to zooID
images.set_index('zooID',inplace=True)

for imageID,userID,user_label in zip(classifications.zooID,classifications.userID,classifications.choiceINT):


    if (images.loc[imageID,'type'] == 'G'):#or (images.loc[imageID,'type'] == 'R'): # If golden image or retired

        true_label = images.loc[imageID,'true_label']
        unique_users.conf_matrix[unique_users.userID==userID].iloc[0][true_label,user_label] += 1

    elif (images.loc[imageID,'type'] == 'T') and (images.loc[imageID,'ML_confidence']>g_c[images.loc[imageID,'ML_label']]) and (not ((images.loc[imageID,'workflow'][images.loc[imageID,'userID'].index(userID)] in [B1,B2,B3]) and (user_label == 9))):

        true_label = images.loc[imageID,'ML_label']
        unique_users.conf_matrix[unique_users.userID==userID].iloc[0][true_label,user_label] += 1

    elif (images.loc[imageID,'type'] == 'M') and (images.loc[imageID,'ML_confidence']>g_c[images.loc[imageID,'ML_label']]) and (not ((images.loc[imageID,'workflow'][images.loc[imageID,'userID'].index(userID)] in [B1,B2,B3]) and (user_label == 9))):

        true_label = images.loc[imageID,'ML_label']
        unique_users.conf_matrix[unique_users.userID==userID].iloc[0][true_label,user_label] += 1

    elif (images.loc[imageID,'type'] == 'R') and (images.loc[imageID,'ML_confidence']>g_c[images.loc[imageID,'ML_label']]) and (not ((images.loc[imageID,'workflow'][images.loc[imageID,'userID'].index(userID)] in [B1,B2,B3]) and (user_label == 9))):

        true_label = images.loc[imageID,'ML_label']
        unique_users.conf_matrix[unique_users.userID==userID].iloc[0][true_label,user_label] += 1

    elif (images.loc[imageID,'type'] == 'RZ') and (images.loc[imageID,'ML_confidence']>g_c[images.loc[imageID,'ML_label']]) and (not ((images.loc[imageID,'workflow'][images.loc[imageID,'userID'].index(userID)] in [B1,B2,B3]) and (user_label == 9))):

        true_label = images.loc[imageID,'ML_label']
        unique_users.conf_matrix[unique_users.userID==userID].iloc[0][true_label,user_label] += 1

        #print('Confusion matrix updated')        
    conf_divided,a1,a2,a3 = np.linalg.lstsq(np.diag(np.sum(unique_users.conf_matrix[unique_users.userID==userID].iloc[0],axis=1)),unique_users.conf_matrix[unique_users.userID==userID].iloc[0])
    alphaTmp = np.diag(conf_divided)

    det_promoted(alphaTmp,userID,images.loc[imageID,'classificationID'][images.loc[imageID,'userID'].index(userID)])

    if (images.loc[imageID,'type'] == 'T') and (images.loc[imageID,'workflow'][images.loc[imageID,'userID'].index(userID)] in [B1,B2,B3]) and (user_label == 9):
        images.loc[imageID,'NOACount'] = images.loc[imageID,'NOACount'] + 1

    if (images.loc[imageID,'type'] == 'T') and (images.loc[imageID,'workflow'][images.loc[imageID,'userID'].index(userID)] in [A,A1]) and (user_label == 9):
        images.loc[imageID,'NOACount'] = images.loc[imageID,'NOACount'] + 1

    if images.loc[imageID,'type'] == 'T': # If training image

        temp_matrix = priors

        if sum(conf_divided[:,user_label]) != 0: # If column of conf_divided corresponding to user label is not blank

            temp_matrix = (conf_divided[:,user_label]*priors[0][user_label])/sum(conf_divided[:,user_label]*priors[0])

        pp_matrix = images.loc[imageID,'pp_matrix'][0]
        pp_index  = images.loc[imageID,'userID'].index(userID) + 1
        pp_matrix[:,pp_index] = temp_matrix
        images.set_value(imageID,'pp_matrix',[pp_matrix])

        decider(images.loc[imageID,'pp_matrix'],imageID)

def makeM(x):
    if len(x.choiceINT) >10:
        images.loc[images.imageID == x.imageID,'type'] = 'M'

images[images['type']!='G'][['choiceINT','imageID']].apply(makeM,axis=1)

images.loc[images['type']=='R','pp_matrix'] = float('nan')
images.loc[images['type']=='RZ','pp_matrix'] = float('nan')
images.loc[images['type']=='M','pp_matrix'] = float('nan')
images.sort_values('type',ascending=False,inplace=True)

# All of the corwd sourcing work has finished and the rest of the code simply manipulates the data to make it easy to save into a mySQL database that will be loaded later in the post processing stages of the CC algorithm.

def prep_for_sql(x):
    try:
        x = str(x.flatten().tolist())
    except:
        x = False
    return x

def prep_for_sql2(x):
    try:
        x = str(np.round(x[0],5).flatten().tolist())
    except:
        x = False
    return x


unique_users.conf_matrix     = unique_users.conf_matrix.apply(prep_for_sql)
unique_users.classification  = unique_users.classification.apply(str)
unique_users.to_csv(open('/home/gravityspyweb/API/ProjectSoftware/API/unique_users_{0}_{1}.csv'.format(timestamp.year,'11'),'w'),index=False)

images.choice                = images.choice.apply(str)
images.choiceINT             = images.choiceINT.apply(str)
images.classificationID      = images.classificationID.apply(str)
images.userID                = images.userID.apply(str)
images.classification_number = images.classification_number.apply(str)
images.workflow              = images.workflow.apply(str)
images.ML_posterior          = images.ML_posterior.apply(str)
images.pp_matrix             = images.pp_matrix.apply(prep_for_sql2)
images = images.reset_index()

images.to_csv(open('/home/gravityspyweb/API/ProjectSoftware/API/images_for_pp_{0}_{1}.csv'.format(timestamp.year,'11'),'w'),index=False)

print(classifications.classificationID.max())
print(len(unique_users.loc[unique_users.promoted == 'S']))
print(len(unique_users.loc[unique_users.promoted == 'B2']))
print(len(unique_users.loc[unique_users.promoted == 'B3']))
print(len(unique_users.loc[unique_users.promoted == 'A']))
print(len(unique_users.loc[unique_users.promoted == 'M']))
"""
