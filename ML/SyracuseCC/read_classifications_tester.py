### Script for testing the NUCC ###

t_values = []
ML_values = []
citizen_values = []
retirement_values = []
cost_function = []
true_percentages = []
ML_true_percentages = []
citizen_true_percentages = []

### Import standard modules ###

import numpy as np
import pandas as pd
import pickle as pk
import json

### Define set of parsing functions ###

def CustomParser(data):
    j1 = json.loads(data)
    return j1

def filter_json(x):
    x=x[0]
    try:
        x['value']=x['value'][0]
    except:
        x['value'] = {u'answers': {}, u'choice': {}, u'filters': {}}
    return x

def extract_choice(x):
    y = []
    y.append((str(x['value']['choice'])))
    return y

def extract_tasks(x):
    x=x['task']
    return x

def extract_answers(x):
    x=x['value']['answers']
    return x

def extract_filters(x):
    x=['value']['filters']
    return x

def extract_zooID(x):
    x=int(list(x.keys())[0])
    return x

def extract_FileName1(x):
    try:
        x = str(x[list(x.keys())[0]]['Filename1'].split('_')[1])
    except:
        x = ''
    return x

def check_upload(x):
    if len(x.split(';')) == 4:
        x = True
    else:
        x = False
    return x

def check_anno(x):
    if len(x) == 1:
        x = True
    else:
        x = False
    return x

def convert_to_int(x):
    try:
        x=int(x)
    except:
        x=0
    return x

### TESTING CELL ###

class_file  = "gravity-spy-classifications.csv" 
class_data  = pd.read_csv(class_file,converters={'annotations':CustomParser,'subject_data':CustomParser})
class_data  = class_data.sort_values('classification_id')
class_data1 = class_data[:40000]
pointer     = len(class_data1)
class_data2 = class_data[40000:]
class_data  = class_data1
# In[71]:

### Read in csv and parse columns in JSON format ###

# Define location of classification file
class_file = "gravity-spy-classifications.csv" 

# Create dataframe from csv
class_data  = pd.read_csv(class_file,converters={'annotations':CustomParser,'subject_data':CustomParser})

# Change ID to int
class_data['userID']                   = class_data['user_id'].apply(convert_to_int)
# Doing a mild work around for the json format of the annotation column
class_data['annotations']              = class_data['annotations'].apply(filter_json)
# Extract choice and making it a column
class_data['choice']                   = class_data['annotations'].apply(extract_choice)
# Extract the task entry and making it a column
class_data['tasks']                    = class_data['annotations'].apply(extract_tasks)
# Extract answers and making it a column
class_data['answers']                  = class_data['annotations'].apply(extract_answers)
# Extract zooniverse ID it gave this subject and making it a column
class_data['zooID']                    = class_data['subject_data'].apply(extract_zooID) 
# Extract uniqueID assigned to the image during image creation and making it a column
class_data['imageID']                  = class_data['subject_data'].apply(extract_FileName1)
# Get cumulative count of number of prior classifications by user
class_data['classification_number']    = class_data.groupby('user_id').cumcount()
# Check that the subject_ids for a given classification is 4. If not I uploaded the images wrong for that subject
class_data['goodUpload']               = class_data['subject_ids'].apply(check_upload)
# Check that the number of annotation is of size 1 (i.e. they did not do multiple annotation)
class_data['numAnnotations']           = class_data['choice'].apply(check_anno)


# Dropping annotations, subject_data, and subject_ids
class_data = class_data.drop('annotations',1)
class_data = class_data.drop('user_id',1)
class_data = class_data.drop('subject_data',1)
class_data = class_data.drop('subject_ids',1)


# In[72]:

### Check if workflow version is acceptable ###
all_versions = list(np.unique(class_data['workflow_version']))
good_versions = [36.7,692.102,714.11399999999992] # List of acceptable versions #692.102 is beta with 18 columns
class_data['goodWorkFlow'] = (class_data['workflow_version'].isin(good_versions)) # Add column of booleans, true means acceptable


# In[73]:

### Version specific quality checks ###

# Data for converting old to new imageIDs
id_data = pd.read_csv('IDmatchall.txt',delim_whitespace=True,skiprows=1,names=['new_imageID','old_imageID'])

# Data for bad golden images
bad_data = pd.read_csv('bad_golden_images.csv',header=None)

# Remove Hanford and Livingston designations
def name_clean(x):
    x = x.split('_')[1]
    return x

# List of bad golden images
bad_images = list(bad_data[0].apply(name_clean))
bad_images.append('ulfd56vzbx')
bad_images.append('uV9zDEjP2N')

not_beta_check = ~class_data['workflow_version'].isin([692.102, 714.11399999999992]) # Check if classification from beta 2.0
new_id_check = class_data['imageID'].isin(id_data['old_imageID']) # Check if imageID has a new ID
not_bad_id = ~class_data['imageID'].isin(bad_images)

class_data['goodID'] = (not_beta_check | new_id_check) & not_bad_id # Apply bitwise boolean operators,, append to dataframe


# In[74]:

### Apply data quality cuts ###
final_check = class_data.goodUpload & class_data.numAnnotations & class_data.goodWorkFlow & class_data.goodID & class_data.userID != 0
class_data  = class_data[final_check]

# Drop unnecessary columns
class_data = class_data.drop('user_ip',1)
class_data = class_data.drop('workflow_name',1)
class_data = class_data.drop('created_at',1)
class_data = class_data.drop('gold_standard',1)
class_data = class_data.drop('expert',1)
class_data = class_data.drop('tasks',1)
class_data = class_data.drop('answers',1)
class_data = class_data.drop('goodUpload',1)
class_data = class_data.drop('numAnnotations',1)
class_data = class_data.drop('goodWorkFlow',1)
class_data = class_data.drop('goodID',1)
class_data = class_data.drop('metadata',1)


# In[75]:

### Convert alpha labels to int labels and old to new imageIDs ###

label_dict = {'45MHZLGHTMDLTN':5,'LGHTMDLTN':5,'50HZ':8,'RCMPRSSR50HZ':8,'BLP':9,'CHRP':2,'XTRMLLD':6,'HLX':14,'KFSH':18,
              'LWFRQNCBRST':1,'LWFRQNCLN':7,'NGLTCH':19,'DNTSGLTCH':19,'NNFTHBV':16,'PRDDVS':11,'60HZPWRLN':10,'60HZPWRMNS':10,
              'PWRLN60HZ':10,'RPTNGBLPS':3,'SCTTRDLGHT':4,'SCRTCH':15,'TMT':12,'VLNHRMNC500HZ':17,'VLNMDHRMNC500HZ':17,
              'HRMNCS':17,'WNDRNGLN':13,'WHSTL':0}

# Convert alpha labels to int labels
def choice_replace(x):
    return label_dict[x[0]]

old_imageID = list(id_data['old_imageID'])
new_imageID = list(id_data['new_imageID'])
id_dict = {}

for a,b in zip(old_imageID,new_imageID):
    id_dict[a] = b

# Convert old to new imageIDs
def imageID_replace(x):
    try:
        x = id_dict[x]
        return x
    except:
        return x
    
class_data['choice']      = class_data['choice'].apply(choice_replace)
class_data['imageID']     = class_data['imageID'].apply(imageID_replace)


# In[76]:

### Sort class_data by classification number ###

class_data = class_data.sort_values('classification_id')

retired_dict = pk.load(open("retired_dict.p", "rb"))

def classify(parameter):

  ### Pivot dataframe to make index imageID and get choice, user_id, and workflow_version ###

  # Function to aggregate data
  def lister(x):
      return list(x)

  # Use pandas pivot_table, create columns corresponding to image type and true label
  image_values         = ['choice', 'userID','workflow_version','classification_number','zooID']
  images               = pd.pivot_table(class_data,index='imageID',values=image_values,aggfunc=lister)
  images['zooID']      = images['zooID'].apply(np.unique)
  images['type']       = ['T']*len(images)
  images['true_label'] = [-1]*len(images)
  images['pp_matrix']  = [0]*len(images)
  images['pp_matrix']  = images['pp_matrix'].astype(object)


  # In[78]:

  ### Append ML_posterior matrix ###

  ML_scores_L       = pd.read_csv('scores_L.csv')
  ML_scores_H       = pd.read_csv('scores_H.csv')
  ML_scores         = ML_scores_L.append(ML_scores_H)
  ML_scores['Name'] = ML_scores['Name'].apply(name_clean)

  # Get number of classes
  classes = len(ML_scores.columns[2:])

  # Create posterior matrix from dataframe columns
  ML_posterior = ML_scores['confidence of class 0']

  # Iterate over columns of dataframe
  for i in range(1,classes): 
      ML_posterior = np.vstack((ML_posterior,ML_scores['confidence of class %s' % str(i)]))

  ML_posterior = ML_posterior.T
  ML_posterior = list(ML_posterior)
  imageIDs = list(ML_scores['Name'])

  # Map imageID to ML_posterior
  ML_dict = {}
  for a,b in zip(imageIDs,ML_posterior):
      ML_dict[a] = b
      
  def ML_append(x):
      try:
          return ML_dict[x]
      except:
          return []

  images_index = pd.Series(images.index)
  ML_posterior = images_index.apply(ML_append)

  # Append ML_posterior matrix to corresponding imageID
  images['ML_posterior'] = list(ML_posterior)


  # In[79]:

  ### Get ML_label and ML_confidence ###

  # Function to get index of max value in ML_posterior
  def max_index(x):
      x = np.array(x)
      try:
          return np.argmax(x)
      except:
          return -1

  # Function to get max confidence value in ML_posterior    
  def get_max(x):
      x = np.array(x)
      try:
          return max(x)
      except:
          return -1
      
  images['ML_label']          = images['ML_posterior'].apply(max_index)
  images['ML_confidence']     = images['ML_posterior'].apply(get_max)


  # In[80]:

  ### Read classification of golden images ###

  goldendata = pd.read_csv('GLabel.csv')

  # Map zooID to true_label
  gold_dict = {}
  beginner_gold = pk.load(open('beginner_data.p','rb'))

  for a,b in zip(goldendata['ZooID'],goldendata['Classification']):
      gold_dict[int(a)] = int(b)

  gold_dict.update(beginner_gold)  
      
  # Change type of golden images 
  def type_map(x):
      x = int(x)
      if x in list(gold_dict.keys()):
          return 'G'
      else:
          return 'T'

  # Change true_label of golden images  
  def label_map(x):
      x = int(x)
      try:
          return gold_dict[x]
      except:
          return -1

  images['type']       = images['zooID'].apply(type_map)
  images['true_label'] = images['zooID'].apply(label_map)

  ### Initialize constants ###

  r_lim = 4                        # Max citizens who can look at image before it is given to upper class if threshold not reached
  c = 20                           # Classes
  priors = (np.ones((1,c))/c)[0]   # Flat priors b/c we do not know what category the image is in
  alpha = .4*np.ones((c,1))        # Threshold vector for user promotion
  g_c = .5*np.ones((c,1))          # Threshold vector for updating confusion matrix
  t = parameter*np.ones((c,1))            # Threshold vector for image retirement


  # In[18]:

  ### Function to create blank pp_matrices ###

  pp_count = {} # Dict mapping imageID to counter of which column of pp_matrix should be updated
  conf_matrices = {} # Dict mapping userID to confusion matrix

  def make_pp_matrices(x): # Create pp_matrices for training images with correct size
      
      if x['type'] == 'T': # If training image
          
          pp_count[x.name] = 0 # pp_count set to zero
          
          return [np.zeros((c,len(x['userID'])+1))] # Create matrix with c rows, # of users +1 columns (+1 is for the ML posterior)
      
      else:
          
          pass

  def make_conf_matrices(x): # Create confusion matrices for new users
      
      for userID in x:
          
          if userID not in list(conf_matrices.keys()): # If user does not have a confusion matrix
              
              conf_matrices[userID] = np.zeros((c,c)) # Create a blank cXc confusion matrix

  images['userID'].apply(make_conf_matrices)
  images['pp_matrix'] = images[['userID','type']].apply(make_pp_matrices, axis = 1)


  # In[19]:

  ### Main loop to update confusion and posterior matrices ###

  for imageID,userID,user_label in zip(class_data['imageID'],class_data['userID'],class_data['choice']): # Iterate over class data
      
                  
      if images.loc[imageID,'type'] == 'G': # If golden image
          
          true_label = images.loc[imageID,'true_label']
                  
          conf_matrices[userID][true_label,user_label] += 1 # Update confusion matrix
          
          #print('Confusion matrix updated')        
      
      
      if images.loc[imageID,'type'] == 'T': # If training image
                  
          conf_divided,a1,a2,a3 = np.linalg.lstsq(np.diag(np.sum(conf_matrices[userID],axis=1)),conf_matrices[userID])
          
          pp_column = priors # If column of conf_divided is all zeroes, pp_column is equal to priors 
          
          if sum(conf_divided[:,user_label]) != 0: # If not all zeroes, calculate pp_column from conf_divided
          
              pp_column = (conf_divided[:,user_label]*priors[user_label])/sum(conf_divided[:,user_label]*priors)
          
          pp_matrix = images.loc[imageID,'pp_matrix'][0] # Get pp_matrix of image
          pp_matrix[:,pp_count[imageID]] = pp_column # Update pp_matrix with pp_column
          images.set_value(imageID,'pp_matrix',[pp_matrix]) # Put pp_matrix in images
          pp_count[imageID] += 1 # Increment count of which column in pp_matrix should be updated
          
          #print('Posterior matrix updated')


  # In[20]:

  ### Function to apply decisions to images ###
  
  #import pdb
  
  true_confidences = []
  
  def decider(x):
      
      x['pp_matrix'][0][:,-1] = np.array(x['ML_posterior'])
      x['pp_matrix'][0].T
      v = np.sum(x['pp_matrix'][0], axis=1)/np.sum(np.sum(x['pp_matrix'][0])) # Create vector of normalized sums of pp_matrix2
      maximum = float(np.amax(v)) # Initialize maximum, max value of v
      maxIdx = np.argmax(v) # Initialize maxIdx, index of max value of v

      if maximum >= t[maxIdx]: # If maximum is above threshold for given class, retire image
              
          true_label = maxIdx # true_label is index of maximum value
          images.set_value(x.name, 'true_label', true_label) # Change true_label of image
          images.set_value(x.name, 'type', 'R') # Change type of image
          true_confidences.append(maximum)
          #print('Image is retired to class', true_label)
          return 1

      elif len(x['choice']) >= r_lim: # Pass to a higher workflow if more than r_lim annotators and no decision reached
              
          #print('Image is given to a higher workflow')
          return 2
              

      else: # If fewer than r_lim annotators have looked at image, keep image
              
          #print('More labels are needed for the image')
          return 3
          
      
  images['decision'] = images[images['type']=='T'][['pp_matrix','ML_posterior','choice']].apply(decider,axis=1)

  ### Function to get ML and citizen accuracy ###

  retired = images[images['type']=='R']
  ML_correct = images[(images['type']=='R') & (images['true_label']==images['ML_label'])]
  nested_citizen_labels = list(retired['choice'])
  citizen_correct = 0

  for labels,true_label in zip(nested_citizen_labels,list(retired['true_label'])):
      for label in labels:
          if label == true_label:
              citizen_correct += 1

  citizen_labels = [val for sublist in nested_citizen_labels for val in sublist]
  ML_percentage = len(ML_correct)/len(retired)
  citizen_percentage = citizen_correct/len(citizen_labels)
  retirement_percentage = len(retired)/len(images)
  true_confidence_avg = np.average(true_confidences)
  
  true_correct = 0
  true_ML = 0
  true_citizen = 0
  citizen_total = 0
  total = 0
  
  for image,label,ML_label,citizen_labels in zip(retired.index.values,retired['true_label'],retired['ML_label'],retired['choice']):
    if image in retired_dict:
      total += 1
      if label in retired_dict[image]:
        true_correct += 1
      if ML_label in retired_dict[image]:
        true_ML += 1
      for citizen_label in citizen_labels:
        citizen_total += 1
        if citizen_label in retired_dict[image]:
          true_citizen += 1
  
  true_percentage = true_correct/total
  ML_true_percentage = true_ML/total
  citizen_true_percentage = true_citizen/citizen_total
  
  return(ML_percentage,citizen_percentage,retirement_percentage,true_confidence_avg,true_percentage,ML_true_percentage,citizen_true_percentage)

parameter = [.7,.73,.76,.79,.82,.85,.88,.91,.94,.95,.995,.9995]
#parameter = [.7,.73]

for item in parameter:
  t_values.append(item)
  ML_percentage,citizen_percentage,retirement_percentage,true_confidence_avg,true_percentage,ML_true_percentage,citizen_true_percentage = classify(item)
  ML_values.append(ML_percentage)
  citizen_values.append(citizen_percentage)
  retirement_values.append(retirement_percentage)
  cost_function.append((true_confidence_avg**2)*retirement_percentage)
  true_percentages.append(true_percentage)
  ML_true_percentages.append(ML_true_percentage)
  citizen_true_percentages.append(citizen_true_percentage)
  
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

f, axarr = plt.subplots(3, sharex=True)
green_patch = mpatches.Patch(color='green', label='ML agreement')
blue_patch = mpatches.Patch(color='blue', label='Citizen agreement')
magenta_patch = mpatches.Patch(color='magenta', label='Cost function')
red_patch = mpatches.Patch(color='red', label='Retirement percentage')
axarr[0].plot(t_values, ML_values, 'g^', t_values, citizen_values, 'bs', t_values, retirement_values, 'ro')
axarr[0].legend(handles=[green_patch,blue_patch,red_patch],bbox_to_anchor=(1.005, 1), loc=2, borderaxespad=0.)
axarr[0].yaxis.set_ticks(np.arange(0,1,.05))
axarr[1].plot(t_values, cost_function, 'mp')
axarr[1].legend(handles=[magenta_patch],bbox_to_anchor=(1.005, 1), loc=2, borderaxespad=0.)
axarr[2].plot(t_values, ML_true_percentages, 'g^', t_values, citizen_true_percentages, 'bs', t_values, true_percentages, 'ro')
green_patch = mpatches.Patch(color='green', label='ML accuracy')
blue_patch = mpatches.Patch(color='blue', label='Citizen accuracy')
red_patch = mpatches.Patch(color='red', label='Combined accuracy')
axarr[2].legend(handles=[green_patch,blue_patch,red_patch],bbox_to_anchor=(1.005, 1), loc=2, borderaxespad=0.)

plt.show()