#Script by CIERA Intern Group, 7/19/16

### Import standard modules ###
import numpy as np
import pandas as pd
import json

### Main function to read classification data ###
def read_data(file):

  
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
          x = x[list(x.keys())[0]]['Filename1'].split('_')[1]
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
      
  
  ### Read in csv with custom read for those column in JSON format ###

  # Define location of classification file
  class_file = "gravity-spy-classifications.csv" 

  # Create dataframe from csv
  data1 = pd.read_csv(class_file,converters={'annotations':CustomParser,'subject_data':CustomParser})

  # Change ID to int
  data1['user_id']        = data1['user_id'].apply(convert_to_int)
  # Doing a mild work around for the json format of the annontation column
  data1['annotations']    = data1['annotations'].apply(filter_json)
  # Extract choice and making it a column
  data1['choice']         = data1['annotations'].apply(extract_choice)
  # Extract the task entry and making it a column
  data1['tasks']          = data1['annotations'].apply(extract_tasks)
  # Extract answers and making it a column
  data1['answers']        = data1['annotations'].apply(extract_answers)
  # Extract zooniverse ID it gave this subject and making it a column
  data1['zooID']          = data1['subject_data'].apply(extract_zooID) 
  # Extract uniqueID assigned to the image during image creation and making it a column
  data1['imageID']        = data1['subject_data'].apply(extract_FileName1)
  # Get cumulative count of number of prior classifications by user
  data1['classification_number'] = data1.groupby('user_id').cumcount()
  # Check that the subject_ids for a given classification is 4. If not I uploaded the images wrong for that subject
  data1['goodUpload']     = data1['subject_ids'].apply(check_upload)
  # Check that the number of annotation is of size 1 (i.e. they did not do multiple annotation)
  data1['numAnnotations'] = data1['choice'].apply(check_anno)


  # Dropping annotations,subject_data, and subject_ids
  data1 = data1.drop('annotations',1)
  data1 = data1.drop('subject_data',1)
  data1 = data1.drop('subject_ids',1)
  
  
  ### Check if workflow version is acceptable ###
  versions = [692.102,714.11399999999992] # List of acceptable versions
  data1['goodWorkFlow'] = (data1['workflow_version'].isin(versions)) # Add column of booleans, true means acceptable
  
  
  ### Version specific quality checks ###

  # Data for converting old to new imageIDs
  id_data = pd.read_csv('IDmatchall.txt',delim_whitespace=True,skiprows=1,names=['new_imageID','old_imageID'])

  beta_check = ~data1['workflow_version'].isin([692.102, 714.11399999999992]) # Check if classification from beta 2.0
  id_check = data1['imageID'].isin(id_data['old_imageID']) # Check if imageID has a new ID

  data1['goodID'] = beta_check | id_check # Apply 'bitwise-or' to checks, append to dataframe
  
  
  ### Apply data quality cuts ###
  data1 = data1[data1.goodUpload & data1.numAnnotations & data1.goodWorkFlow & data1.goodID & data1.user_id != 0]

  # Drop unnecessary columns
  data1 = data1.drop('user_ip',1)
  data1 = data1.drop('workflow_name',1)
  data1 = data1.drop('created_at',1)
  data1 = data1.drop('gold_standard',1)
  data1 = data1.drop('expert',1)
  data1 = data1.drop('tasks',1)
  data1 = data1.drop('answers',1)
  data1 = data1.drop('goodUpload',1)
  data1 = data1.drop('numAnnotations',1)
  data1 = data1.drop('goodWorkFlow',1)
  data1 = data1.drop('goodID',1)
  data1 = data1.drop('metadata',1)
  
  
  ### Convert alpha labels to int labels and old to new imageIDs ###

  label_dict = {'45MHZLGHTMDLTN':5,'LGHTMDLTN':5,'50HZ':8,'RCMPRSSR50HZ':8,'BLP':9,'CHRP':2,'XTRMLLD':6,'HLX':14,'KFSH':18,
                'LWFRQNCBRST':1,'LWFRQNCLN':7,'NGLTCH':19,'DNTSGLTCH':19,'NNFTHBV':16,'PRDDVS':11,'60HZPWRLN':10,'60HZPWRMNS':10,
                'PWRLN60HZ':10,'RPTNGBLPS':3,'SCTTRDLGHT':4,'SCRTCH':15,'TMT':12,'VLNHRMNC500HZ':17,'VLNMDHRMNC500HZ':17,
                'HRMNCS':17,'WNDRNGLN':13,'WHSTL':0}

  def choice_replace(x):
      return label_dict[x[0]]

  old_imageID = list(id_data['old_imageID'])
  new_imageID = list(id_data['new_imageID'])
  id_dict = {}

  for a,b in zip(old_imageID,new_imageID):
      id_dict[a] = b

  def imageID_replace(x):
      try:
          x = id_dict[x]
          return x
      except:
          return x
      
  data1['choice']      = data1['choice'].apply(choice_replace)
  data1['imageID']     = data1['imageID'].apply(imageID_replace)
  
  
  ### Pivot dataframe to make index imageID and get choice, user_id, and workflow_version ###

  # Function to aggregate data
  def lister(x):
      return list(x)

  # Use pandas pivot_table, create columns corresponding to image type and true label
  image_values         = ['choice', 'user_id','workflow_version','classification_number','zooID']
  images               = pd.pivot_table(data1,index='imageID',values=image_values,aggfunc=lister)
  images['zooID']      = images['zooID'].apply(np.unique)
  images['type']       = ['T']*len(images)
  images['true_label'] = [-1]*len(images)
  
  
  ### Read in ML_scores ###

  # Remove Hanford and Livingston designations
  def name_clean(x):
      x = x.split('_')[1]
      return x

  ML_scores_L       = pd.read_csv('scores_L.csv')
  ML_scores_H       = pd.read_csv('scores_H.csv')
  ML_scores         = ML_scores_L.append(ML_scores_H)
  ML_scores['Name'] = ML_scores['Name'].apply(name_clean)
  
  
  ### Append ML_posterior matrix ###

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
  
  
  ### Read classification of golden images ###

  goldendata = pd.read_csv('GLabel.csv')

  # Map zooID to true_label
  gold_dict = {}
  for a,b in zip(goldendata['zooID'],goldendata['Classification']):
      gold_dict[int(a)] = int(b)

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
  
  return images