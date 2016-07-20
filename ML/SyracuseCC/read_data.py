#Script by CIERA Intern Group, 7/19/16

#import modules
import numpy as np
import json
import pandas as pd

def read_data():
  
  #read in files
  data = pd.read_csv('gravity-spy-classifications.csv', skiprows=29573, nrows=20008) #reads data after Beta 2.0
  data = np.asarray(data) #convert to numpy array
  
  #create dict for converting old imageIDs to new imageIDs (due to timestamp error)
  tmp_new = [] #empty list of new imageIDs
  tmp_old = [] #empty list of old imageIDs
  no_match = [] #list of imageIDs with no match (timestamp errors)
  id_dict = {} #create empty dict

  id_match = pd.read_csv('IDmatchall.txt') #read in file

  for i in id_match['# New       Old']: #iterate over data in file
      
      if len(i)>10: #to ignore new imageIDs that have no corresponding old imageIDs
          
          i = i.split(' ') #split new and old imageIDs, append to corresponding lists
          tmp_new.append(i[0])
          tmp_old.append(i[1])

  id_match = pd.DataFrame({'new':tmp_new,'old':tmp_old}) #create dataframe of old and new imageIDs

  for a,b in zip(id_match['new'],id_match['old']): #iterate over imageID dataframe
      id_dict[b] = a #map old imageID to new imageID in dict
      
  #create dict for mapping alpha labels to numeric labels
  label_dict = {'45MHZLGHTMDLTN':1,'LGHTMDLTN':1,'50HZ':2,'RCMPRSSR50HZ':2,'BLP':3,'CHRP':4,'XTRMLLD':5,'HLX':6,'KFSH':7,
              'LWFRQNCBRST':8,'LWFRQNCLN':9,'NGLTCH':10,'DNTSGLTCH':10,'NNFTHBV':11,'PRDDVS':12,'60HZPWRLN':13,'60HZPWRMNS':13,
              'PWRLN60HZ':13,'RPTNGBLPS':14,'SCTTRDLGHT':15,'SCRTCH':16,'TMT':17,'VLNHRMNC500HZ':18,'VLNMDHRMNC500HZ':18,
              'HRMNCS':18,'WNDRNGLN':19,'WHSTL':20}
              
  #create dataframe from classification data
  tmp_user= [] #create empty lists
  tmp_user_id = []
  tmp_workflow = []
  tmp_task = []
  tmp_choice = []
  tmp_retired = []
  tmp_unique_id = []
  tmp_zoo_id = []

  for i in range(len(data)): #create list to hold to output information of each classification
      
      output = [] #check that there was only 1 choice made...
      annotations = json.loads(data[i][11])
      idcheck = data[i][2]
      
      if str(annotations).count('choice') == 1 and not np.isnan(idcheck):
          
          user = data[i,1]
          user_id = data[i,2]
          workflow = data[i,5]
          
          #annotations
          task = annotations[0]["task"]
          choice = annotations[0]["value"][0]["choice"]
          
          #subject data
          subject_data = json.loads(data[i][12])
          for key in subject_data:
              zoo_id = key
              retired = subject_data[key]['retired']
              unique_id = subject_data[key]['subject_id']
          
          #append this information into a temporary output file
          tmp_user_id.append(user_id)
          tmp_workflow.append(workflow)
          tmp_task.append(task)
          tmp_choice.append(choice)
          tmp_retired.append(retired)
          tmp_unique_id.append(unique_id)
          tmp_zoo_id.append(zoo_id)
          
  #store classification data in dataframe
  classifications = pd.DataFrame({'imageID':tmp_unique_id,'userID':tmp_user_id,'workflow':tmp_workflow,
                                  'task':tmp_task,'label':tmp_choice,'type':tmp_retired, 'zooID':tmp_zoo_id})
  
  #create list of unique imageIDs that have new IDs and appear in classification data                                
  uniques = set(np.unique(classifications['imageID'])) #create set of unique imageIDs
  keys = set(id_dict.keys()) #create set of new imageIDs from id_dict
  uniques = list(uniques.intersection(keys)) #find intersection of sets, convert to list
  
  #function to create lists of empty lists
  def emptylist(x):
      elist = []
      for i in range(x):
          elist.append([])
      return elist
      
  #read data from GravSpy beta
  pd.options.mode.chained_assignment = None  # default='warn', turns off unnecessary warning about setting values to slice of dataframe

  #create dataframe, length of uniques, without labels or userIDs
  images = pd.DataFrame({'type':['T']*len(uniques),'labels':emptylist(len(uniques)),
                          'userIDs':emptylist(len(uniques)),'ML_posterior':emptylist(len(uniques)),
                          'truelabel':[-1]*len(uniques),'imageID':uniques,'zooID':emptylist(len(uniques))})

  for i in range(len(uniques)): #iterate over unique imageIDs
      
      classifications_idx = np.where((uniques[i] == classifications['imageID']))[0][0]
      
      images['zooID'][i] = int(classifications.loc[[classifications_idx], 'zooID'])
      
      for locations in np.where(uniques[i] == classifications['imageID']): #iterate over arrays of where unique imageID appears
          
          images_idx = np.where(uniques[i] == images['imageID'])[0][0] #find index of line in images where unique imageID appears
          
          for location in locations: #iterate over elements in array of locations in classifications where unique imageID appears
          
              images['labels'][images_idx].append(label_dict[classifications['label'][location]]) #append numeric label
              images['userIDs'][images_idx].append(int(classifications['userID'][location])) #append userID
              
  for imageID in images['imageID']: #iterate over imageIDs

      imageID = id_dict[imageID] #convert old imageIDs to new imageIDs using dict
   
  #read data from golden images
  goldendata = pd.read_csv('GLabel.csv')

  for i in range(len(goldendata)): #iterate over data
      
      try:
          images_idx = np.where(int(goldendata['zooID'][i]) == images['zooID'])[0][0] #find location in images dataframe
          images['truelabel'][images_idx] = int(goldendata['Classification'][i]) #change true label to golden classification
          images['type'][images_idx] = 'G' #change image type to golden
          
      except:
          pass #to catch errors caused by images in goldendata not being in images dataframe
          
  return images