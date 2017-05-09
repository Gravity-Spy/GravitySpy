from panoptes_client import *

import pandas as pd
import ast
import numpy as np
import os, sys
import ast
import pdb
import datetime
import collections
import operator

from sqlalchemy.engine import create_engine

engine = create_engine('postgresql://{0}:{1}@gravityspy.ciera.northwestern.edu:5432/gravityspy'.format(os.environ['QUEST_SQL_USER'],os.environ['QUEST_SQL_PASSWORD']))

pathToFiles = '/home/scoughlin/O2/Test/GravitySpy/API/'

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
worktolevel = {1610:1,1934:2,1935:3,2360:4,2117:5,3063:4}
leveltoworkflow = {1:1610,2:1934,3:1935,4:2360,5:2117}
def workflowtolevel(x):
    return worktolevel[x]

def leveltowork(x):
    return leveltoworkflow[x]


classifications = pd.read_sql('classifications',engine) 
images = pd.read_hdf('{0}/images.h5'.format(pathToFiles))
classifications = classifications.loc[classifications.links_workflow.isin([1610,1934,1935,2360,3063])]

classifications['Level'] = classifications.links_workflow.apply(workflowtolevel)
# Merge classificaitons and images
image_and_classification = classifications.merge(images)
confusion_matrices = pd.read_pickle('{0}/confusion_matrices.pkl'.format(pathToFiles))
print('old user list ' +str(len(confusion_matrices)))
c =  max(label_dict.iteritems(), key=operator.itemgetter(1))[1] + 1

#def make_conf_matrices(x):
#    return np.zeros((c,c))

for x in classifications.loc[~classifications.links_user.isin(confusion_matrices.userID),'links_user'].unique():
    confusion_matrices = confusion_matrices.append(pd.DataFrame({'userID': [x], 'confusion_matrices' : [np.zeros((c,c))],'currentworkflow' : [1],'Level2' : [0],'Level3' : [0],'Level4' : [0],'Level5' : [0]},index = [len(confusion_matrices)]))
#confusion_matrices['confusion_matrices'] = confusion_matrices.userID.apply(make_conf_matrices)
#def determine_min_level(x):
#    return classifications.loc[classifications.links_user == x,'Level'].max()
#confusion_matrices['currentworkflow'] =  confusion_matrices.userID.apply(determine_min_level)
#confusion_matrices['currentworkflow'] = 1
#confusion_matrices['Level2'] = 0
#confusion_matrices['Level3'] = 0
#confusion_matrices['Level4'] = 0
#confusion_matrices['Level5'] = 0

print(len(image_and_classification))
print('updated user list ' +str(len(confusion_matrices)))

alpha = .7*np.ones(c)
Panoptes.connect()
project = Project.find(slug='zooniverse/gravity-spy')

def update_conf_matrix(x):
    # Define what classes belongt o what levels
    promotion_1 = [1,19]
    promotion_2 = [17, 12, 5, 1, 19]
    promotion_3 = [5, 7, 10, 12, 14, 2,  1, 19, 17]
    promotion_4 = [ 13, 16,  3,  7,  6, 15,  4,  1,  5, 18, 11,  8, 14, 12,  0, 10,19, 17,  2]
    # Increase the matrix at [true_label,userlabel]
    confusion_matrices.loc[confusion_matrices.userID == x.links_user,'confusion_matrices'].iloc[0][x['metadata_#Label'],x['annotations_value_choiceINT']] += 1
    for iWorkflow in range(1,5):
        if (confusion_matrices.loc[confusion_matrices.userID == x.links_user,'currentworkflow'].iloc[0] == iWorkflow) and np.diag(confusion_matrices.loc[confusion_matrices.userID == x.links_user,'confusion_matrices'].iloc[0])[locals()['promotion_' + str(iWorkflow)]].all():
            # Take normalized diagonal.
            conf_divided,a1,a2,a3 = np.linalg.lstsq(np.diag(np.sum(confusion_matrices.loc[confusion_matrices.userID == x.links_user,'confusion_matrices'].iloc[0],axis=1)),confusion_matrices.loc[confusion_matrices.userID == x.links_user,'confusion_matrices'].iloc[0])
            alphaTmp = np.diag(conf_divided)
            if (confusion_matrices.loc[confusion_matrices.userID == x.links_user,'currentworkflow'].iloc[0] == iWorkflow) and ((alphaTmp[locals()['promotion_' + str(iWorkflow)]] >= alpha[locals()['promotion_' + str(iWorkflow)]]).all()):
                confusion_matrices.loc[confusion_matrices.userID == x.links_user,'currentworkflow'] = iWorkflow + 1
                confusion_matrices.loc[confusion_matrices.userID == x.links_user,'Level{0}'.format(iWorkflow +1)] = x['id']
                user = User.find(x.links_user)
                new_settings = {"workflow_id": "{0}".format(leveltoworkflow[iWorkflow + 1])}
                print(user)
                print(new_settings)
                ProjectPreferences.save_settings(project=project, user=user, settings=new_settings)

image_and_classification = image_and_classification.sort_values('created_at')
lastID = pd.read_csv('{0}/lastIDPromotion.csv'.format(pathToFiles))['lastID'].iloc[0]
newlastID = image_and_classification['id'].max()
print('lastID of Promotion ' + str(lastID))
image_and_classification = image_and_classification.loc[image_and_classification['id']>lastID]
image_and_classification.loc[image_and_classification.metadata_Type == 2,['links_user','metadata_#Label','annotations_value_choiceINT','id']].apply(update_conf_matrix,axis=1)
for iWorkflow in range(1,6):
    print('Level {0}: {1}'.format(iWorkflow,len(confusion_matrices.loc[confusion_matrices.currentworkflow == iWorkflow])))

confusion_matrices.to_pickle('{0}/confusion_matrices.pkl'.format(pathToFiles))
pd.DataFrame({'lastID':newlastID},index=[0]).to_csv(open('{0}/lastIDPromotion.csv'.format(pathToFiles),'w'),index=False)
print('New lastID of Promotion ' + str(newlastID))
