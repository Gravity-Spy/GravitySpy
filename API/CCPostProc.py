import pandas as pd
import numpy as np
from sqlalchemy.engine import create_engine
import ast
import os
import pdb
from panoptes_client import *

# Load my sql tables of confusion matrices and pp_matrices
SQL_USER = os.environ['SQL_USER']
SQL_PASS = os.environ['SQL_PASS']
engine = create_engine('mysql://{0}:{1}@localhost/GravitySpy'.format(SQL_USER,SQL_PASS))

# Open classification table and extract most recent classificationID
images             = pd.read_sql('SELECT * FROM images_for_pp',engine)
confusion_matrices = pd.read_sql('SELECT * FROM confusion_matrices',engine)

# Connect to panoptes and query all classifications done on project 1104 (i.e. GravitySpy)
Panoptes.connect()
project = Project.find(slug='zooniverse/gravity-spy')

workflow_dict = {"B1":1610, "B2":1934, "B3":1935,"A":2360,"M":2117}

# Determine who is promoted and change their workflow preference
def promote_users(x):
    user = User.find("{0}".format(x.userID))
    new_settings = {"workflow_id": "{0}".format(workflow_dict['B2'])}#.format(workflow_dict[x.promoted])}
    print(user)
    print(new_settings)
    #ProjectPreferences.save_settings(project=project, user=user, settings=new_settings)

confusion_matrices[confusion_matrices.userID == 386563][['userID','promoted']].apply(promote_users,axis =1)

def move_image(x):
    # Find subject set it should be moved to which for the beginner workflows is the NOA beginner set and for the other workflows apprentice NOA.
    tmp = SubjectSet.find(5675)
    subject = Subject.find(x.zooID)
    # Add to new subject set
    tmp.add(subject)
    # Remove it form subject set it is currently in.
    tmp = SubjectSet.find(x.subject_set)
    tmp.remove(subject)
    return

images[images.decision == 2][['subject_set','zooID']].apply(move_image,axis=1)
