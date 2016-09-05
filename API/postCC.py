import pandas as pd
import numpy as np
from sqlalchemy.engine import create_engine
import ast
import os
import pdb

from matplotlib import use
use('agg')
from matplotlib import (pyplot as plt, cm)

# Load my sql tables of confusion matrices and pp_matrices
SQL_USER = os.environ['SQL_USER']
SQL_PASS = os.environ['SQL_PASS']
engine = create_engine('mysql://{0}:{1}@localhost/GravitySpy'.format(SQL_USER,SQL_PASS))

# Open classification table and extract most recent classificationID
images             = pd.read_sql('SELECT * FROM images_for_pp',engine)
confusion_matrices = pd.read_sql('SELECT * FROM confusion_matrices',engine)

# convert strings to array and reshape to C by C
images.ML_posterior = images.ML_posterior.apply(ast.literal_eval)
images.choice       = images.choice.apply(ast.literal_eval)

C = len(images[images['type']=='T'].ML_posterior[0])

def reshape_array(x):
    try:
        x = np.asarray(x)
        x = x.reshape(C,-1)
    except:
        x = False
    return x

confusion_matrices.conf_matrix = confusion_matrices.conf_matrix.apply(ast.literal_eval)
confusion_matrices.conf_matrix = confusion_matrices.conf_matrix.apply(reshape_array)
images.pp_matrix               = images.pp_matrix.apply(ast.literal_eval)
images.pp_matrix               = images.pp_matrix.apply(reshape_array)

# These choices are strings and we need to change them to integers in order to run the crowd sourcing classifer (CC for short). This is the thing that evaluates users and images.

int_to_label = ["AIRCOMP","BLIP","CHIRP","EXTREMELOUD","HELIX","KFISH","LM","LFB","LFL","NOGTCH","NOA","PRDDVS","PLINE","REBLIPS","STTDLGHT","STCHY","TOMTE","VM","WL","WSTLE"]

B1 = 1610
B2 = 1934
B3 = 1935
B4 = 1936
A  = 2360
M  = 2117
B1_Types = [1,19]
B2_Types = [1,5,12,19]
B3_Types = [1,2,5,12,14,19]
B4_Types = [1,2,5,9,10,12,14,19]

options   = [[]]

def plt_conf_matrices(x):
    reducedConf = x.conf_matrix[~np.all(x.conf_matrix == 0,axis=1)]
    reducedConf = reducedConf[:,~np.all(reducedConf == 0,axis=0)]
    conf_divided,a1,a2,a3 = np.linalg.lstsq(np.diag(np.sum(reducedConf,axis=1)),reducedConf)

    fig, ax = plt.subplots()
    cax = ax.matshow(conf_divided, cmap='viridis_r', vmin=0, vmax=1)
    colorbarticks = np.linspace(0,1,6)
    cbaxes = fig.add_axes([0.9, 0.1, 0.03, 0.8])
    fig.colorbar(cax,ticks=colorbarticks,cax=cbaxes)

    ax.set_xlabel('predicted class')
    ax.set_ylabel('actual class')
    fig.suptitle('confusion matrix: {0}'.format(x.userID))

    confEntries = np.unique(np.where(x.conf_matrix != 0 ))
    ax.set_xticks(range(confEntries.size))
    ax.set_xticklabels(np.array(int_to_label)[confEntries].tolist())
    ax.set_yticks(range(confEntries.size))
    ax.set_yticklabels(np.array(int_to_label)[confEntries].tolist())
    ax.xaxis.set_ticks_position('bottom')

    fig.savefig('confusion/{0}_confusion.png'.format(x.userID))
    plt.close(fig)

def plt_pp_matrix(x):
    fig,ax = plt.subplots()

    cax = ax.matshow(x.pp_matrix, cmap='viridis_r',vmin=0, vmax=1)
    cbaxes = fig.add_axes([0.9, 0.1, 0.03, 0.8])
    colorbarticks = np.linspace(0,1,6)
    fig.colorbar(cax,ticks=colorbarticks,cax=cbaxes)

    fig.suptitle('pp_matrix of {0}'.format(x.imageID))
    ax.set_xlabel('p(i|j)')
    ax.set_ylabel('classes')
    ax.set_yticks(range(len(x.pp_matrix)))
    ax.set_yticklabels(int_to_label)
    fig.savefig('pp_matrices/{0}_pp_matrix.png'.format(x.imageID))
    plt.close(fig)

images[images['type']=='T'][['imageID','pp_matrix']].apply(plt_pp_matrix,axis=1)
confusion_matrices[['userID','conf_matrix']].apply(plt_conf_matrices,axis=1)
