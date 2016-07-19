#Script by CIERA Intern Group, 7/8/16
#Updated by Scott Coughlin July 12, 2016

#import modules
import numpy as np
import pandas as pd
import random
from pdb import set_trace

#function for creating random permutations
def random_permutation(iterable, r=None):
    "Random selection from itertools.permutations(iterable, r)"
    pool = tuple(iterable)
    r = len(pool) if r is None else r
    return tuple(random.sample(pool, r))

#function to generate one batch of test data for use with gravspy_main2.py
def gen_data():
    images = pd.DataFrame(np.zeros((120,6)), columns=['type','labels','userIDs','ML_posterior','truelabel','imageID']) #images is final dataframe
    images['truelabel'] = images['truelabel'].astype('int')
    images['imageID'] = images['imageID'].astype('int')

    #initialization
    N = 100 #100 images
    R = 30 #30 citizens
    C = 15 #15 classes

    #simulate training labels and true labels for test data
    true_labels = (np.random.randint(0,high=C,size=(1,N)))[0] #generate 1xN array of numbers 1 to C, corresponding to true labels of images
    citizen_training_labels = (np.random.randint(0,high=C,size=(1,int(N/5))))[0] #generate 1x(N/5) array of numbers 1 to C, corresponding to citizen labels of images

    #simulate ML decisions
    ML_dec = np.zeros((C,N)) #create empty matrix for machine decisions

    for i in range(N): #iterate over images

        big_prob = .7 + .2*np.random.rand() #simulate machine deciding on one class
        rest_prob = 1 - big_prob #calculate decisions for other classes

        for j in range(C): #iterate over classes

            if j == true_labels[i]: #if class is true label of image

                ML_dec[j,i] = big_prob #assign big_prob to corresponding element

            else: #if class is not true label of image

                ML_dec[j,i] = rest_prob/(C-1) #assign rest_prob/(C-1) to corresponding element

    ML_dec = np.transpose(ML_dec) #transpose matrix, result is NxC

    #simulate confusion matrices
    conf_matrices = [] #create Empty Array
    conf_userIDs  = []

    for k in range(R): #iterate over citizens

        conf_matrix = np.zeros((C,C)) #create empty CxC matrix

        for ii in range(C): #iterate over rows

            for jj in range(C): #iterate over columns

                if ii == jj: #if diagonal of matrix

                    conf_matrix[ii,jj] = 180 + np.random.randint(0,high=41) #assign value 180 to 220

                else:

                    conf_matrix[ii,jj] = np.random.randint(1,high=6) #assign value 1 to 5

        conf_matrices.append(conf_matrix) #map userID to corresponding conf_matrix
        conf_userIDs.append(k)

    #simulate citizen labels and associated userIDs
    all_labels = [] #create empty list of citizen labels
    all_userIDs = [] #create empty list of userIDs

    for i in range(N): #iterate over images

        labels = [] #create empty list
        total = int(20+10*np.random.rand()) #total amount of labels applied to image, 20 to 30
        correct = int(6+14*np.random.rand()) #amount of correct labels, 6 to 20
        rest = (np.random.randint(0,high=C,size=(1,total-correct)))[0] #generate 1x(total-correct) array of numbers 1 to C, corresponding to random citizen labels

        for j in range(correct): #iterate over range of correct

            labels.append(true_labels[i]) #append correct labels to list of labels

        labels.extend(list(rest)) #append random labels

        all_labels.append(np.array(labels)) #append array of labels to main list of labels
        userIDs = np.array(random_permutation(range(30),total)) #sample (total) random non-repeating IDs
        all_userIDs.append(userIDs) #append array of IDs to main list of IDs

    #simulate citizen training labels and associated userIDs
    all_training_labels = [] #create empty list of citizen training labels
    all_training_userIDs = [] #create empty list of training userIDs

    for i in range(int(N/5)): #iterate over training images

        training_labels = [] #create empty list
        training_correct = int(6+14*np.random.rand()) #amount of correct labels, 6 to 20
        training_rest = (np.random.randint(0,high=C,size=(1,R-training_correct)))[0] #generate 1x(R-correct) array of numbers 1 to C, corresponding to random citizen labels

        for j in range(training_correct): #iterate over range of training_correct

            training_labels.append(citizen_training_labels[i]) #append training label to list of citizen training labels

        training_labels.extend(list(training_rest)) #append random labels

        all_training_labels.append(np.array(training_labels)) #append array of training labels to main list of training labels
        training_userIDs = np.array(random_permutation(range(30),30)) #sample 30 random non-repeating training IDs
        all_training_userIDs.append(training_userIDs) #append array of training IDs to main list of training IDs

#put data in image dataframe
    for i in range(N): #for citizen labels

        images.loc[[i],'type']         = 'T'
        images.loc[[i],'labels']       = pd.Series([all_labels[i]],index=[i])
        images.loc[[i],'userIDs']      = pd.Series([all_userIDs[i]],index=[i])
        images.loc[[i],'ML_posterior'] = pd.Series([ML_dec[i,:]],index=[i])
        images.loc[[i],'truelabel']    = -1

    for i in range(N,int(N+N/5)):

        images.loc[[i],'type']         = 'G'
        images.loc[[i],'labels']       = pd.Series([all_training_labels[i-N]],index=[i])
        images.loc[[i],'userIDs']      = pd.Series([all_training_userIDs[i-N]],index=[i])
        images.loc[[i],'truelabel']    = citizen_training_labels[i-N]

    dummy = random_permutation(range(250),int(N+N/5))

    for i in range(int(N+N/5)):
        images.loc[[i],'imageID'] = dummy[i]

    conf_matrices  = pd.DataFrame({ 'userID' : conf_userIDs,'conf_matrix' : conf_matrices})

    return images,conf_matrices

images,conf_matrices = gen_data()
