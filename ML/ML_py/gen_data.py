#gen_data script by CIERA Intern Group, 7/8/16

#import modules
import numpy as np
import itertools as it
from scipy.io import loadmat
from pdb import set_trace

#function to generate one batch of test data for use with gravspy_main2.py

#initialization
N = 100 #100 images
R = 30 #30 citizens
C = 15 #15 classes

#simulate training labels and true labels for test data
true_labels = (np.random.randint(1,high=C+1,size=(1,N)))[0] #generate 1xN array of numbers 1 to C, corresponding to true labels of images
training_labels = (np.random.randint(1,high=C+1,size=(1,int(N/5))))[0] #generate 1x(N/5) array of numbers 1 to C, corresponding to citizen labels of images

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
conf_matrices = {} #create empty dict

for k in range(R): #iterate over citizens

  conf_matrix = np.zeros((C,C)) #create empty CxC matrix
  
  for ii in range(C): #iterate over rows
  
    for jj in range(C): #iterate over columns
    
      if ii == jj: #if diagonal of matrix
      
        conf_matrix[ii,jj] = 180 + np.random.randint(0,high=41) #assign value 180 to 220
      
      else:
      
        conf_matrix[ii,jj] = np.random.randint(1,high=6) #assign value 1 to 5
        
  conf_matrices[k] = conf_matrix #map userID to corresponding conf_matrix
  
#simulate citizen labels and associated userID's
for i in range(N): #iterate over images
  
  labels = [] #create empty list