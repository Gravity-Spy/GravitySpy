#gravspy_main script by Luke Calian, 6/9/16

#before running execute generate_toy_data_trainingandtest in matlab, save variables as data.mat
#import scipy and use to read data, stored as dict mapping variable names to arrays
import scipy.io as sio
data = sio.loadmat('data.mat')

#import numpy for matrix generation
import numpy as np

#define t, threshold vector
t = .4*np.ones((data['C'][0][0],1))

#define R, citizen limit
R_lim = 30

#calculate prior probability of each image
no_labels = np.histogram((data['true_labels'][0]),np.unique((data['true_labels'][0])))
priors = no_labels[0]/len(data['true_labels'][0])

#define N, number of images in batch
N = len(data['images'])

#main loop to process images
for i in range(N): #iterate over images
  
  if data['images'][i]['type'][0][0] == 'G': #check if training image
    labels = data['images'][i]['labels'][0][0] #take citizen labels of image
    IDs = data['images'][i]['IDs'][0][0] #take IDs of citizens who label image
    tlabel = data['images'][i]['truelabel'][0][0][0] #take true label of image
    
    for ii in range(len(IDs)): #iterate over IDs of image
      conf_matrix = data['conf_matrices'][IDs[ii]-1][0] #take confusion matrix of citizen
      conf_matrix[tlabel,labels[ii]-2] = conf_matrix[tlabel,labels[ii]-1] #update confusion matrix
      data['conf_matrices'][IDs[ii]-1][0] = conf_matrix #confusion matrix put back in stack
    
    #decision(i) = 0; Since it is a raining image, there is no decision class(i) = tlabel; how do I do this?
    print('The image is from the training set')
  
  else:
    labels = data['images'][i]['labels'][0][0] #take citizen labels of image
    IDs = data['images'][i]['IDs'][0][0] #take IDs of citizens who label image
    no_annotators = len(labels) #define number of citizens who annotate image
    ML_dec = data['images'][i]['ML_posterior'][0][0] #take ML posteriors of image
    
    #for j in range(data['C'][0][0]): #iterate over classes
      #for k in range(no_annotators): #iterate over citizens that labeled image
        #conf = data['conf_matrices'][IDs[data['k']-1]] #take confusion matrix of citizen
        