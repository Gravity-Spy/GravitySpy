#gravspy_main script by Luke Calian, 6/9/16

"""The Zooniverse server (specifically Nero https://github.com/zooniverse/nero)
will continuously send data containing the following classification
information:

  The ID of the user, the ID of the image they classified, and 
  the classification made by that user for that image. Potentially,
  information concerning whether this image was a golden set image, that
  is an image where the label was known a head of time, versus a ML
  classified images, that is an image where the label is only the
  confidence level of the ML classifier, will also be sent in the message
  from Nero.

This information will then be parsed and saved in the following format.
A structure array with rows relating to a given image and columns 
containing the following information about that image:

      The Type - A label (string) either 'T' or 'G' to determine if it is a ML
      classified label or a pre-labelled "golden" image.

      The Labels - An array (double) of a 1XN row vector where N is the number
      of labels this image has been given at a certain time. Each column
      is a different answer that is associated with a different user.
      This takes us to the next column...

      The User IDs - An array (double) of a  1XN row vector where N is the number
      of labels this image has been given. Each column
      is the userID associated with the answer given in The Labels
      column.

      ML Posterior - An array (double) of a  1XC row vector where C is
      the number of pre-determined morphologies that the classifier has
      been trained on. Each column is the ML confidence that the image
      belongs in one of the C classes.

      True Label - (int) For images labelled 'T' this values is set to -1
      but for images labelled 'G' This value indicates the "true" class
      that this image belongs in for the purposes of comparing a citizens
      classification with this true label.

Once data for X amount of images has been parsed and stored in the above
format we will run the following script to update the retirability of an
image as well as the skill level of the citizens.

In addition to the above information, we will also store information on
the "confusion matrix" of a user. This information (stored in a .dat
file) will be a NX1 array where N is the number of users. in each row we
have a cell array that contains the CXC "confusion matrix" for that user.
A perfectly skilled user would only have values on the diagonal of this
matrix and all off diagonal values indicate wrong answers were given to
one category or another when presented with a 'G' true labelled image."""

#before running execute generate_toy_data_trainingandtest in matlab, save variables as data.mat
#import scipy and use to read data, stored as dict mapping variable names to arrays

import scipy.io as sio
data = sio.loadmat('data.mat')

#import numpy for matrix generation
import numpy as np

#define t, threshold vector
t = .4*np.ones((data['C'][0][0],1))
"""Initialize varaible t. t is a CX1 column vector where C is
the number of pre-determined morphologies and where each row is the
predetermined certainty threshold that an image most surpass to be
considered part of class C. Here all classes have the same threshold but
in realty different categories will have more difficult or more relax
thresholds for determination of class and therefore retirability."""

"""Initialize T_alpha, T_alpha is a CX1 column vector where C is
the number of pre-determined morphologies and where each row is the
predetermined "skill" level threshold for each class of glitch that a user
must surprass in order to move on to the next user level (Levels are
B1-B4, Apprentice, and Master).  Here all classes have the same threshold but
in realty different categories will have more difficult or more relax
thresholds based on how challenging a given class is."""

#define R, citizen limit
R_lim = 30
"""The citizen limit refers to
the max amount of citizens who can look at an image in a given level
before it is based on to the higher levels for more analysis. The idea is
that if an image's retirability cannot be determined from 30 labels then
this image needs to be looked at by more skilled users or LIGO experts."""

#calculate prior probability of each image
no_labels = np.histogram((data['true_labels'][0]),np.unique((data['true_labels'][0])))
priors = no_labels[1]/len(data['true_labels'][0])

#define N, number of images in batch
N = len(data['images'])

#define dec_matrix, matrix of each image's decision
dec_matrix = np.zeros([1,N])

#define class_matrix, matrix of each decision's class
class_matrix = np.zeros([1,N])

#main loop to process images
for i in range(N): #iterate over images
  
  if data['images'][i]['type'][0][0] == 'G': #check if training image
    labels = data['images'][i]['labels'][0][0] #take citizen labels of image
    IDs = data['images'][i]['IDs'][0][0] #take IDs of citizens who label image
    tlabel = data['images'][i]['truelabel'][0][0][0] #take true label of image
    
    for ii in range(len(IDs)): #iterate over IDs of image
      conf_matrix = data['conf_matrices'][IDs[ii]-1][0] #take confusion matrix of citizen
      conf_matrix[tlabel-1,labels[ii]-1] = conf_matrix[tlabel-1,labels[ii]-1]+1 #update confusion matrix
      data['conf_matrices'][IDs[ii]-1][0] = conf_matrix #confusion matrix put back in stack
    
    dec_matrix[0,i] = 0
    class_matrix[0,i] = tlabel
    print('The image is from the training set')
  
  else:
    labels = data['images'][i]['labels'][0][0] #take citizen labels of image
    IDs = data['images'][i]['IDs'][0][0] #take IDs of citizens who label image
    no_annotators = len(labels) #define number of citizens who annotate image
    ML_dec = data['images'][i]['ML_posterior'][0][0] #take ML posteriors of image
    
    for j in range(1,data['C'][0][0]+1): #iterate over classes
      for k in range(1,no_annotators+1): #iterate over citizens that labeled image
        conf = data['conf_matrices'][IDs[k-1]-1][0] #take confusion matrix of citizen
        conf_divided = np.diag(sum(conf,2))/conf #calculate p(l|j) value
        pp_matrix = np.zeros([data['C'][0][0],no_annotators]) #create posterior matrix
        #import pdb
        #pdb.set_trace()
        pp_matrix[j,k] = (conf_divided[j-1,labels[k-1]]*priors[j-1])/sum(conf_divided[:,labels[k-1]]*priors) #calculate posteriors